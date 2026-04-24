"""Run c=16 through a sequence of single-knob experiments and report deltas.

Each experiment: apply env override → restart affected container → wait
healthy → fire 20 requests @ c=16 → capture CPU + SGLang metrics → revert
the env change → next.

Baseline (shipped 2026-04-24):
  LAYOUT_VARIANT=paddle2onnx, LAYOUT_ONNX_PROVIDER=openvino,
  LAYOUT_BATCH_ENABLED=true, LAYOUT_PREFIX_PIN=true,
  SGL_MEM_FRACTION_STATIC=0.83, SGL_MAX_RUNNING_REQUESTS=64,
  SGL_CUDA_GRAPH_MAX_BS=(unset/default), OCR_MAX_WORKERS=32,
  LAYOUT_BATCH_WINDOW_MS=20

Usage (from repo root, stack up):
    PYTHONIOENCODING=utf-8 python scripts/c16_experiment_matrix.py
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import random
import re
import subprocess
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV_PATH = REPO / ".env"
IMAGES_DIR = REPO / "datasets" / "OmniDocBench" / "images"
CPU_ENDPOINT = "http://localhost:5002/glmocr/parse"
CPU_METRICS = "http://localhost:5002/metrics"
SGL_METRICS = "http://localhost:30000/metrics"
CPU_HEALTH = "http://localhost:5002/health"
SGL_HEALTH = "http://localhost:30000/health"
N = 20
CONCURRENCY = 16
SEED = 42

EXPERIMENTS = [
    # label,                                  env overrides,                                             container(s) to restart
    ("baseline",                              {},                                                         []),
    ("E1: mem=0.95 + max_running=16",         {"SGL_MEM_FRACTION_STATIC": "0.95", "SGL_MAX_RUNNING_REQUESTS": "16"}, ["sglang"]),
    ("E2: cuda_graph_max_bs=16",              {"SGL_CUDA_GRAPH_MAX_BS": "16"},                           ["sglang"]),
    ("E3: ocr_max_workers=16",                {"OCR_MAX_WORKERS": "16"},                                 ["cpu"]),
    ("E4: layout_batch_window_ms=50",         {"LAYOUT_BATCH_WINDOW_MS": "50"},                          ["cpu"]),
]


def read_env() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" in s:
            k, v = s.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def patch_env(overrides: dict[str, str], restore: bool = False) -> dict[str, str | None]:
    """Apply overrides to .env, return the prior values so we can restore."""
    text = ENV_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    prior: dict[str, str | None] = {}
    new_lines = list(lines)
    for key, value in overrides.items():
        found = False
        for i, line in enumerate(new_lines):
            if line.startswith(f"{key}="):
                prior[key] = line.split("=", 1)[1]
                new_lines[i] = f"{key}={value}"
                found = True
                break
        if not found:
            prior[key] = None
            new_lines.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(new_lines) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")
    return prior


def restore_env(prior: dict[str, str | None]) -> None:
    text = ENV_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    new_lines: list[str] = []
    # First pass: update existing keys
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k = stripped.split("=", 1)[0]
            if k in prior:
                if prior[k] is None:
                    continue  # drop lines we appended
                new_lines.append(f"{k}={prior[k]}")
                continue
        new_lines.append(line)
    ENV_PATH.write_text("\n".join(new_lines) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")


def restart(service: str) -> None:
    subprocess.run(["docker", "compose", "up", "-d", service], check=False, capture_output=True)


def wait_healthy(url: str, timeout: float = 900) -> None:
    """Block until GET url returns HTTP 200. /health bodies are inconsistent
    (empty on sglang, {"status":"ok"} on cpu) so we don't match the body."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"{url} not healthy after {timeout}s")


def sgl_metrics() -> dict[str, float]:
    out = {"tok_prefill_compute": 0, "tok_prefill_cache": 0, "tok_decode": 0,
           "ttft_sum": 0, "ttft_count": 0, "e2e_sum": 0, "e2e_count": 0}
    t = urllib.request.urlopen(SGL_METRICS, timeout=5).read().decode()
    for line in t.splitlines():
        if line.startswith("#"):
            continue
        for pat, fmt in [
            (r'sglang:realtime_tokens_total\{[^}]*mode="([^"]+)"[^}]*\}\s+([0-9.eE+-]+)', "tok_{}"),
            (r'sglang:time_to_first_token_seconds_(sum|count)\{[^}]*\}\s+([0-9.eE+-]+)', "ttft_{}"),
            (r'sglang:e2e_request_latency_seconds_(sum|count)\{[^}]*\}\s+([0-9.eE+-]+)', "e2e_{}"),
        ]:
            m = re.match(pat, line)
            if m:
                out[fmt.format(m.group(1))] = float(m.group(2))
    return out


def cpu_metrics() -> dict[str, float]:
    out: dict[str, float] = {}
    t = urllib.request.urlopen(CPU_METRICS, timeout=5).read().decode()
    for line in t.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r'^(glmocr_layout_seconds|glmocr_ocr_region_seconds|flask_http_request_duration_seconds)(_sum|_count)(?:\{([^}]*)\})?\s+([0-9.eE+-]+)', line)
        if not m:
            continue
        name, suf, labels, v = m.group(1), m.group(2), m.group(3) or "", float(m.group(4))
        if name == "flask_http_request_duration_seconds" and "glmocr/parse" not in labels:
            continue
        out[name + suf] = out.get(name + suf, 0) + v
    return out


def one_request(image_name: str) -> tuple[float, int]:
    body = json.dumps({"images": [f"file:///app/datasets/OmniDocBench/images/{image_name}"]}).encode()
    req = urllib.request.Request(CPU_ENDPOINT, data=body, headers={"Content-Type": "application/json"}, method="POST")
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
        return time.perf_counter() - t, len(d.get("markdown_result") or "")
    except Exception:
        return time.perf_counter() - t, 0


def measure(n: int = N, c: int = CONCURRENCY) -> dict:
    rng = random.Random(SEED)
    all_imgs = sorted(IMAGES_DIR.iterdir())
    picks = [all_imgs[rng.randrange(len(all_imgs))].name for _ in range(n)]

    # Brief warmup: 3 serial requests
    for _ in range(3):
        one_request(picks[0])

    s0, c0 = sgl_metrics(), cpu_metrics()
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=c) as ex:
        results = list(ex.map(one_request, picks))
    wall = time.perf_counter() - t0
    s1, c1 = sgl_metrics(), cpu_metrics()

    ok = sum(1 for r in results if r[1] > 0)
    empties = sum(1 for r in results if r[1] == 0)
    lats = sorted([r[0] for r in results if r[1] > 0])

    # Helpers for deltas
    def d(d1, d0, key):
        return d1.get(key, 0) - d0.get(key, 0)

    req_n = d(c1, c0, "flask_http_request_duration_seconds_count")
    req_s = d(c1, c0, "flask_http_request_duration_seconds_sum")
    lay_n = d(c1, c0, "glmocr_layout_seconds_count")
    lay_s = d(c1, c0, "glmocr_layout_seconds_sum")
    ocr_n = d(c1, c0, "glmocr_ocr_region_seconds_count")
    ocr_s = d(c1, c0, "glmocr_ocr_region_seconds_sum")
    ttft_n = d(s1, s0, "ttft_count")
    ttft_s = d(s1, s0, "ttft_sum")
    e2e_n = d(s1, s0, "e2e_count")
    e2e_s = d(s1, s0, "e2e_sum")
    hit_tok = d(s1, s0, "tok_prefill_cache")
    compute_tok = d(s1, s0, "tok_prefill_compute")
    hit_pct = (hit_tok / (hit_tok + compute_tok) * 100) if (hit_tok + compute_tok) else 0

    return {
        "client": {
            "n": n, "c": c, "ok": ok, "empty": empties,
            "wall_s": wall, "rps": n / wall,
            "mean_s": sum(lats) / len(lats) if lats else 0,
            "p50_s": lats[len(lats) // 2] if lats else 0,
            "p95_s": lats[min(len(lats) - 1, int(len(lats) * 0.95))] if lats else 0,
        },
        "cpu_stage": {
            "req_mean_s": (req_s / req_n) if req_n else 0,
            "layout_mean_s": (lay_s / lay_n) if lay_n else 0,
            "ocr_region_mean_s": (ocr_s / ocr_n) if ocr_n else 0,
            "regions_per_req": (ocr_n / req_n) if req_n else 0,
        },
        "sgl_stage": {
            "ttft_mean_s": (ttft_s / ttft_n) if ttft_n else 0,
            "decode_only_s": ((e2e_s - ttft_s) / e2e_n) if e2e_n else 0,
            "prefix_cache_hit_pct": hit_pct,
        },
    }


def pretty(label: str, r: dict) -> None:
    cl, cpu, sgl = r["client"], r["cpu_stage"], r["sgl_stage"]
    print(
        f"{label:<40s}  rps={cl['rps']:5.2f}  "
        f"mean={cl['mean_s']:5.2f}s  p95={cl['p95_s']:5.2f}s  "
        f"ok={cl['ok']}/{cl['n']} empty={cl['empty']}  "
        f"layout={cpu['layout_mean_s']:4.1f}s  TTFT={sgl['ttft_mean_s']:4.1f}s  "
        f"decode={sgl['decode_only_s']:4.2f}s  hit={sgl['prefix_cache_hit_pct']:4.1f}%",
        flush=True,
    )


def main() -> None:
    print(f"[matrix] c={CONCURRENCY} n={N} seed={SEED}")
    # Before each experiment, the stack must already be at baseline.
    # We don't apply baseline env changes (it's a measurement-only step).
    results: dict[str, dict] = {}

    for label, overrides, containers in EXPERIMENTS:
        print(f"\n=== {label} ===", flush=True)
        prior = patch_env(overrides) if overrides else {}
        try:
            # Restart containers to pick up env
            for svc in containers:
                print(f"  restarting {svc}...", flush=True)
                restart(svc)
                wait_healthy(CPU_HEALTH if svc == "cpu" else SGL_HEALTH, timeout=600)
            if not containers:
                print("  (no restart needed)", flush=True)
            # brief settle
            time.sleep(5)
            r = measure(N, CONCURRENCY)
            results[label] = r
            pretty(label, r)
        finally:
            # Revert env back to baseline for next experiment
            if overrides:
                restore_env(prior)
                for svc in containers:
                    print(f"  reverting {svc}...", flush=True)
                    restart(svc)
                    wait_healthy(CPU_HEALTH if svc == "cpu" else SGL_HEALTH, timeout=600)

    # Summary
    print("\n=== SUMMARY ===")
    base = results.get("baseline", {}).get("client", {}).get("rps", 0)
    print(f"{'experiment':<40s} {'rps':>6s} {'Δ-rps':>7s} {'mean':>7s} {'TTFT':>6s} {'layout':>7s} {'hit%':>6s} {'empty':>6s}")
    for label, r in results.items():
        cl, cpu, sgl = r["client"], r["cpu_stage"], r["sgl_stage"]
        drps = ((cl["rps"] - base) / base * 100) if base and label != "baseline" else 0
        print(f"{label:<40s} {cl['rps']:>6.2f} {drps:>+6.1f}% {cl['mean_s']:>6.2f}s {sgl['ttft_mean_s']:>5.2f}s {cpu['layout_mean_s']:>6.2f}s {sgl['prefix_cache_hit_pct']:>5.1f}% {cl['empty']:>6d}")

    out = REPO / "loadtest" / "results" / "c16-experiment-matrix-2026-04-24.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
