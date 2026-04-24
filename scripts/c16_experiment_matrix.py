"""Run single-knob experiments and report TTFT deltas with multi-rep averaging.

Each experiment: apply env override → restart affected container → wait
healthy → fire N requests @ c `reps` times → capture CPU + SGLang metrics →
revert env → next.

The multi-rep design (`--reps N`) is load-bearing: single-burst numbers
at c≥16 have ±150% rps noise on this 8 GB card per the session's
feedback_matrix_noise.md update. By running each config N times
back-to-back (NO env revert between reps, so cache state is comparable)
and reporting median + IQR, we get signal on TTFT/prefix-cache even
at c=16.

TTFT-related metrics have much tighter distributions than end-to-end
rps (they're per-region histograms, not wall-to-count ratios), so
this harness ranks configs on `ttft_mean_s` / `decode_only_s` /
`prefix_cache_hit_pct` — NOT rps.

Baseline (shipped 2026-04-24):
  LAYOUT_VARIANT=paddle2onnx, LAYOUT_ONNX_PROVIDER=openvino,
  LAYOUT_BATCH_ENABLED=true, LAYOUT_PREFIX_PIN=true,
  SGL_MEM_FRACTION_STATIC=0.83, SGL_MAX_RUNNING_REQUESTS=64,
  SGL_CUDA_GRAPH_MAX_BS=(unset/default), OCR_MAX_WORKERS=32,
  LAYOUT_BATCH_WINDOW_MS=20

Usage (from repo root, stack up):
    PYTHONIOENCODING=utf-8 python scripts/c16_experiment_matrix.py
    PYTHONIOENCODING=utf-8 python scripts/c16_experiment_matrix.py --reps 5 --c 8 --n 20
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
import statistics
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
DEFAULT_REPS = 3

EXPERIMENTS = [
    # label,                                  env overrides,                                             container(s) to restart
    ("baseline",                              {},                                                         []),
    # Prior session's single-burst matrix (kept for reference; don't re-run unless at high reps)
    ("E1: mem=0.95 + max_running=16",         {"SGL_MEM_FRACTION_STATIC": "0.95", "SGL_MAX_RUNNING_REQUESTS": "16"}, ["sglang"]),
    ("E2: cuda_graph_max_bs=16",              {"SGL_CUDA_GRAPH_MAX_BS": "16"},                           ["sglang"]),
    ("E3: ocr_max_workers=16",                {"OCR_MAX_WORKERS": "16"},                                 ["cpu"]),
    ("E4: layout_batch_window_ms=50",         {"LAYOUT_BATCH_WINDOW_MS": "50"},                          ["cpu"]),
    # TTFT-reduction plan Item 2: image-token shrink sweep.
    # Smaller max_pixels → fewer image tokens per region → less prefill work.
    ("I2: max_pixels=921600",                 {"PAGE_LOADER_MAX_PIXELS": "921600"},                      ["cpu"]),
    ("I2: max_pixels=589824",                 {"PAGE_LOADER_MAX_PIXELS": "589824"},                      ["cpu"]),
    ("I2: max_pixels=262144",                 {"PAGE_LOADER_MAX_PIXELS": "262144"},                      ["cpu"]),
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


def _pick(reps: list[dict], *path: str) -> list[float]:
    """Extract a nested metric across reps, e.g. _pick(reps, 'sgl_stage', 'ttft_mean_s')."""
    out = []
    for r in reps:
        cur = r
        for k in path:
            cur = cur.get(k, {}) if isinstance(cur, dict) else 0
        if isinstance(cur, (int, float)):
            out.append(float(cur))
    return out


def agg(values: list[float]) -> dict[str, float]:
    """median + IQR + min/max over a list of per-rep measurements."""
    if not values:
        return {"median": 0, "p25": 0, "p75": 0, "min": 0, "max": 0, "n": 0}
    s = sorted(values)
    n = len(s)
    return {
        "median": statistics.median(s),
        "p25": s[max(0, int(n * 0.25))] if n > 1 else s[0],
        "p75": s[min(n - 1, int(n * 0.75))] if n > 1 else s[0],
        "min": s[0],
        "max": s[-1],
        "n": n,
    }


def pretty_rep(label: str, rep_idx: int, r: dict) -> None:
    cl, cpu, sgl = r["client"], r["cpu_stage"], r["sgl_stage"]
    print(
        f"  rep {rep_idx+1}: rps={cl['rps']:5.2f}  mean={cl['mean_s']:5.2f}s  "
        f"TTFT={sgl['ttft_mean_s']:5.2f}s  decode={sgl['decode_only_s']:4.2f}s  "
        f"hit={sgl['prefix_cache_hit_pct']:4.1f}%  "
        f"layout={cpu['layout_mean_s']:4.1f}s  ok={cl['ok']}/{cl['n']} empty={cl['empty']}",
        flush=True,
    )


def pretty_agg(label: str, reps: list[dict]) -> None:
    ttft = agg(_pick(reps, "sgl_stage", "ttft_mean_s"))
    decode = agg(_pick(reps, "sgl_stage", "decode_only_s"))
    hit = agg(_pick(reps, "sgl_stage", "prefix_cache_hit_pct"))
    layout = agg(_pick(reps, "cpu_stage", "layout_mean_s"))
    rps = agg(_pick(reps, "client", "rps"))
    mean_s = agg(_pick(reps, "client", "mean_s"))
    empty = sum(_pick(reps, "client", "empty"))
    print(
        f"  [median±iqr over {len(reps)} reps]  "
        f"TTFT={ttft['median']:.2f}s (p25={ttft['p25']:.2f}, p75={ttft['p75']:.2f})  "
        f"decode={decode['median']:.2f}s  "
        f"hit={hit['median']:.1f}%  "
        f"layout={layout['median']:.2f}s  "
        f"rps={rps['median']:.2f}  mean={mean_s['median']:.2f}s  "
        f"total_empty={int(empty)}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS,
                    help="measurements per experiment; median+IQR reported across these")
    ap.add_argument("--c", type=int, default=CONCURRENCY, help="concurrency level")
    ap.add_argument("--n", type=int, default=N, help="requests per measurement")
    ap.add_argument("--out", type=Path,
                    default=REPO / "loadtest" / "results" / "c16-experiment-matrix-2026-04-24.json")
    ap.add_argument("--experiments-only", type=str, default=None,
                    help="comma-separated experiment label prefixes to run (default: all)")
    args = ap.parse_args()

    filt = None
    if args.experiments_only:
        filt = [s.strip() for s in args.experiments_only.split(",") if s.strip()]

    print(f"[matrix] c={args.c} n={args.n} reps={args.reps} seed={SEED}")
    if filt:
        print(f"[matrix] filtered experiments: {filt}")

    results: dict[str, dict] = {}

    for label, overrides, containers in EXPERIMENTS:
        if filt and not any(label.startswith(f) for f in filt):
            continue
        print(f"\n=== {label} ===", flush=True)
        prior = patch_env(overrides) if overrides else {}
        try:
            for svc in containers:
                print(f"  restarting {svc}...", flush=True)
                restart(svc)
                wait_healthy(CPU_HEALTH if svc == "cpu" else SGL_HEALTH, timeout=900)
            if not containers:
                print("  (no restart needed)", flush=True)
            time.sleep(5)

            # Multi-rep loop — NO env revert between reps, NO restart between reps.
            # Cache state evolves naturally within a config; we measure the
            # distribution of TTFT / decode / hit across reps.
            reps: list[dict] = []
            for i in range(args.reps):
                r = measure(args.n, args.c)
                pretty_rep(label, i, r)
                reps.append(r)
            results[label] = {
                "overrides": overrides,
                "containers": containers,
                "reps": reps,
                "agg": {
                    "ttft_mean_s":         agg(_pick(reps, "sgl_stage", "ttft_mean_s")),
                    "decode_only_s":       agg(_pick(reps, "sgl_stage", "decode_only_s")),
                    "prefix_cache_hit_pct": agg(_pick(reps, "sgl_stage", "prefix_cache_hit_pct")),
                    "layout_mean_s":       agg(_pick(reps, "cpu_stage", "layout_mean_s")),
                    "ocr_region_mean_s":   agg(_pick(reps, "cpu_stage", "ocr_region_mean_s")),
                    "rps":                 agg(_pick(reps, "client", "rps")),
                    "mean_s":              agg(_pick(reps, "client", "mean_s")),
                    "p95_s":               agg(_pick(reps, "client", "p95_s")),
                    "empty_total":         sum(_pick(reps, "client", "empty")),
                },
            }
            pretty_agg(label, reps)
        finally:
            if overrides:
                restore_env(prior)
                for svc in containers:
                    print(f"  reverting {svc}...", flush=True)
                    restart(svc)
                    wait_healthy(CPU_HEALTH if svc == "cpu" else SGL_HEALTH, timeout=900)

    # Summary — rank by TTFT median (primary shipping metric)
    print("\n=== SUMMARY — ranked by TTFT median ===")
    print(f"{'experiment':<40s} {'TTFT (med/IQR)':>22s} {'decode':>8s} {'hit%':>7s} {'rps':>6s} {'mean':>8s} {'empty':>6s}")
    base_ttft = None
    ranked = sorted(results.items(), key=lambda kv: kv[1]["agg"]["ttft_mean_s"]["median"])
    for label, r in ranked:
        a = r["agg"]
        if label == "baseline":
            base_ttft = a["ttft_mean_s"]["median"]
        delta = ""
        if base_ttft and label != "baseline" and base_ttft > 0:
            pct = (a["ttft_mean_s"]["median"] - base_ttft) / base_ttft * 100
            delta = f" ({pct:+.0f}% vs base)"
        ttft_str = f"{a['ttft_mean_s']['median']:.2f}/±{(a['ttft_mean_s']['p75']-a['ttft_mean_s']['p25'])/2:.2f}"
        print(f"{label:<40s} {ttft_str:>22s} {a['decode_only_s']['median']:>7.2f}s {a['prefix_cache_hit_pct']['median']:>6.1f}% {a['rps']['median']:>6.2f} {a['mean_s']['median']:>7.2f}s {int(a['empty_total']):>6d}{delta}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()
