"""Fast matrix sweep for the shipped paddle2onnx + OV EP stack.

Runs N=50 pages per concurrency level across {8, 16, 24, 32}. For each
level captures: rps, mean/p50/p95 request latency, empty-markdown rate,
and per-stage breakdown (layout / OCR region / regions-per-request)
from /metrics histogram deltas bracketing the run. Outputs a JSON
summary that a downstream report.md script can render.

Not a replacement for the full asyncio-matrix harness — that one
computes more statistics and handles warmup, cooldown, gap-based
segmentation. This is purposely compact so it runs in ≤10 min and
refreshes the key numbers after a backend change.

Usage (from host, with stack up):
    PYTHONIOENCODING=utf-8 python scripts/matrix_sweep_quick.py [--out path]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
import time
import urllib.request
from pathlib import Path

SEED = 42
N_PER_LEVEL = 50
CONCURRENCIES = (8, 16, 24, 32)
ENDPOINT = "http://localhost:5002/glmocr/parse"
METRICS = "http://localhost:5002/metrics"
CONTAINER_PREFIX = "file:///app/datasets/OmniDocBench/images"
IMAGES_DIR = Path(__file__).resolve().parents[1] / "datasets" / "OmniDocBench" / "images"


def fetch_metrics() -> dict[str, float]:
    txt = urllib.request.urlopen(METRICS, timeout=10).read().decode()
    out: dict[str, float] = {}
    for line in txt.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r'^([a-zA-Z0-9_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$', line)
        if not m:
            continue
        out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
    return out


def parse_stage_stats(m0: dict, m1: dict) -> dict[str, dict]:
    def delta(name: str, scope: str | None = None) -> tuple[float, float]:
        def grab(suffix: str) -> tuple[float, float]:
            t0, t1 = 0.0, 0.0
            for k, v in m0.items():
                if k.startswith(name) and k.endswith(suffix) and (scope is None or scope in k):
                    t0 += v
            for k, v in m1.items():
                if k.startswith(name) and k.endswith(suffix) and (scope is None or scope in k):
                    t1 += v
            return t0, t1
        s0, s1 = grab("_sum")
        c0, c1 = grab("_count")
        return s1 - s0, c1 - c0

    fs_s, fs_n = delta("flask_http_request_duration_seconds", 'url_rule="/glmocr/parse"')
    la_s, la_n = delta("glmocr_layout_seconds", None)
    oc_s, oc_n = delta("glmocr_ocr_region_seconds", None)
    return {
        "flask_parse":       {"sum_s": fs_s, "count": fs_n, "mean_s_per_req": (fs_s / fs_n) if fs_n else 0},
        "layout":            {"sum_s": la_s, "count": la_n, "mean_s_per_call": (la_s / la_n) if la_n else 0},
        "ocr_region":        {"sum_s": oc_s, "count": oc_n, "mean_s_per_region": (oc_s / oc_n) if oc_n else 0,
                              "regions_per_req": (oc_n / fs_n) if fs_n else 0},
    }


def one_request(image_name: str, timeout: float = 240.0) -> tuple[int, float, int, int]:
    body = json.dumps({"images": [f"{CONTAINER_PREFIX}/{image_name}"]}).encode()
    req = urllib.request.Request(ENDPOINT, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        dt = time.perf_counter() - t
        md = d.get("markdown_result") or ""
        jr = d.get("json_result") or [[]]
        return r.status, dt, len(md), len(jr[0]) if jr and jr[0] else 0
    except Exception as e:
        return 0, time.perf_counter() - t, 0, 0


def run_level(concurrency: int, n: int, rng: random.Random) -> dict:
    all_imgs = sorted(IMAGES_DIR.iterdir())
    picks = [all_imgs[rng.randrange(len(all_imgs))].name for _ in range(n)]

    # Warmup: fire a handful serially so workers are hot before the measured run
    for _ in range(min(3, n)):
        one_request(picks[0], timeout=60)

    m0 = fetch_metrics()
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(one_request, picks))
    wall = time.perf_counter() - t0
    m1 = fetch_metrics()

    stats_ok = [r[1] for r in results if r[0] == 200 and r[2] > 0]
    empties = sum(1 for r in results if r[0] == 200 and r[2] == 0)
    errors = sum(1 for r in results if r[0] != 200)
    blocks = [r[3] for r in results if r[0] == 200 and r[2] > 0]

    stage = parse_stage_stats(m0, m1)
    sorted_lat = sorted(stats_ok)

    return {
        "concurrency": concurrency,
        "n_requests": n,
        "wall_s": wall,
        "rps": n / wall if wall else 0,
        "client": {
            "ok": len(stats_ok),
            "empty_markdown": empties,
            "errors": errors,
            "mean_s": sum(stats_ok) / len(stats_ok) if stats_ok else 0,
            "p50_s": sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0,
            "p95_s": sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else 0,
            "p99_s": sorted_lat[int(len(sorted_lat) * 0.99)] if sorted_lat else 0,
            "blocks_mean": sum(blocks) / len(blocks) if blocks else 0,
        },
        "server": stage,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("loadtest/results/matrix-2026-04-24-paddle-ov.json"))
    ap.add_argument("--n-per-level", type=int, default=N_PER_LEVEL)
    ap.add_argument("--concurrencies", type=int, nargs="+", default=list(CONCURRENCIES))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    started = time.time()
    results = []
    for c in args.concurrencies:
        print(f"\n=== c={c} n={args.n_per_level} ===", flush=True)
        r = run_level(c, args.n_per_level, rng)
        results.append(r)
        client = r["client"]
        srv = r["server"]
        print(f"  wall={r['wall_s']:.1f}s  rps={r['rps']:.2f}  "
              f"ok={client['ok']}/{r['n_requests']} empty={client['empty_markdown']} err={client['errors']}",
              flush=True)
        print(f"  latency  mean={client['mean_s']:.2f}s  p50={client['p50_s']:.2f}s  p95={client['p95_s']:.2f}s  p99={client['p99_s']:.2f}s",
              flush=True)
        print(f"  layout {srv['layout']['mean_s_per_call']:.2f}s/call  "
              f"ocr_region {srv['ocr_region']['mean_s_per_region']:.2f}s/reg ({srv['ocr_region']['regions_per_req']:.1f} reg/req)  "
              f"blocks_mean={client['blocks_mean']:.1f}",
              flush=True)

    summary = {
        "seed": SEED,
        "started_unix": started,
        "host": "localhost:5002",
        "env": {
            "LAYOUT_VARIANT": "paddle2onnx",
            "LAYOUT_ONNX_PROVIDER": "openvino",
            "LAYOUT_BATCH_ENABLED": "true",
        },
        "levels": results,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {args.out}", flush=True)


if __name__ == "__main__":
    main()
