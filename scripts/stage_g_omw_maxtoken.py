"""Stage G — 2D grid sweep on SGL_MAX_TOTAL_TOKENS × client concurrency.

OCR_MAX_WORKERS is locked at 16 (validated as the winner in the earlier
3×4 OMW×MAX_TOKENS sweep — see the stage-G report at
loadtest/results/omnidoc-20260419-221959-stage-g.md). This re-shape
holds OMW fixed and varies client-side concurrency so we can see how
the same KV-cache budget handles different offered load.

Pool is restricted to datasets/OmniDocBench/images/ (excludes
data_diversity.png, which isn't representative).

Each cell: N=100, probe captured, abort at 15% (slightly loose so we
still see the full shape of cells near the gate).

Iteration order: SGL_MAX_TOTAL_TOKENS is the OUTER axis so we only
incur one sglang recreate per value change (concurrency changes are
client-side only, no container restart).
"""
import os
import sys
import time
import json
import pathlib
import tempfile
import subprocess
from datetime import datetime

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ.setdefault("PROBE_VRAM_TOTAL_MB", "8192")

sys.path.insert(0, str(pathlib.Path("scripts").resolve()))
import tune_params as tp

OMW            = 16                               # fixed — validated winner
CONCURRENCIES  = [8, 12, 16]                      # inner axis — client-side
TOTAL_TOKENS   = [30000, 100000, 200000]          # outer axis — sglang recreate
N              = 100
MAX_FAIL_RATE  = 0.15
MIN_SAMPLE     = 40
POOL_SEED      = 42

RUN_ID = f"omnidoc-{datetime.now().strftime('%Y%m%d-%H%M%S')}-stage-g"
RAW = pathlib.Path("loadtest/results/raw") / RUN_ID
RAW.mkdir(parents=True, exist_ok=True)
print(f"[stage-g] run_id = {RUN_ID}", flush=True)
print(f"[stage-g] grid: MAX_TOKENS ∈ {TOTAL_TOKENS} × c ∈ {CONCURRENCIES}  (OMW fixed at {OMW})", flush=True)
print(f"[stage-g] per-cell: N={N}, abort@{MAX_FAIL_RATE:.0%}", flush=True)

tmp = pathlib.Path(tempfile.mkdtemp(prefix="stage-g-"))
urls_file = tp.build_image_pool(128, tmp, seed=POOL_SEED)
print(f"[stage-g] pool: {urls_file.read_text().count(chr(10))} images from images/ subdir", flush=True)

tracked = list(tp.KNOBS.keys()) + ["CPU_WORKERS", "CPU_THREADS"]
current = tp.snapshot_env(tracked)
print(f"[stage-g] baseline: {current}", flush=True)

trials = []
total_cells = len(TOTAL_TOKENS) * len(CONCURRENCIES)
idx = 0

# Pin OMW once up front — it's constant across cells so no need to re-apply.
current = tp.apply_knobs({**current, "OCR_MAX_WORKERS": str(OMW)}, current)

for max_tokens in TOTAL_TOKENS:
    # Apply per max_tokens outside the inner loop so we only restart sglang
    # once per outer-axis value.
    target = {**current, "SGL_MAX_TOTAL_TOKENS": str(max_tokens)}
    try:
        current = tp.apply_knobs(target, current)
    except RuntimeError as e:
        print(f"[stage-g] services not healthy: {e}", flush=True)
        continue

    for c in CONCURRENCIES:
        idx += 1
        tag = f"[{idx}/{total_cells} {100*idx/total_cells:.0f}%]"
        print(f"\n{'='*70}", flush=True)
        print(f"{tag} MAX_TOKENS={max_tokens}  c={c}  (OMW={OMW})", flush=True)
        print(f"{'='*70}", flush=True)

        bench_json  = RAW / f"g-mt{max_tokens}-c{c:02d}.json"
        probe_jsonl = RAW / f"probe-mt{max_tokens}-c{c:02d}.jsonl"

        probe = tp._probe_spawn(probe_jsonl)
        time.sleep(4)
        try:
            summary = tp.run_bench(
                c, N, urls_file, bench_json,
                pool_seed=POOL_SEED,
                max_fail_rate=MAX_FAIL_RATE,
                min_sample_for_abort=MIN_SAMPLE,
            )
        finally:
            tp._probe_reap(probe)

        probe_sum = tp._probe_aggregate(probe_jsonl)
        util = tp._utilization_ratios(probe_sum, {
            **current,
            "OCR_MAX_WORKERS": OMW,
            "SGL_MAX_RUNNING_REQUESTS": current.get("SGL_MAX_RUNNING_REQUESTS", 32),
        })
        bneck = tp._classify_bottleneck(probe_sum, {
            **current,
            "OCR_MAX_WORKERS": OMW,
            "SGL_MAX_RUNNING_REQUESTS": current.get("SGL_MAX_RUNNING_REQUESTS", 32),
        })

        summary["probe_summary"] = probe_sum
        summary["utilization"] = util
        summary["bottleneck"] = bneck
        summary["omw"] = OMW
        summary["max_total_tokens"] = max_tokens
        summary["concurrency"] = c
        bench_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        att = summary.get("requests_attempted") or summary.get("total")
        fr = (summary.get("failures", 0) / att) if att else 0
        lat = summary.get("latency_ms", {})
        print(
            f"{tag} -> rps={summary.get('throughput_rps', 0):.3f} "
            f"fail={fr:.0%} p50={lat.get('p50', 0):.0f}ms "
            f"p95={lat.get('p95', 0):.0f}ms p99={lat.get('p99', 0):.0f}ms "
            f"mean={lat.get('mean', 0):.0f}ms  "
            f"GPU%={util.get('gpu_compute', 0)*100:.0f} "
            f"VRAM%={util.get('gpu_memory', 0)*100:.0f} "
            f"batch%={util.get('sgl_batch', 0)*100:.0f}  "
            f"{bneck}  aborted={summary.get('aborted', False)}",
            flush=True,
        )

        trials.append({
            "kind": "stage-g",
            "omw": OMW,
            "sgl": int(current.get("SGL_MAX_RUNNING_REQUESTS", 32)),
            "cpu_w": str(current.get("CPU_WORKERS", "?")),
            "cpu_t": str(current.get("CPU_THREADS", "?")),
            "knobs": {
                "OCR_MAX_WORKERS": str(OMW),
                "SGL_MAX_TOTAL_TOKENS": str(max_tokens),
                "concurrency": str(c),
                "bottleneck": bneck,
            },
            "summary": summary,
        })
        (RAW / "_trials.json").write_text(json.dumps(trials, indent=2), encoding="utf-8")

# Final summary table
print(f"\n{'='*70}\n[stage-g] GRID RESULTS\n{'='*70}", flush=True)
print(f"{'MAX_TOKENS':>11} {'c':>3} {'rps':>6} {'fail%':>6} {'p50':>5} {'p95':>5} {'p99':>5} {'mean':>5} "
      f"{'GPU%':>5} {'VRAM%':>6} {'batch%':>7} {'SLO':>4}  bottleneck", flush=True)
for t in trials:
    s = t["summary"]; u = s["utilization"]; lat = s["latency_ms"]
    att = s.get("requests_attempted") or s.get("total")
    fr = (s.get("failures", 0) / att) if att else 0
    slo = "OK" if (lat.get("p99", 99999) <= 120000 and fr <= 0.10 and not s.get("aborted")) else "miss"
    print(f"{s['max_total_tokens']:>11} {s['concurrency']:>3} {s['throughput_rps']:>6.3f} {fr:>6.0%} "
          f"{lat.get('p50', 0)/1000:>5.0f} {lat.get('p95', 0)/1000:>5.0f} {lat.get('p99', 0)/1000:>5.0f} "
          f"{lat.get('mean', 0)/1000:>5.0f} {u['gpu_compute']*100:>5.0f} "
          f"{u['gpu_memory']*100:>6.0f} {u['sgl_batch']*100:>7.0f} {slo:>4}  {s['bottleneck']}",
          flush=True)

print(f"\n[stage-g] trials.json: {RAW / '_trials.json'}", flush=True)
