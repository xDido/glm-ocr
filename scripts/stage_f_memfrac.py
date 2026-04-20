"""Stage F — targeted mem_fraction sweep at c=8 with spec decoding on.

Hypothesis: Stage E pinned VRAM at 85-94% on 8 GB. Lowering mem_fraction
from 0.88 leaves more room for KV cache, which should let SGLang admit
more concurrent running requests. Spec decoding stays on so we don't
lose the 3x decode speedup.

For each mem_fraction value:
  1. Write to .env, restart sglang + cpu, wait healthy.
  2. Capture the SGLang startup line that reports KV cache size (tells
     us whether the new fraction actually changed the allocation).
  3. Probe + bench at c=8, N=150 (enough to get past MIN_SAMPLE=40 for
     abort, short enough to iterate fast).
  4. Aggregate utilization + bottleneck; write extended cell JSON.

Output: loadtest/results/raw/omnidoc-<ts>-stage-f/ with f-mf<val>.json,
probe-mf<val>.jsonl, _trials.json + sglang-startup-mf<val>.log.
"""
import os
import re
import sys
import time
import json
import pathlib
import subprocess
import tempfile
from datetime import datetime

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ.setdefault("PROBE_VRAM_TOTAL_MB", "8192")   # real 8 GB card

sys.path.insert(0, str(pathlib.Path("scripts").resolve()))
import tune_params as tp

MEM_FRACTIONS = [0.82, 0.80, 0.78]
C = 8
N = 150
POOL_SEED = 42

RUN_ID = f"omnidoc-{datetime.now().strftime('%Y%m%d-%H%M%S')}-stage-f"
RAW = pathlib.Path("loadtest/results/raw") / RUN_ID
RAW.mkdir(parents=True, exist_ok=True)
LOG = pathlib.Path("loadtest/results") / f"{RUN_ID}-live.log"
print(f"[stage-f] run_id = {RUN_ID}", flush=True)
print(f"[stage-f] sweeping SGL_MEM_FRACTION_STATIC = {MEM_FRACTIONS}, c={C}, N={N}", flush=True)

# Build the pool ONCE — every cell uses the same draw.
tmp = pathlib.Path(tempfile.mkdtemp(prefix="stage-f-"))
urls_file = tp.build_image_pool(128, tmp, seed=POOL_SEED)

# Baseline snapshot — apply_knobs needs it.
tracked = list(tp.KNOBS.keys()) + ["CPU_WORKERS", "CPU_THREADS"]
current = tp.snapshot_env(tracked)
print(f"[stage-f] baseline: {current}", flush=True)

KV_RE = re.compile(r"KV Cache|#tokens|max_total_tokens|memory budget|\bkv\b", re.IGNORECASE)

trials = []
for mf in MEM_FRACTIONS:
    print(f"\n{'='*60}", flush=True)
    print(f"[stage-f] mem_fraction = {mf}", flush=True)
    print(f"{'='*60}", flush=True)

    target = {**current, "SGL_MEM_FRACTION_STATIC": str(mf)}
    # apply_knobs handles the restart + cpu recycle.
    try:
        current = tp.apply_knobs(target, current)
    except RuntimeError as e:
        print(f"[stage-f] SGLang did NOT come healthy at mf={mf}: {e}", flush=True)
        print(f"[stage-f] probable OOM at startup — skipping this + lower values", flush=True)
        break

    # Grab SGLang's startup log to find the KV cache line.
    startup_log = RAW / f"sglang-startup-mf{mf}.log"
    try:
        out = subprocess.run(
            ["docker", "logs", "--tail", "200", "glmocr-sglang"],
            check=False, capture_output=True, text=True,
        ).stdout
        startup_log.write_text(out, encoding="utf-8")
        kv_lines = [l for l in out.splitlines() if KV_RE.search(l)][-5:]
        print(f"[stage-f] sglang startup KV-related lines:", flush=True)
        for l in kv_lines:
            print(f"  {l}", flush=True)
    except Exception as e:
        print(f"[stage-f] couldn't grab sglang logs: {e}", flush=True)

    # Run one cell.
    bench_json  = RAW / f"f-mf{mf}.json"
    probe_jsonl = RAW / f"probe-mf{mf}.jsonl"
    probe = tp._probe_spawn(probe_jsonl)
    time.sleep(4)
    try:
        summary = tp.run_bench(
            C, N, urls_file, bench_json,
            pool_seed=POOL_SEED,
            max_fail_rate=0.10,
            min_sample_for_abort=40,
        )
    finally:
        tp._probe_reap(probe)

    probe_sum = tp._probe_aggregate(probe_jsonl)
    knobs_for_classifier = {
        **{k: str(v) for k, v in current.items()},
        "SGL_MEM_FRACTION_STATIC": str(mf),
    }
    util = tp._utilization_ratios(probe_sum, knobs_for_classifier)
    bneck = tp._classify_bottleneck(probe_sum, knobs_for_classifier)

    summary["probe_summary"] = probe_sum
    summary["utilization"] = util
    summary["bottleneck"] = bneck
    summary["mem_fraction_static"] = mf
    bench_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    att = summary.get("requests_attempted") or summary.get("total")
    fr = (summary.get("failures", 0) / att) if att else 0
    lat = summary.get("latency_ms", {})
    print(
        f"[stage-f] mf={mf} -> rps={summary.get('throughput_rps', 0):.3f} "
        f"fail={fr:.0%} p50={lat.get('p50', 0):.0f}ms "
        f"p95={lat.get('p95', 0):.0f}ms p99={lat.get('p99', 0):.0f}ms  "
        f"GPU%={util.get('gpu_compute', 0)*100:.0f} "
        f"VRAM%={util.get('gpu_memory', 0)*100:.0f} "
        f"batch%={util.get('sgl_batch', 0)*100:.0f}  "
        f"bottleneck={bneck}  aborted={summary.get('aborted', False)}",
        flush=True,
    )

    trials.append({
        "kind": f"stage-f (mf={mf})",
        "omw": 8, "sgl": 32, "cpu_w": "2", "cpu_t": "4",
        "knobs": {
            "SGL_MEM_FRACTION_STATIC": str(mf),
            "SGL_SPECULATIVE": "true",
            "concurrency": str(C),
            "bottleneck": bneck,
        },
        "summary": summary,
    })

(RAW / "_trials.json").write_text(json.dumps(trials, indent=2), encoding="utf-8")

print(f"\n[stage-f] done. Summary:", flush=True)
print(f"{'mf':>5} {'rps':>6} {'fail%':>6} {'p99s':>6} {'GPU%':>5} {'VRAM%':>6} {'batch%':>7}  bottleneck", flush=True)
for t in trials:
    s = t["summary"]; u = s["utilization"]; lat = s["latency_ms"]
    att = s.get("requests_attempted") or s.get("total")
    fr = (s.get("failures", 0) / att) if att else 0
    print(
        f"{s['mem_fraction_static']:>5} {s['throughput_rps']:>6.3f} {fr:>6.0%} "
        f"{lat.get('p99', 0)/1000:>6.0f} {u['gpu_compute']*100:>5.0f} "
        f"{u['gpu_memory']*100:>6.0f} {u['sgl_batch']*100:>7.0f}  {s['bottleneck']}",
        flush=True,
    )

print(f"\n[stage-f] raw dir: {RAW}", flush=True)
print(f"[stage-f] trials.json: {RAW / '_trials.json'}", flush=True)
