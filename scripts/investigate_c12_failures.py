"""One-shot: run c=12 at the current stack config with abort DISABLED
so all 200 requests complete (or die trying). Then analyze which
documents failed, what error each got, and how long each took before
failing.

Use when you need to identify pathological documents in the pool vs
contention-induced failures. Pathological docs: same URL fails repeatedly
with similar timing across trials. Contention: docs fail randomly,
timing clusters near gunicorn_timeout (~180s).
"""
import os
import sys
import json
import time
import pathlib
import tempfile
import subprocess
from collections import Counter, defaultdict
from datetime import datetime

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ.setdefault("PROBE_VRAM_TOTAL_MB", "8192")

sys.path.insert(0, str(pathlib.Path("scripts").resolve()))
import tune_params as tp

RUN_ID = f"omnidoc-{datetime.now().strftime('%Y%m%d-%H%M%S')}-investigate-c12"
RAW = pathlib.Path("loadtest/results/raw") / RUN_ID
RAW.mkdir(parents=True, exist_ok=True)
print(f"[investigate] run_id={RUN_ID}", flush=True)

tmp = pathlib.Path(tempfile.mkdtemp(prefix="invest-"))
urls_file = tp.build_image_pool(128, tmp, seed=42)

bench_json  = RAW / "c12.json"
probe_jsonl = RAW / "probe-c12.jsonl"

print("[investigate] starting probe + bench at c=12, N=200, abort DISABLED...", flush=True)
probe = tp._probe_spawn(probe_jsonl)
time.sleep(4)
try:
    # max_fail_rate=None disables the abort — every one of 200 requests runs.
    summary = tp.run_bench(
        concurrency=12, total=200,
        urls_file=urls_file, out_json=bench_json,
        pool_seed=42,
        max_fail_rate=None,
        min_sample_for_abort=40,
    )
finally:
    tp._probe_reap(probe)

# Post-process failures.
fd = summary.get("failure_details") or []
print(f"\n[investigate] total: {summary.get('total')}  ok: {summary.get('successes')}  "
      f"fail: {summary.get('failures')}  aborted: {summary.get('aborted', False)}", flush=True)
lat = summary.get("latency_ms", {})
print(f"[investigate] p50={lat.get('p50', 0):.0f}ms  p95={lat.get('p95', 0):.0f}ms  "
      f"p99={lat.get('p99', 0):.0f}ms  wall={summary.get('wall_seconds', 0):.0f}s  "
      f"rps={summary.get('throughput_rps', 0):.3f}", flush=True)

if not fd:
    print("[investigate] NO FAILURES — nothing to investigate.", flush=True)
    sys.exit(0)

# Group by error class.
print(f"\n[investigate] failures by error class:", flush=True)
by_err = Counter()
for f in fd:
    # Strip variable parts (bytes hashes, addresses) so similar errors cluster.
    key = (f.get("error") or "").split("(")[0].strip() or "?"
    by_err[key] += 1
for err, n in by_err.most_common():
    print(f"  {n:3d}  {err}", flush=True)

# Group by image URL — are some docs serial offenders?
print(f"\n[investigate] failure count per image URL (top 20):", flush=True)
by_url = Counter(f["image_url"] for f in fd)
for url, n in by_url.most_common(20):
    # basename only, easier to scan
    name = url.rsplit("/", 1)[-1]
    print(f"  {n:3d} x  {name}", flush=True)

# Timing distribution of failures — did they die near the 180s timeout?
print(f"\n[investigate] failure timing distribution:", flush=True)
buckets = [(0, 1), (1, 10), (10, 60), (60, 120), (120, 180), (180, 300), (300, 1e9)]
labels = ["<1s", "1-10s", "10-60s", "60-120s", "120-180s", "180-300s", ">300s"]
hist = [0] * len(buckets)
for f in fd:
    el = (f.get("elapsed_ms") or 0) / 1000
    for i, (lo, hi) in enumerate(buckets):
        if lo <= el < hi:
            hist[i] += 1
            break
for l, n in zip(labels, hist):
    bar = "#" * min(n, 50)
    print(f"  {l:>10s} : {n:3d}  {bar}", flush=True)

# Save the full failure report for grep/jq.
report_json = RAW / "failure-report.json"
report_json.write_text(json.dumps({
    "summary": {k: v for k, v in summary.items() if k != "failure_details"},
    "failures_by_error_class": dict(by_err),
    "failures_by_url": {url: n for url, n in by_url.most_common()},
    "failure_timing_histogram": {l: n for l, n in zip(labels, hist)},
    "failure_details": fd,
}, indent=2), encoding="utf-8")
print(f"\n[investigate] full report: {report_json}", flush=True)

# Classification verdict.
print(f"\n[investigate] VERDICT:", flush=True)
if by_url.most_common(1) and by_url.most_common(1)[0][1] >= 3:
    top_url, top_n = by_url.most_common(1)[0]
    print(f"  pathological-doc signal: '{top_url.rsplit('/', 1)[-1]}' failed {top_n} times — "
          f"deterministic bad doc.", flush=True)
if hist[4] + hist[5] >= 0.5 * sum(hist):
    print(f"  timeout-cluster signal: {hist[4] + hist[5]} failures in the 120-300s band — "
          f"gunicorn timeout is killing slow docs.", flush=True)
if hist[0] + hist[1] >= 0.3 * sum(hist):
    print(f"  fast-failure signal: {hist[0] + hist[1]} failures under 10s — "
          f"likely scheduler rejects or pool-exhaustion, not timeouts.", flush=True)
