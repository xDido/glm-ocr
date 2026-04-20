"""Watch a tune_params.py raw-trials directory and push sweep-progress
metrics to Prometheus Pushgateway on every poll.

Pushgateway is scraped by the local Prometheus; Alloy remote-writes
allowed `glmocr_*` series to Grafana Cloud Mimir; panels on
dido.grafana.net visualize them. No direct Cloud writes happen here.

Emits (all gauges, labelled by run_id + stage):
    glmocr_sweep_cells_done
    glmocr_sweep_cells_total
    glmocr_sweep_progress_pct             0..100
    glmocr_sweep_cells_aborted_total
    glmocr_sweep_cells_succeeded_total    (cell-level, fail_rate <= max_fail_rate)
    glmocr_sweep_last_cell_rps
    glmocr_sweep_last_cell_fail_rate      0..1
    glmocr_sweep_last_cell_aborted        0|1
    glmocr_sweep_last_cell_ts             unix-seconds the last cell wrote its JSON

Usage:
    python scripts/sweep_progress_push.py \
        --run-id omnidoc-20260419-123742-stage-a \
        --stage a \
        --total 36 \
        --interval 10
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
from datetime import datetime

try:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
except ImportError:
    sys.stderr.write("prometheus_client missing (pip install prometheus-client)\n")
    sys.exit(1)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--stage", required=True, choices=["a", "b", "c", "d", "e", "f", "g"])
    p.add_argument("--total", type=int, required=True,
                   help="total cells the sweep is expected to run")
    p.add_argument("--raw-dir", default=None,
                   help="override loadtest/results/raw/<run-id>/")
    p.add_argument("--pushgateway-url", default="http://localhost:9091")
    p.add_argument("--interval", type=float, default=10.0,
                   help="seconds between polls")
    p.add_argument("--stop-when-done", action="store_true",
                   help="exit when done == total (default = keep publishing)")
    return p.parse_args()


def _cell_file_glob(stage: str) -> str:
    return {"a": "a-*.json", "b": "b-*.json", "c": "c-*.json",
            "d": "d-*.json", "e": "e-*.json", "f": "f-*.json",
            "g": "g-*.json"}[stage]


def _read_cell_files(raw_dir: pathlib.Path, stage: str) -> list[pathlib.Path]:
    # _trials.json shows up at end of run; ignore it here.
    files = [p for p in raw_dir.glob(_cell_file_glob(stage))
             if not p.name.startswith("_")]
    return sorted(files, key=lambda p: p.stat().st_mtime)


def _last_cell_stats(paths: list[pathlib.Path]) -> dict | None:
    if not paths:
        return None
    try:
        return json.loads(paths[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


RUN_ID_TS_RE = re.compile(r"omnidoc-(\d{8}-\d{6})-")


def _parse_run_start_ts(run_id: str) -> float | None:
    """Run IDs are minted as `omnidoc-YYYYMMDD-HHMMSS-<stage>`; extract the
    datetime so we can report elapsed time without waiting for the first
    cell to land."""
    m = RUN_ID_TS_RE.match(run_id)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1), "%Y%m%d-%H%M%S")
        # Interpret as local time so wall-clock math lines up with the
        # operator's perception.
        return dt.timestamp()
    except ValueError:
        return None


def _aggregate_totals(paths: list[pathlib.Path]) -> dict[str, float]:
    """Sum per-cell counters across all completed cells."""
    totals = {
        "requests_attempted": 0,
        "successes": 0,
        "failures": 0,
        "bench_seconds": 0.0,
    }
    for p in paths:
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        totals["requests_attempted"] += s.get("requests_attempted") or s.get("total") or 0
        totals["successes"] += s.get("successes", 0)
        totals["failures"]  += s.get("failures", 0)
        totals["bench_seconds"] += float(s.get("wall_seconds", 0.0))
    return totals


# Bottleneck string → numeric code. Used for the Grafana gauge, which
# needs a scalar; the full string is kept in the markdown report.
# Order = classifier priority (0–5 = saturation, 6–7 = under-use, 8 = n/a).
BOTTLENECK_CODES = [
    ("GPU memory (VRAM)",              0),
    ("GPU compute (batch full)",       1),
    ("GPU compute (undersubscribed)",  2),
    ("CPU container",                  3),
    ("GPU queue",                      4),   # "GPU queue (scheduler-limited)"
    ("CPU ingress",                    5),   # "CPU ingress (gunicorn)"
    ("!! UNDER-UTILIZED",               6),
    ("slack",                          7),
    ("no probe data",                  8),
]


def _bottleneck_to_code(label: str) -> int:
    """Prefix match against BOTTLENECK_CODES. Unknown → 8."""
    if not label:
        return 8
    for prefix, code in BOTTLENECK_CODES:
        if label.startswith(prefix):
            return code
    return 8


def _is_slo_met(path: pathlib.Path,
                p99_ceiling_ms: float = 120_000.0,
                max_fail_rate: float = 0.10) -> bool:
    try:
        s = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    lat = s.get("latency_ms") or {}
    p99 = float(lat.get("p99") or 0.0)
    attempted = s.get("requests_attempted") or s.get("total") or 1
    fail_rate = float(s.get("failures", 0)) / attempted
    return (p99 <= p99_ceiling_ms) and (fail_rate <= max_fail_rate) and not s.get("aborted", False)


def _count_aborted_and_succeeded(paths: list[pathlib.Path]) -> tuple[int, int]:
    aborted = 0
    succeeded = 0
    for p in paths:
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if s.get("aborted"):
            aborted += 1
        # A "clean" cell: not aborted and fail_rate <= max_fail_rate (if we
        # recorded it). Otherwise fall back to fail_rate <= 0.10.
        total = s.get("total", 0) or 1
        fails = s.get("failures", 0)
        threshold = s.get("max_fail_rate")
        if threshold is None:
            threshold = 0.10
        if not s.get("aborted") and (fails / total) <= threshold:
            succeeded += 1
    return aborted, succeeded


def push(args: argparse.Namespace, raw_dir: pathlib.Path) -> bool:
    """Push one snapshot. Returns True if done == total."""
    cells = _read_cell_files(raw_dir, args.stage)
    done = len(cells)
    aborted, succeeded = _count_aborted_and_succeeded(cells)
    last = _last_cell_stats(cells)
    pct = (100.0 * done / args.total) if args.total else 0.0

    registry = CollectorRegistry()
    labels = ["run_id", "stage"]
    label_vals = [args.run_id, args.stage]

    def gauge(name: str, help_text: str, value: float) -> None:
        g = Gauge(name, help_text, labelnames=labels, registry=registry)
        g.labels(*label_vals).set(value)

    gauge("glmocr_sweep_cells_done",            "Sweep cells completed",              done)
    gauge("glmocr_sweep_cells_total",           "Sweep cells total",                  args.total)
    gauge("glmocr_sweep_progress_pct",          "Sweep progress (0..100)",            pct)
    gauge("glmocr_sweep_cells_aborted_total",   "Sweep cells aborted early",          aborted)
    gauge("glmocr_sweep_cells_succeeded_total", "Sweep cells at/below fail gate",     succeeded)

    # Time + totals visibility. started_at is parsed from the run_id
    # timestamp so we have elapsed even before cell 1 finishes.
    now_ts = time.time()
    started_at = _parse_run_start_ts(args.run_id) or (cells[0].stat().st_mtime if cells else now_ts)
    elapsed = max(0.0, now_ts - started_at)
    totals = _aggregate_totals(cells)
    gauge("glmocr_sweep_started_at_ts",         "Sweep start time (unix seconds)",    started_at)
    gauge("glmocr_sweep_elapsed_seconds",       "Wall-clock elapsed since sweep start", elapsed)
    gauge("glmocr_sweep_total_requests_attempted", "Docs attempted across all cells", totals["requests_attempted"])
    gauge("glmocr_sweep_total_successes",       "Docs successfully parsed across all cells", totals["successes"])
    gauge("glmocr_sweep_total_failures",        "Docs failed across all cells",       totals["failures"])
    gauge("glmocr_sweep_total_bench_seconds",   "Summed wall_seconds across cells",   totals["bench_seconds"])

    # Derived pace + ETA. Guard divides so we don't publish +Inf/NaN.
    mean_cell_seconds = (totals["bench_seconds"] / done) if done else 0.0
    gauge("glmocr_sweep_mean_cell_seconds",     "Average cell duration",              mean_cell_seconds)
    cells_per_hour = (done / elapsed * 3600.0) if elapsed > 0 else 0.0
    gauge("glmocr_sweep_cells_per_hour",        "Cells completed per hour (live)",    cells_per_hour)
    eta_seconds = ((args.total - done) / cells_per_hour * 3600.0) if cells_per_hour > 0 else 0.0
    gauge("glmocr_sweep_eta_seconds",           "Estimated seconds until all cells complete", eta_seconds)

    if last is not None:
        lat = last.get("latency_ms") or {}
        gauge("glmocr_sweep_last_cell_rps",        "Last cell throughput",   float(last.get("throughput_rps", 0.0)))
        # Use attempted (the bench-level count of completed requests),
        # not target N, so aborted cells report their real fail rate.
        attempted = last.get("requests_attempted") or last.get("total") or 1
        fail_rate = float(last.get("failures", 0)) / attempted
        gauge("glmocr_sweep_last_cell_fail_rate",  "Last cell fail rate",    fail_rate)
        gauge("glmocr_sweep_last_cell_aborted",    "Last cell aborted (0|1)", 1 if last.get("aborted") else 0)
        gauge("glmocr_sweep_last_cell_p95_ms",     "Last cell p95 latency",  float(lat.get("p95", 0.0)))
        gauge("glmocr_sweep_last_cell_p99_ms",     "Last cell p99 latency",  float(lat.get("p99", 0.0)))
        gauge("glmocr_sweep_last_cell_max_ms",     "Last cell max latency",  float(lat.get("max", 0.0)))
        gauge("glmocr_sweep_last_cell_mean_ms",    "Last cell mean latency", float(lat.get("mean", 0.0)))
        gauge("glmocr_sweep_last_cell_ts",         "Last cell write time",   float(cells[-1].stat().st_mtime))
        # SLO compliance: p99 ≤ 120s AND fail rate ≤ 10%. Panel on the
        # Grafana dashboard uses this to flag which cells would meet
        # production tail-latency targets.
        p99_ms = float(lat.get("p99", 0.0))
        slo_met = int(p99_ms <= 120_000 and fail_rate <= 0.10)
        gauge("glmocr_sweep_last_cell_slo_met",    "Last cell meets p99<=120s, fail<=10% (0|1)", slo_met)
        # Also push a rolling counter of SLO-compliant cells across the sweep.
        slo_count = sum(1 for p in cells if _is_slo_met(p))
        gauge("glmocr_sweep_cells_slo_met_total",  "Cells meeting the SLO",  slo_count)

        # v4: bottleneck + per-resource utilization. `bottleneck` and
        # `utilization` are written by stage_d() in tune_params.py. When
        # earlier stages (a/b/c) run without probe orchestration, these
        # fields are absent and the gauges publish as 0 / code=8.
        bottleneck = last.get("bottleneck", "")
        gauge("glmocr_sweep_last_cell_bottleneck_code",
              "Last cell bottleneck (0..8 — see BOTTLENECK_CODES)",
              _bottleneck_to_code(bottleneck))
        util = last.get("utilization") or {}
        gauge("glmocr_sweep_last_cell_util_gpu_compute",
              "Last cell GPU compute utilization (0..1)",
              float(util.get("gpu_compute", 0.0)))
        gauge("glmocr_sweep_last_cell_util_gpu_memory",
              "Last cell GPU memory utilization (0..1)",
              float(util.get("gpu_memory", 0.0)))
        gauge("glmocr_sweep_last_cell_util_sgl_batch",
              "Last cell SGLang batch utilization (running/cap, 0..1)",
              float(util.get("sgl_batch", 0.0)))
        gauge("glmocr_sweep_last_cell_util_cpu_container",
              "Last cell CPU container utilization (cores used / cpu_slots, 0..1)",
              float(util.get("cpu_container", 0.0)))
        peak_util = max(util.values(), default=0.0)
        gauge("glmocr_sweep_last_cell_util_peak",
              "Last cell peak utilization across all tracked resources (0..1)",
              float(peak_util))
        # Under-utilized flag: peak < 0.50 → true. Cheap scalar for a
        # stat panel and alerting.
        gauge("glmocr_sweep_last_cell_under_utilized",
              "Last cell has no resource above 50% utilization (0|1)",
              int(peak_util < 0.50))

    try:
        push_to_gateway(args.pushgateway_url, job="glmocr_sweep", registry=registry,
                        grouping_key={"run_id": args.run_id, "stage": args.stage})
    except Exception as exc:
        print(f"[sweep-push] pushgateway failed: {exc!r}", file=sys.stderr, flush=True)
    else:
        print(f"[sweep-push] {args.run_id} {done}/{args.total} "
              f"({pct:.0f}%) aborted={aborted} succeeded={succeeded}"
              + (f" last_rps={last.get('throughput_rps', 0):.3f}" if last else ""),
              flush=True)

    return done >= args.total


def main() -> int:
    args = parse_args()
    raw_dir = pathlib.Path(args.raw_dir) if args.raw_dir else (
        REPO_ROOT / "loadtest" / "results" / "raw" / args.run_id
    )
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sweep-push] watching {raw_dir} (total={args.total}, "
          f"interval={args.interval}s, push={args.pushgateway_url})",
          flush=True)

    try:
        while True:
            done = push(args, raw_dir)
            if done and args.stop_when_done:
                print("[sweep-push] complete — exiting", flush=True)
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("[sweep-push] stopped", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
