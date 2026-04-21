"""Staged sweep driver for GLM-OCR.

Legacy default (no --stage) = the 2D grid OCR_MAX_WORKERS × SGL_MAX_RUNNING_REQUESTS.

Staged modes for the v2 report:

  --stage a  1D scans of each SGLang knob at baseline, c=8, N=200. Use to
             identify which knobs actually move throughput/latency.
  --stage b  2D grid on two chosen axes (--axes KEY1,KEY2), N=200 per cell.
             Defaults to the legacy OMW×SGL pair so existing make targets
             keep working.
  --stage c  c-curve verification at a fixed config, N=200 per concurrency.
             Replaces the noisy N=40 final curve in the v1 report.

Per trial: mutate .env, restart the impacted services, run bench.py with a
seeded image pool (--pool-seed), record raw JSON under
loadtest/results/raw/<run-id>/. Rendered markdown + PNGs come from
scripts/lib/render_report.py (new stage-a/b/c sections).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Knob catalog — which services must restart when each value changes.
# ---------------------------------------------------------------------------

# "sglang" implies also cpu (the CPU container's aiohttp pool has stale
# sockets pointed at the old sglang container after a recreate).
KNOBS: dict[str, dict] = {
    "SGL_MAX_RUNNING_REQUESTS":   {"values": [8, 16, 24, 32, 48],             "restarts": ["sglang"]},
    "SGL_MAX_TOTAL_TOKENS":       {"values": [50000, 100000, 150000, 200000], "restarts": ["sglang"]},
    "SGL_MAX_PREFILL_TOKENS":     {"values": [8192, 16384, 32768],            "restarts": ["sglang"]},
    "SGL_CHUNKED_PREFILL":        {"values": ["true", "false"],               "restarts": ["sglang"]},
    "SGL_SCHEDULE_POLICY":        {"values": ["lpm", "fcfs"],                 "restarts": ["sglang"]},
    "SGL_CHUNKED_PREFILL_SIZE":   {"values": [4096, 8192, 16384],             "restarts": ["sglang"]},
    "SGL_MEM_FRACTION_STATIC":    {"values": [0.80, 0.85, 0.88, 0.92],        "restarts": ["sglang"]},
    "SGL_SPECULATIVE":            {"values": ["true", "false"],               "restarts": ["sglang"]},
    "SGL_SPEC_NUM_STEPS":         {"values": [2, 3, 5, 7],                    "restarts": ["sglang"], "requires": {"SGL_SPECULATIVE": "true"}},
    "SGL_SPEC_NUM_DRAFT_TOKENS":  {"values": [2, 4, 8],                       "restarts": ["sglang"], "requires": {"SGL_SPECULATIVE": "true"}},
    "SGL_SPEC_EAGLE_TOPK":        {"values": [1, 2, 4],                       "restarts": ["sglang"], "requires": {"SGL_SPECULATIVE": "true"}},
    "OCR_CONN_POOL":              {"values": [32, 64, 128, 256],              "restarts": ["cpu"]},
    "OCR_MAX_WORKERS":            {"values": [4, 8, 12, 16, 24],              "restarts": ["cpu"]},
    "CPU_WORKERS":                {"values": [2, 4, 8],                       "restarts": ["cpu"]},
    "CPU_THREADS":                {"values": [1, 2, 4, 8],                    "restarts": ["cpu"]},
}

# Stage D = 2×2×2 grid on CPU shape × OCR_MAX_WORKERS × SGL_MAX_RUNNING_REQUESTS.
# All other knobs fixed; see plan v4. Each entry lists the full set of
# values to set before the cell runs — tune_params applies them via
# apply_knobs() which restarts only services whose knobs changed.
STAGE_D_COMBOS: list[dict[str, object]] = [
    {"CPU_WORKERS": 2, "CPU_THREADS": 4, "OCR_MAX_WORKERS": 4, "SGL_MAX_RUNNING_REQUESTS": 24},
    {"CPU_WORKERS": 2, "CPU_THREADS": 4, "OCR_MAX_WORKERS": 4, "SGL_MAX_RUNNING_REQUESTS": 32},
    {"CPU_WORKERS": 2, "CPU_THREADS": 4, "OCR_MAX_WORKERS": 8, "SGL_MAX_RUNNING_REQUESTS": 24},
    {"CPU_WORKERS": 2, "CPU_THREADS": 4, "OCR_MAX_WORKERS": 8, "SGL_MAX_RUNNING_REQUESTS": 32},
    {"CPU_WORKERS": 8, "CPU_THREADS": 1, "OCR_MAX_WORKERS": 4, "SGL_MAX_RUNNING_REQUESTS": 24},
    {"CPU_WORKERS": 8, "CPU_THREADS": 1, "OCR_MAX_WORKERS": 4, "SGL_MAX_RUNNING_REQUESTS": 32},
    {"CPU_WORKERS": 8, "CPU_THREADS": 1, "OCR_MAX_WORKERS": 8, "SGL_MAX_RUNNING_REQUESTS": 24},
    {"CPU_WORKERS": 8, "CPU_THREADS": 1, "OCR_MAX_WORKERS": 8, "SGL_MAX_RUNNING_REQUESTS": 32},
]

STAGE_D_CONCURRENCY = 8
STAGE_D_TOTAL       = 200

# Stage E — c-curve saturation at the Stage-D winner. Fixed config,
# sweep client concurrency to find where GPU utilization hits ~80% and
# where p99 exits the 120s SLO. Each cell runs with the probe so we
# can attribute the bottleneck at each c-level.
STAGE_E_CONCURRENCIES = [4, 8, 12, 16, 20, 24]
STAGE_E_TOTAL         = 200
STAGE_E_FIXED_KNOBS: dict[str, object] = {
    # Empty by design: stage_e is a c-curve at whatever the user
    # currently has set in .env. CPU shape, OMW, SGL caps, mf, etc.
    # are all inherited. Only client concurrency varies per cell.
    # Add overrides here only if you want stage_e to pin a specific
    # reference config regardless of .env — dangerous because stale
    # defaults will silently overwrite the user's .env tuning.
}

# Endpoint override so the same stage_e can hammer the async endpoint.
STAGE_E_ENDPOINT_ENV = "STAGE_E_ENDPOINT"   # default /glmocr/parse

# Legacy grid axes (default when --stage b is used without --axes).
LEGACY_OMW_VALUES = [4, 8, 12, 16, 24]
LEGACY_SGL_VALUES = [16, 24, 32]

# Per-stage bench sizing.
STAGE_A_CONCURRENCY = 8
STAGE_A_TOTAL       = 200
STAGE_B_CONCURRENCY = 8
STAGE_B_TOTAL       = 200
STAGE_C_CONCURRENCIES = [4, 8, 12, 16, 24]
STAGE_C_TOTAL       = 200

# Legacy (no --stage) sizing — unchanged from v1 so existing docs still
# describe what runs.
LEGACY_TRIAL_CONCURRENCY = 8
LEGACY_TRIAL_TOTAL       = 30
LEGACY_FINAL_CONCURRENCIES = [4, 8, 16]
LEGACY_FINAL_TOTAL       = 40

FAIL_RATE_GATE = 0.10
DEFAULT_POOL_SEED = 42


# ---------------------------------------------------------------------------
# File + stack helpers.
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV_PATH  = REPO_ROOT / ".env"

HEALTH = {
    "cpu":    "http://localhost:5002/health",
    "sglang": "http://localhost:30000/health",
}


def log(msg: str) -> None:
    print(f"[tune] {msg}", flush=True)


def read_env_value(var: str) -> str | None:
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith(f"{var}="):
            return line.split("=", 1)[1]
    return None


def update_env_value(var: str, value) -> None:
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    out, found = [], False
    for line in lines:
        if line.lstrip().startswith(f"{var}="):
            out.append(f"{var}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{var}={value}")
    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")


def restart_service(service: str) -> None:
    log(f"restarting {service}...")
    subprocess.run(
        ["docker", "compose", "up", "-d", "--force-recreate", service],
        check=True, cwd=REPO_ROOT, capture_output=True,
    )


def wait_healthy(service: str, timeout: float = 180.0) -> None:
    url = HEALTH[service]
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    elapsed = time.time() - start
                    log(f"{service} healthy after {elapsed:.0f}s")
                    return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"{service} not healthy after {timeout}s")


def build_image_pool(n: int, tmp_dir: pathlib.Path, seed: int | None) -> pathlib.Path:
    dataset = REPO_ROOT / "datasets" / "OmniDocBench"
    if not dataset.exists():
        raise RuntimeError(f"dataset missing at {dataset}")
    # Scope to the `images/` subdir only — skips non-page assets at the
    # dataset root (e.g. data_diversity.png, which is an infographic,
    # not a document page, and poisons p99 without being representative).
    images_dir = dataset / "images"
    search_root = images_dir if images_dir.is_dir() else dataset
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        images.extend(search_root.rglob(ext))
    if not images:
        raise RuntimeError(f"no images found under {search_root}")
    # Deterministic pool — always the same N files in the same order,
    # regardless of filesystem walk order. Sorted by relative path so
    # every run, on every host, gets byte-identical input.
    # The `seed` param is ignored for pool construction now; bench.py
    # --pool-seed still controls the per-request draw order.
    images.sort(key=lambda p: p.relative_to(dataset).as_posix())
    sample = images[: min(n, len(images))]
    urls = [
        f"file:///app/datasets/OmniDocBench/{p.relative_to(dataset).as_posix()}"
        for p in sample
    ]
    out = tmp_dir / "urls.txt"
    out.write_text("\n".join(urls) + "\n", encoding="utf-8")
    return out


def run_bench(
    concurrency: int,
    total: int,
    urls_file: pathlib.Path,
    out_json: pathlib.Path,
    pool_seed: int | None = None,
    warmup: int = 2,
    max_fail_rate: float | None = None,
    min_sample_for_abort: int = 40,
    endpoint: str | None = None,
) -> dict:
    args = [
        sys.executable, str(REPO_ROOT / "loadtest" / "asyncio" / "bench.py"),
        "--host", "http://localhost:5002",
        "--concurrency", str(concurrency),
        "--total", str(total),
        "--image-list-file", str(urls_file),
        "--json-out", str(out_json),
        "--warmup", str(warmup),
    ]
    if pool_seed is not None:
        args += ["--pool-seed", str(pool_seed)]
    if max_fail_rate is not None:
        args += [
            "--max-fail-rate", str(max_fail_rate),
            "--min-sample-for-abort", str(min_sample_for_abort),
        ]
    if endpoint:
        args += ["--endpoint", endpoint]
    subprocess.run(args, check=False, cwd=REPO_ROOT)
    return json.loads(out_json.read_text(encoding="utf-8"))


def _capture_metrics_snapshots(
    raw_dir: pathlib.Path, tag: str, when: str,
    cpu_url: str = "http://localhost:5002/metrics",
    sgl_url: str = "http://localhost:30000/metrics",
) -> None:
    """Snapshot /metrics from cpu + sglang into raw_dir.

    Writes:
      {raw_dir}/cpu_metrics_{tag}_{when}.txt
      {raw_dir}/sglang_metrics_{tag}_{when}.txt

    `when` is typically "pre" or "post". Captured BETWEEN warmup and the
    real run so the post-minus-pre histogram diff excludes cold-start
    samples that would otherwise poison p99/max.
    """
    for name, url in (("cpu", cpu_url), ("sglang", sgl_url)):
        path = raw_dir / f"{name}_metrics_{tag}_{when}.txt"
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                path.write_bytes(r.read())
        except Exception as e:
            log(f"[snapshot] failed to fetch {url}: {e}")


def _probe_spawn(out_path: pathlib.Path) -> subprocess.Popen:
    """Spawn runtime_probe_loop.py against the running stack. JSONL lands
    at out_path — one sample every 2s until the process is killed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "scripts" / "runtime_probe_loop.py"),
         str(out_path)],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _probe_reap(proc: subprocess.Popen) -> None:
    """Graceful stop; SIGTERM then SIGKILL if it lingers."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


def _probe_aggregate(probe_path: pathlib.Path) -> dict:
    """Parse the probe JSONL and compute per-field averages + 95th
    percentile for gpu util (the only signal where tails matter for
    bottleneck classification)."""
    try:
        lines = probe_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}
    samples = []
    for line in lines:
        if not line.strip():
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not samples:
        return {}

    def _vals(key: str) -> list[float]:
        return [float(s[key]) for s in samples
                if isinstance(s.get(key), (int, float))]

    def _avg(key: str) -> float | None:
        v = _vals(key)
        return (sum(v) / len(v)) if v else None

    def _max(key: str) -> float | None:
        v = _vals(key)
        return max(v) if v else None

    def _q(key: str, q: float) -> float | None:
        v = sorted(_vals(key))
        if not v:
            return None
        return v[min(len(v) - 1, int(q * len(v)))]

    def _p95(key: str) -> float | None:
        return _q(key, 0.95)

    def _p99(key: str) -> float | None:
        return _q(key, 0.99)

    return {
        "samples":            len(samples),
        "gpu_util_avg":       _avg("gpu_util_pct"),
        "gpu_util_p95":       _p95("gpu_util_pct"),
        "gpu_util_max":       _max("gpu_util_pct"),
        "vram_used_mb_avg":   _avg("vram_used_mb"),
        "vram_used_mb_p99":   _p99("vram_used_mb"),
        "vram_used_mb_max":   _max("vram_used_mb"),
        "vram_free_mb_avg":   _avg("vram_free_mb"),
        "cpu_cores_cpu_avg":  _avg("cpu_cores_cpu"),
        "cpu_cores_cpu_max":  _max("cpu_cores_cpu"),
        "cpu_cores_sgl_avg":  _avg("cpu_cores_sglang"),
        "cpu_cores_sgl_max":  _max("cpu_cores_sglang"),
        "mem_rss_cpu_mb_avg": _avg("mem_rss_cpu_mb"),
        "mem_rss_cpu_mb_p99": _p99("mem_rss_cpu_mb"),
        "mem_rss_cpu_mb_max": _max("mem_rss_cpu_mb"),
        "mem_rss_sgl_mb_avg": _avg("mem_rss_sglang_mb"),
        "mem_rss_sgl_mb_p99": _p99("mem_rss_sglang_mb"),
        "mem_rss_sgl_mb_max": _max("mem_rss_sglang_mb"),
        "in_flight_avg":      _avg("in_flight"),
        "in_flight_max":      _max("in_flight"),
        "sglang_running_avg": _avg("sglang_running"),
        "sglang_queued_avg":  _avg("sglang_queued"),
    }


def _utilization_ratios(probe: dict, knobs: dict[str, str]) -> dict[str, float]:
    """Compute 0..1 utilization ratios per resource. Missing/zero inputs
    return 0 rather than NaN so the classifier's max() is well-defined."""
    sgl_cap = float(knobs.get("SGL_MAX_RUNNING_REQUESTS", 16))
    cpu_slots = max(1.0, float(knobs.get("CPU_WORKERS", 2))
                         * float(knobs.get("CPU_THREADS", 8)))
    def _g(k: str) -> float:
        v = probe.get(k)
        return float(v) if v is not None else 0.0

    vram_total_mb = _gpu_vram_total_mb(probe)
    return {
        "gpu_compute":   _g("gpu_util_avg") / 100.0,
        "gpu_memory":    _g("vram_used_mb_avg") / vram_total_mb if vram_total_mb else 0.0,
        "sgl_batch":     _g("sglang_running_avg") / sgl_cap if sgl_cap else 0.0,
        "cpu_container": _g("cpu_cores_cpu_avg") / cpu_slots,
        "cpu_ingress":   _g("in_flight_avg") / cpu_slots,
    }


# VRAM total is derived per-sample from DCGM's FB_USED + FB_FREE (both
# reported in MiB). Falls back to PROBE_VRAM_TOTAL_MB env var when FB_FREE
# is unavailable (older DCGM or a different exporter). Default 16000
# matches the production T4 for reports rendered before the env is set.
_GPU_VRAM_MB_TOTAL_DEFAULT = float(os.environ.get("PROBE_VRAM_TOTAL_MB", "16000"))


def _gpu_vram_total_mb(probe: dict) -> float:
    used = probe.get("vram_used_mb_avg") or 0.0
    free = probe.get("vram_free_mb_avg")
    if free is not None and (used + free) > 0:
        return float(used + free)
    return _GPU_VRAM_MB_TOTAL_DEFAULT


def _classify_bottleneck(probe: dict, knobs: dict[str, str]) -> str:
    """Priority-ordered rules. Explicit labels for both saturation
    (resource pegged) and under-utilization (no resource near its cap)
    so operators see when the stack could handle more load."""
    if not probe or probe.get("samples", 0) < 3:
        return "no probe data"

    util = _utilization_ratios(probe, knobs)
    def _get(k: str, default: float = 0.0) -> float:
        v = probe.get(k)
        return float(v) if v is not None else default
    sgl_queued = _get("sglang_queued_avg")

    # Bottleneck (saturation) side first — priority order mirrors the
    # physical constraint hierarchy. VRAM threshold is 0.85 not 0.95
    # because SGLang's scheduler gets KV-aggressive well before the
    # card physically OOMs, so the practical ceiling is lower than the
    # hardware cap.
    if util["gpu_memory"] > 0.85:
        return f"GPU memory (VRAM={util['gpu_memory']:.0%})"
    if util["gpu_compute"] >= 0.85:
        if util["sgl_batch"] >= 0.90:
            return "GPU compute (batch full)"
        return "GPU compute (undersubscribed)"
    if util["cpu_container"] >= 0.85:
        return "CPU container"
    if sgl_queued >= 2:
        return "GPU queue (scheduler-limited)"
    if util["cpu_ingress"] >= 0.80 and sgl_queued < 1:
        return "CPU ingress (gunicorn)"

    # Under-utilization side: name the peak resource so the operator
    # can tell why nothing's maxed.
    peak_name, peak_val = max(util.items(), key=lambda kv: kv[1])
    # ASCII-only label — Windows cp1252 stdout chokes on ⚠.
    if peak_val < 0.50:
        return f"!! UNDER-UTILIZED ({peak_name}={peak_val:.0%} - bump concurrency)"
    return f"slack ({peak_name}={peak_val:.0%}, no resource saturated)"


def apply_knobs(target: dict[str, object], current: dict[str, str]) -> dict[str, str]:
    """Write .env for any knob whose value changed; restart impacted services.
    Returns the new current-state mapping."""
    changed = {k: str(v) for k, v in target.items() if current.get(k) != str(v)}
    if not changed:
        return current
    need_sgl = False
    need_cpu = False
    for k, v in changed.items():
        update_env_value(k, v)
        restarts = KNOBS.get(k, {}).get("restarts", ["cpu"])
        if "sglang" in restarts:
            need_sgl = True
        if "cpu" in restarts:
            need_cpu = True
    # Any sglang recreate forces a cpu recreate to flush the aiohttp pool.
    if need_sgl:
        restart_service("sglang"); wait_healthy("sglang")
        need_cpu = True
    if need_cpu:
        restart_service("cpu"); wait_healthy("cpu")
    new_state = dict(current)
    new_state.update(changed)
    return new_state


# ---------------------------------------------------------------------------
# Trial container.
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    kind: str                        # "grid", "final (c=N)", "stage-a:<knob>", "stage-b", "stage-c (c=N)"
    omw: int | None
    sgl: int | None
    cpu_w: str | None
    cpu_t: str | None
    summary: dict
    knobs: dict[str, str] = field(default_factory=dict)

    @property
    def rps(self) -> float:
        return float(self.summary.get("throughput_rps", 0.0))

    @property
    def fail_rate(self) -> float:
        # For aborted cells, the bench loop stops before hitting `total`,
        # so dividing failures by `total` reports a diluted number (e.g.
        # 5 fails of 49 attempted → 10.2% in reality but would display
        # as 2.5% if we used `total`). Use attempted as the denominator
        # whenever it's set and non-zero.
        s = self.summary
        attempted = s.get("requests_attempted") or s.get("total") or 1
        return float(s.get("failures", 0)) / attempted

    @property
    def p95(self) -> float:
        return float((self.summary.get("latency_ms") or {}).get("p95", float("nan")))

    @property
    def p99(self) -> float:
        return float((self.summary.get("latency_ms") or {}).get("p99", float("nan")))

    @property
    def p50(self) -> float:
        return float((self.summary.get("latency_ms") or {}).get("p50", float("nan")))

    @property
    def mean_ms(self) -> float:
        return float((self.summary.get("latency_ms") or {}).get("mean", float("nan")))

    @property
    def wall(self) -> float:
        return float(self.summary.get("wall_seconds", 0.0))

    def score(self) -> float:
        if self.fail_rate > FAIL_RATE_GATE:
            return -self.fail_rate
        return self.rps

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "omw": self.omw,
            "sgl": self.sgl,
            "cpu_w": self.cpu_w,
            "cpu_t": self.cpu_t,
            "knobs": self.knobs,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Stage A: 1D scans.
# ---------------------------------------------------------------------------

# Order matters — if the user cancels the sweep mid-run, the earliest
# scans should be the most informative. Speculative-decoding changes the
# GPU's decode path entirely, so it goes first. Spec sub-knobs follow
# (they only run when SPECULATIVE=true in baseline — see `requires`
# gate in KNOBS). Mem-fraction interacts with spec's KV cache pressure,
# so it's right after. Then the classic batch/cache knobs. Finally the
# CPU-side connection pool.
STAGE_A_SCAN_KNOBS = [
    "SGL_SPECULATIVE",
    "SGL_SPEC_NUM_STEPS",
    "SGL_SPEC_NUM_DRAFT_TOKENS",
    "SGL_SPEC_EAGLE_TOPK",
    "SGL_MEM_FRACTION_STATIC",
    "SGL_MAX_RUNNING_REQUESTS",
    "SGL_MAX_TOTAL_TOKENS",
    "SGL_MAX_PREFILL_TOKENS",
    "SGL_CHUNKED_PREFILL",
    "SGL_SCHEDULE_POLICY",
    "OCR_CONN_POOL",
]


def snapshot_env(keys: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k in keys:
        v = read_env_value(k)
        if v is not None:
            out[k] = v
    return out


def _knob_requires_met(knob: str, baseline: dict[str, str]) -> tuple[bool, str]:
    """If a knob has a `requires` gate (like SGL_SPEC_NUM_STEPS needing
    SGL_SPECULATIVE=true), confirm the gate is satisfied by baseline."""
    req = KNOBS.get(knob, {}).get("requires", {})
    for rk, rv in req.items():
        if str(baseline.get(rk, "")).lower() != str(rv).lower():
            return False, f"requires {rk}={rv} (baseline has {baseline.get(rk, '<unset>')})"
    return True, ""


def _format_progress(i: int, total: int) -> str:
    pct = (100 * i / total) if total else 0.0
    return f"[{i}/{total} {pct:3.0f}%]"


def stage_a(
    baseline: dict[str, str],
    urls_file: pathlib.Path,
    raw_dir: pathlib.Path,
    pool_seed: int | None,
    dry_run: bool,
    scan_knobs: list[str],
    max_fail_rate: float | None = None,
    min_sample_for_abort: int = 40,
) -> list[Trial]:
    trials: list[Trial] = []
    current = dict(baseline)

    # Build the flat schedule up-front so we have a total count for
    # progress reporting. Also honors `requires` gates so spec sub-knob
    # cells are skipped entirely (rather than counted-then-skipped)
    # when speculative decoding is off in baseline.
    schedule: list[tuple[str, object]] = []
    skipped_notes: list[str] = []
    for knob in scan_knobs:
        if knob not in KNOBS:
            skipped_notes.append(f"  unknown knob {knob}: not in catalog")
            continue
        ok, why = _knob_requires_met(knob, baseline)
        if not ok:
            skipped_notes.append(f"  skipping {knob}: {why}")
            continue
        for v in KNOBS[knob]["values"]:
            schedule.append((knob, v))
    total = len(schedule)

    log(f"stage A baseline: {baseline}")
    log(f"stage A total cells: {total}"
        + (f" (max_fail_rate={max_fail_rate:.0%}, min_sample={min_sample_for_abort})"
           if max_fail_rate is not None else ""))
    for note in skipped_notes:
        log(note)

    for idx, (knob, v) in enumerate(schedule, start=1):
        tag = _format_progress(idx, total)
        target = dict(baseline); target[knob] = str(v)
        label = f"{knob}={v}"
        if dry_run:
            log(f"{tag} [dry] stage A: {label}")
            continue
        log(f"{tag} stage A: {label} (c={STAGE_A_CONCURRENCY}, n={STAGE_A_TOTAL})")
        current = apply_knobs(target, current)
        out = raw_dir / f"a-{knob}-{v}.json"
        summary = run_bench(
            STAGE_A_CONCURRENCY, STAGE_A_TOTAL, urls_file, out,
            pool_seed=pool_seed,
            max_fail_rate=max_fail_rate,
            min_sample_for_abort=min_sample_for_abort,
        )
        t = Trial(
            kind=f"stage-a:{knob}",
            omw=int(current.get("OCR_MAX_WORKERS", 0)) or None,
            sgl=int(current.get("SGL_MAX_RUNNING_REQUESTS", 0)) or None,
            cpu_w=current.get("CPU_WORKERS"),
            cpu_t=current.get("CPU_THREADS"),
            summary=summary,
            knobs={knob: str(v)},
        )
        trials.append(t)
        aborted_flag = " ABORTED" if summary.get("aborted") else ""
        log(
            f"{tag}  -> rps={t.rps:.3f} fail={t.fail_rate:.0%} "
            f"p50={t.p50:.0f}ms p95={t.p95:.0f}ms p99={t.p99:.0f}ms "
            f"mean={t.mean_ms:.0f}ms{aborted_flag}"
        )
    # Reset to baseline so next stage starts clean.
    if not dry_run:
        current = apply_knobs(baseline, current)
    return trials


# ---------------------------------------------------------------------------
# Stage B: 2D grid.
# ---------------------------------------------------------------------------

def stage_b(
    axes: tuple[str, str],
    baseline: dict[str, str],
    urls_file: pathlib.Path,
    raw_dir: pathlib.Path,
    pool_seed: int | None,
    dry_run: bool,
    max_fail_rate: float | None = None,
    min_sample_for_abort: int = 40,
) -> list[Trial]:
    x_knob, y_knob = axes
    if x_knob not in KNOBS or y_knob not in KNOBS:
        raise SystemExit(f"stage-b axes must be in KNOBS: got {axes}")
    trials: list[Trial] = []
    current = dict(baseline)
    total = len(KNOBS[x_knob]["values"]) * len(KNOBS[y_knob]["values"])
    log(f"stage B axes: {x_knob} × {y_knob} — {total} cells"
        + (f" (max_fail_rate={max_fail_rate:.0%})"
           if max_fail_rate is not None else ""))
    idx = 0
    for yv in KNOBS[y_knob]["values"]:
        for xv in KNOBS[x_knob]["values"]:
            idx += 1
            tag = _format_progress(idx, total)
            target = dict(baseline); target[x_knob] = str(xv); target[y_knob] = str(yv)
            label = f"{x_knob}={xv} {y_knob}={yv}"
            if dry_run:
                log(f"{tag} [dry] stage B: {label}")
                continue
            log(f"{tag} stage B: {label} (c={STAGE_B_CONCURRENCY}, n={STAGE_B_TOTAL})")
            current = apply_knobs(target, current)
            out = raw_dir / f"b-{x_knob}-{xv}-{y_knob}-{yv}.json"
            summary = run_bench(
                STAGE_B_CONCURRENCY, STAGE_B_TOTAL, urls_file, out,
                pool_seed=pool_seed,
                max_fail_rate=max_fail_rate,
                min_sample_for_abort=min_sample_for_abort,
            )
            t = Trial(
                kind="stage-b",
                omw=int(current.get("OCR_MAX_WORKERS", 0)) or None,
                sgl=int(current.get("SGL_MAX_RUNNING_REQUESTS", 0)) or None,
                cpu_w=current.get("CPU_WORKERS"),
                cpu_t=current.get("CPU_THREADS"),
                summary=summary,
                knobs={x_knob: str(xv), y_knob: str(yv)},
            )
            trials.append(t)
            aborted_flag = " ABORTED" if summary.get("aborted") else ""
            log(
                f"{tag}  -> rps={t.rps:.3f} fail={t.fail_rate:.0%} "
                f"p50={t.p50:.0f}ms p95={t.p95:.0f}ms p99={t.p99:.0f}ms "
                f"mean={t.mean_ms:.0f}ms{aborted_flag}"
            )
    return trials


# ---------------------------------------------------------------------------
# Stage C: c-curve verification at a fixed config.
# ---------------------------------------------------------------------------

def stage_c(
    fixed_config: dict[str, str],
    concurrencies: list[int],
    baseline: dict[str, str],
    urls_file: pathlib.Path,
    raw_dir: pathlib.Path,
    pool_seed: int | None,
    dry_run: bool,
    max_fail_rate: float | None = None,
    min_sample_for_abort: int = 40,
    total: int | None = None,
    warmup: int = 2,
) -> list[Trial]:
    trials: list[Trial] = []
    current = dict(baseline)
    if not dry_run:
        current = apply_knobs(fixed_config, current)
    n_per_cell = total or STAGE_C_TOTAL
    num_cells = len(concurrencies)
    log(f"stage C fixed config applied: {fixed_config} — {num_cells} cells")
    for idx, c in enumerate(concurrencies, start=1):
        tag = _format_progress(idx, num_cells)
        out = raw_dir / f"c-c{c}.json"
        probe_jsonl = raw_dir / f"probe-c{c:02d}.jsonl"
        if dry_run:
            log(f"{tag} [dry] stage C: c={c} n={n_per_cell}")
            continue
        log(f"{tag} stage C: c={c} n={n_per_cell}")

        # Warm the server externally so cold-start latency lands in
        # /metrics BEFORE our "pre" snapshot — the post-minus-pre diff
        # then excludes warmup samples from the phase histograms.
        if warmup > 0:
            warm_json = raw_dir / f"warmup-c{c:02d}.json"
            # Parallel warmup at concurrency = min(warmup, CPU_WORKERS) so
            # each gunicorn worker sees at least one request and triggers
            # its (potentially expensive) first-call compile / graph
            # capture in parallel rather than serially.
            cpu_w = int(current.get("CPU_WORKERS", 4) or 4)
            warm_c = max(1, min(warmup, cpu_w))
            log(f"{tag} warmup: {warmup} req(s) at c={warm_c} before snapshot")
            run_bench(
                concurrency=warm_c, total=warmup,
                urls_file=urls_file, out_json=warm_json,
                pool_seed=pool_seed, warmup=0,
            )

        _capture_metrics_snapshots(raw_dir, f"c{c:02d}", "pre")

        probe = _probe_spawn(probe_jsonl)
        time.sleep(4)
        try:
            summary = run_bench(
                c, n_per_cell, urls_file, out,
                pool_seed=pool_seed,
                warmup=0,
                max_fail_rate=max_fail_rate,
                min_sample_for_abort=min_sample_for_abort,
            )
        finally:
            _probe_reap(probe)

        _capture_metrics_snapshots(raw_dir, f"c{c:02d}", "post")

        probe_summary = _probe_aggregate(probe_jsonl)
        knobs_for_util = {**current,
                          **{k: str(v) for k, v in fixed_config.items()}}
        utilization = _utilization_ratios(probe_summary, knobs_for_util)
        bottleneck = _classify_bottleneck(probe_summary, knobs_for_util)
        summary["probe_summary"] = probe_summary
        summary["utilization"] = utilization
        summary["bottleneck"] = bottleneck
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        t = Trial(
            kind=f"stage-c (c={c})",
            omw=int(current.get("OCR_MAX_WORKERS", 0)) or None,
            sgl=int(current.get("SGL_MAX_RUNNING_REQUESTS", 0)) or None,
            cpu_w=current.get("CPU_WORKERS"),
            cpu_t=current.get("CPU_THREADS"),
            summary=summary,
            knobs={**fixed_config,
                   "concurrency": str(c),
                   "bottleneck": bottleneck},
        )
        trials.append(t)
        aborted_flag = " ABORTED" if summary.get("aborted") else ""
        log(
            f"{tag}  -> rps={t.rps:.3f} fail={t.fail_rate:.0%} "
            f"p50={t.p50:.0f}ms p95={t.p95:.0f}ms p99={t.p99:.0f}ms "
            f"mean={t.mean_ms:.0f}ms bottleneck={bottleneck!r}{aborted_flag}"
        )
    return trials


def parse_config_overrides(raw: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in raw or []:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


# ---------------------------------------------------------------------------
# Stage D: focused combo grid with per-cell bottleneck attribution.
# ---------------------------------------------------------------------------

def stage_d(
    baseline: dict[str, str],
    urls_file: pathlib.Path,
    raw_dir: pathlib.Path,
    pool_seed: int | None,
    dry_run: bool,
    max_fail_rate: float | None = None,
    min_sample_for_abort: int = 40,
) -> list[Trial]:
    trials: list[Trial] = []
    current = dict(baseline)
    total = len(STAGE_D_COMBOS)
    log(f"stage D: {total} combo cells"
        + (f" (max_fail_rate={max_fail_rate:.0%})"
           if max_fail_rate is not None else ""))

    for idx, combo in enumerate(STAGE_D_COMBOS, start=1):
        tag = _format_progress(idx, total)
        knob_str = " ".join(f"{k}={v}" for k, v in combo.items())
        if dry_run:
            log(f"{tag} [dry] stage D: {knob_str}")
            continue

        log(f"{tag} stage D: {knob_str} (c={STAGE_D_CONCURRENCY}, n={STAGE_D_TOTAL})")
        # Merge combo with baseline so ALL knobs we track are explicit —
        # apply_knobs() diffs against current and only restarts services
        # whose knobs actually changed.
        target = {**baseline, **{k: str(v) for k, v in combo.items()}}
        current = apply_knobs(target, current)

        bench_json  = raw_dir / f"d-cell{idx:02d}.json"
        probe_jsonl = raw_dir / f"probe-cell{idx:02d}.jsonl"

        probe = _probe_spawn(probe_jsonl)
        # Give the probe two ticks so its first sample lands before the
        # bench's first requests queue up at SGLang — otherwise the
        # probe file can end up empty on very short/aborted cells.
        time.sleep(4)
        try:
            summary = run_bench(
                STAGE_D_CONCURRENCY, STAGE_D_TOTAL, urls_file, bench_json,
                pool_seed=pool_seed,
                max_fail_rate=max_fail_rate,
                min_sample_for_abort=min_sample_for_abort,
            )
        finally:
            _probe_reap(probe)

        probe_summary = _probe_aggregate(probe_jsonl)
        utilization = _utilization_ratios(probe_summary, target)
        bottleneck = _classify_bottleneck(probe_summary, target)

        # Stash probe aggregate + utilization + bottleneck inside the
        # bench summary so _trials.json carries everything (avoids a
        # second file read in render_report).
        summary["probe_summary"] = probe_summary
        summary["utilization"] = utilization
        summary["bottleneck"] = bottleneck
        bench_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        t = Trial(
            kind="stage-d",
            omw=int(combo.get("OCR_MAX_WORKERS", 0)) or None,
            sgl=int(combo.get("SGL_MAX_RUNNING_REQUESTS", 0)) or None,
            cpu_w=str(combo.get("CPU_WORKERS", current.get("CPU_WORKERS"))),
            cpu_t=str(combo.get("CPU_THREADS", current.get("CPU_THREADS"))),
            summary=summary,
            knobs={**{k: str(v) for k, v in combo.items()},
                   "bottleneck": bottleneck},
        )
        trials.append(t)

        aborted_flag = " ABORTED" if summary.get("aborted") else ""
        log(
            f"{tag}  -> rps={t.rps:.3f} fail={t.fail_rate:.0%} "
            f"p50={t.p50:.0f}ms p95={t.p95:.0f}ms p99={t.p99:.0f}ms "
            f"mean={t.mean_ms:.0f}ms bottleneck={bottleneck!r}{aborted_flag}"
        )
    return trials


# ---------------------------------------------------------------------------
# Stage E: saturation c-curve at fixed winner config, per-cell probe +
# bottleneck attribution (same apparatus as Stage D so the report line
# up cleanly). Toggle endpoint=/glmocr/parse-async via env var when
# comparing sync vs async pipeline-overlap handlers.
# ---------------------------------------------------------------------------

def stage_e(
    baseline: dict[str, str],
    urls_file: pathlib.Path,
    raw_dir: pathlib.Path,
    pool_seed: int | None,
    dry_run: bool,
    max_fail_rate: float | None = None,
    min_sample_for_abort: int = 40,
    concurrencies: list[int] | None = None,
    total: int | None = None,
    warmup: int = 2,
) -> list[Trial]:
    """Hold Stage-D winner config fixed, sweep client concurrency,
    capture the probe per cell so we can see GPU/CPU utilisation at
    each c."""
    concurrencies = concurrencies or STAGE_E_CONCURRENCIES
    n_per_cell = total or STAGE_E_TOTAL
    endpoint = os.environ.get(STAGE_E_ENDPOINT_ENV) or None
    trials: list[Trial] = []
    current = dict(baseline)

    # Apply fixed config ONCE up front so sglang isn't recreated every
    # cell — we're only varying client-side concurrency, not server knobs.
    if not dry_run:
        current = apply_knobs(
            {**baseline, **{k: str(v) for k, v in STAGE_E_FIXED_KNOBS.items()}},
            current,
        )
    log(
        f"stage E fixed config: {STAGE_E_FIXED_KNOBS}  "
        f"endpoint={endpoint or '/glmocr/parse'}"
    )
    num_cells = len(concurrencies)

    for idx, c in enumerate(concurrencies, start=1):
        tag = _format_progress(idx, num_cells)
        if dry_run:
            log(f"{tag} [dry] stage E: c={c} n={n_per_cell}")
            continue

        log(f"{tag} stage E: c={c} n={n_per_cell}")
        bench_json  = raw_dir / f"e-c{c:02d}.json"
        probe_jsonl = raw_dir / f"probe-c{c:02d}.jsonl"

        probe = _probe_spawn(probe_jsonl)
        time.sleep(4)
        try:
            summary = run_bench(
                c, n_per_cell, urls_file, bench_json,
                pool_seed=pool_seed,
                warmup=warmup,
                max_fail_rate=max_fail_rate,
                min_sample_for_abort=min_sample_for_abort,
                endpoint=endpoint,
            )
        finally:
            _probe_reap(probe)

        probe_summary = _probe_aggregate(probe_jsonl)
        utilization = _utilization_ratios(probe_summary, {
            **STAGE_E_FIXED_KNOBS,
            "CPU_WORKERS": current.get("CPU_WORKERS", 2),
            "CPU_THREADS": current.get("CPU_THREADS", 4),
        })
        bottleneck = _classify_bottleneck(probe_summary, {
            **STAGE_E_FIXED_KNOBS,
            "CPU_WORKERS": current.get("CPU_WORKERS", 2),
            "CPU_THREADS": current.get("CPU_THREADS", 4),
        })
        summary["probe_summary"] = probe_summary
        summary["utilization"] = utilization
        summary["bottleneck"] = bottleneck
        bench_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        t = Trial(
            kind=f"stage-e (c={c})",
            omw=int(current.get("OCR_MAX_WORKERS", 0)) or None,
            sgl=int(current.get("SGL_MAX_RUNNING_REQUESTS", 0)) or None,
            cpu_w=str(current.get("CPU_WORKERS", "?")),
            cpu_t=str(current.get("CPU_THREADS", "?")),
            summary=summary,
            knobs={
                **{k: str(v) for k, v in STAGE_E_FIXED_KNOBS.items()},
                "concurrency": str(c),
                "endpoint": endpoint or "/glmocr/parse",
                "bottleneck": bottleneck,
            },
        )
        trials.append(t)

        aborted_flag = " ABORTED" if summary.get("aborted") else ""
        log(
            f"{tag}  -> rps={t.rps:.3f} fail={t.fail_rate:.0%} "
            f"p50={t.p50:.0f}ms p95={t.p95:.0f}ms p99={t.p99:.0f}ms "
            f"GPU%={utilization.get('gpu_compute', 0)*100:.0f} "
            f"batch%={utilization.get('sgl_batch', 0)*100:.0f} "
            f"bottleneck={bottleneck!r}{aborted_flag}"
        )
    return trials


# ---------------------------------------------------------------------------
# Legacy 2D grid (unchanged behavior when --stage is not given).
# ---------------------------------------------------------------------------

def legacy_grid_and_final(
    initial: dict[str, str],
    urls_file: pathlib.Path,
    raw_dir: pathlib.Path,
    pool_seed: int | None,
) -> list[Trial]:
    trials: list[Trial] = []
    current_sgl: int | None = None
    cpu_w = initial.get("CPU_WORKERS")
    cpu_t = initial.get("CPU_THREADS")

    for sgl in LEGACY_SGL_VALUES:
        if sgl != current_sgl:
            update_env_value("SGL_MAX_RUNNING_REQUESTS", sgl)
            restart_service("sglang"); wait_healthy("sglang")
            current_sgl = sgl

        for omw in LEGACY_OMW_VALUES:
            update_env_value("OCR_MAX_WORKERS", omw)
            restart_service("cpu"); wait_healthy("cpu")

            out = raw_dir / f"grid-omw{omw}-sgl{sgl}.json"
            log(f"grid trial OMW={omw} SGL={sgl} (c={LEGACY_TRIAL_CONCURRENCY}, n={LEGACY_TRIAL_TOTAL})")
            summary = run_bench(
                LEGACY_TRIAL_CONCURRENCY, LEGACY_TRIAL_TOTAL, urls_file, out,
                pool_seed=pool_seed,
            )
            t = Trial("grid", omw, sgl, cpu_w, cpu_t, summary)
            trials.append(t)
            log(
                f"  -> rps={t.rps:.3f} fail={t.fail_rate:.0%} "
                f"p50={t.p50:.0f}ms p95={t.p95:.0f}ms p99={t.p99:.0f}ms"
            )

    grid_trials = [t for t in trials if t.kind == "grid"]
    grid_trials.sort(key=lambda x: x.score(), reverse=True)
    winner = grid_trials[0]
    log(
        f"===== Grid winner: OMW={winner.omw} SGL={winner.sgl} "
        f"rps={winner.rps:.3f} fail={winner.fail_rate:.0%} ====="
    )

    update_env_value("OCR_MAX_WORKERS",          winner.omw)
    update_env_value("SGL_MAX_RUNNING_REQUESTS", winner.sgl)
    if current_sgl != winner.sgl:
        restart_service("sglang"); wait_healthy("sglang")
    restart_service("cpu"); wait_healthy("cpu")

    for c in LEGACY_FINAL_CONCURRENCIES:
        out = raw_dir / f"final-c{c}.json"
        log(f"final trial c={c} n={LEGACY_FINAL_TOTAL}")
        summary = run_bench(
            c, LEGACY_FINAL_TOTAL, urls_file, out, pool_seed=pool_seed,
        )
        t = Trial(f"final (c={c})", winner.omw, winner.sgl, cpu_w, cpu_t, summary)
        trials.append(t)
        log(
            f"  -> rps={t.rps:.3f} fail={t.fail_rate:.0%} "
            f"p50={t.p50:.0f}ms p95={t.p95:.0f}ms p99={t.p99:.0f}ms"
        )
    return trials


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out", default=None, help="report markdown path")
    ap.add_argument("--pool-size", type=int, default=128)
    ap.add_argument("--pool-seed", type=int, default=DEFAULT_POOL_SEED,
                    help="seed for pool construction + per-request draw; "
                         "use 0 or empty to keep it non-deterministic")
    ap.add_argument("--stage", choices=["a", "b", "c", "d", "e"], default=None,
                    help="a=1D scans, b=2D grid, c=c-curve verify, "
                         "d=focused combo grid with bottleneck attribution, "
                         "e=saturation c-curve at fixed winner config. "
                         "Default (None) = legacy OMW×SGL grid + c-curve.")
    ap.add_argument("--axes", default=None,
                    help="stage b axes, e.g. OCR_MAX_WORKERS,SGL_MAX_RUNNING_REQUESTS")
    ap.add_argument("--knobs", default=None,
                    help="stage a comma-separated subset of knobs to scan; "
                         "default = all SGL_* + OCR_CONN_POOL")
    ap.add_argument("--set", action="append", default=[],
                    help="stage c fixed-config override, e.g. "
                         "--set OCR_MAX_WORKERS=4 --set SGL_MAX_RUNNING_REQUESTS=24")
    ap.add_argument("--concurrencies", default=None,
                    help="stage c comma-separated client concurrencies; "
                         f"default = {STAGE_C_CONCURRENCIES}")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the trial matrix and exit without launching")
    ap.add_argument("--max-fail-rate", type=float, default=0.10,
                    help="abort a cell mid-bench when fails/attempted "
                         "exceeds this fraction. Default 0.10. Set to 0 "
                         "to disable (run every cell to N).")
    ap.add_argument("--min-sample-for-abort", type=int, default=40,
                    help="min observations before abort can trigger.")
    ap.add_argument("--total", type=int, default=None,
                    help="override per-cell sample count N. Applies to "
                         "stage-c and stage-e; default is STAGE_C_TOTAL / "
                         "STAGE_E_TOTAL (200).")
    ap.add_argument("--warmup", type=int, default=2,
                    help="warmup requests before the main run. Default 2. "
                         "Set to 0 when warmup is done externally and you "
                         "want /metrics histograms to reflect only the real "
                         "run.")
    args = ap.parse_args()
    max_fail = args.max_fail_rate if args.max_fail_rate > 0 else None

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    stage_tag = f"stage-{args.stage}" if args.stage else "tune"
    run_id = args.run_id or f"omnidoc-{ts}-{stage_tag}"
    report_path = pathlib.Path(args.out) if args.out else (
        REPO_ROOT / "loadtest" / "results" / f"{run_id}.md"
    )
    raw_dir = REPO_ROOT / "loadtest" / "results" / "raw" / run_id
    pool_seed = args.pool_seed if args.pool_seed else None

    # Snapshot the env vars we may touch. baseline = "what .env says now".
    tracked_keys = list(KNOBS.keys()) + ["CPU_WORKERS", "CPU_THREADS"]
    baseline = snapshot_env(tracked_keys)
    log(f"run_id={run_id}")
    log(f"baseline env: {baseline}")

    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="tune-"))
    log(f"tmp dir: {tmp_dir}")

    if not args.dry_run:
        raw_dir.mkdir(parents=True, exist_ok=True)

    urls_file = build_image_pool(args.pool_size, tmp_dir, pool_seed)
    log(f"pool_size={args.pool_size} pool_seed={pool_seed}")

    try:
        if args.stage == "a":
            scan = (args.knobs.split(",") if args.knobs else STAGE_A_SCAN_KNOBS)
            trials = stage_a(baseline, urls_file, raw_dir, pool_seed,
                             args.dry_run, scan,
                             max_fail_rate=max_fail,
                             min_sample_for_abort=args.min_sample_for_abort)
        elif args.stage == "b":
            if args.axes:
                x, y = args.axes.split(",", 1)
            else:
                x, y = "OCR_MAX_WORKERS", "SGL_MAX_RUNNING_REQUESTS"
            trials = stage_b((x.strip(), y.strip()), baseline, urls_file, raw_dir,
                             pool_seed, args.dry_run,
                             max_fail_rate=max_fail,
                             min_sample_for_abort=args.min_sample_for_abort)
        elif args.stage == "c":
            fixed = parse_config_overrides(args.set)
            conc = ([int(x) for x in args.concurrencies.split(",")]
                    if args.concurrencies else STAGE_C_CONCURRENCIES)
            trials = stage_c(fixed, conc, baseline, urls_file, raw_dir,
                             pool_seed, args.dry_run,
                             max_fail_rate=max_fail,
                             min_sample_for_abort=args.min_sample_for_abort,
                             total=args.total,
                             warmup=args.warmup)
        elif args.stage == "d":
            trials = stage_d(baseline, urls_file, raw_dir, pool_seed,
                             args.dry_run,
                             max_fail_rate=max_fail,
                             min_sample_for_abort=args.min_sample_for_abort)
        elif args.stage == "e":
            conc = ([int(x) for x in args.concurrencies.split(",")]
                    if args.concurrencies else None)
            trials = stage_e(baseline, urls_file, raw_dir, pool_seed,
                             args.dry_run,
                             max_fail_rate=max_fail,
                             min_sample_for_abort=args.min_sample_for_abort,
                             concurrencies=conc,
                             total=args.total,
                             warmup=args.warmup)
        else:
            trials = legacy_grid_and_final(baseline, urls_file, raw_dir, pool_seed)

        if args.dry_run:
            log("dry-run: no report written")
            return 0

        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_report(report_path, run_id, baseline, trials, args.pool_size,
                     pool_seed, args.stage, raw_dir)
        log(f"wrote report -> {report_path}")

        # Emit a combined JSON snapshot of all trials for downstream tooling
        # (render_report.py --input <file>, plotting scripts, etc.).
        trials_json = raw_dir / "_trials.json"
        trials_json.write_text(
            json.dumps([t.to_dict() for t in trials], indent=2), encoding="utf-8"
        )
        log(f"wrote trial index -> {trials_json}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return 0


# ---------------------------------------------------------------------------
# Report writer — legacy and staged.
# ---------------------------------------------------------------------------

def _latency_cols(t: Trial) -> str:
    return (f"{t.rps:.3f} | {t.p50:.0f} | {t.p95:.0f} | {t.p99:.0f} | "
            f"{t.mean_ms:.0f} | {t.wall:.1f}")


def _stage_a_section(trials: list[Trial]) -> str:
    rows = [
        "| knob | value | ok | fail | fail% | rps | p50 | p95 | p99 | mean | wall s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in trials:
        k = next(iter(t.knobs))
        v = t.knobs[k]
        s = t.summary
        rows.append(
            f"| `{k}` | {v} | {s.get('successes',0)} | {s.get('failures',0)} | "
            f"{t.fail_rate:.0%} | {_latency_cols(t)} |"
        )
    return "\n".join(rows)


def _stage_b_section(trials: list[Trial]) -> str:
    if not trials:
        return "_(no trials)_"
    axes_keys = list(trials[0].knobs.keys())
    rows = [
        "| " + " | ".join(axes_keys) + " | ok | fail | fail% | rps | p50 | p95 | p99 | mean | wall s |",
        "|" + "|".join(["---"] * len(axes_keys)) + "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in trials:
        s = t.summary
        vals = [t.knobs.get(k, "") for k in axes_keys]
        rows.append(
            "| " + " | ".join(vals) + f" | {s.get('successes',0)} | {s.get('failures',0)} | "
            f"{t.fail_rate:.0%} | {_latency_cols(t)} |"
        )
    return "\n".join(rows)


def _stage_c_section(trials: list[Trial]) -> str:
    rows = [
        "| c | ok | fail | fail% | rps | p50 | p95 | p99 | mean | wall s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in trials:
        s = t.summary
        c = s.get("concurrency", "?")
        rows.append(
            f"| {c} | {s.get('successes',0)} | {s.get('failures',0)} | "
            f"{t.fail_rate:.0%} | {_latency_cols(t)} |"
        )
    return "\n".join(rows)


def write_report(path, run_id, baseline, trials, pool_size, pool_seed,
                 stage, raw_dir):
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    if stage == "a":
        body = (
            f"# GLM-OCR stage-A 1D scans — {run_id}\n\n"
            f"**Completed:** {now}  \n"
            f"**Shape:** c={STAGE_A_CONCURRENCY}, N={STAGE_A_TOTAL}, "
            f"pool={pool_size}, seed={pool_seed}  \n"
            f"**Baseline:** `{baseline}`  \n\n"
            "One knob varies per row; all others stay at baseline. Use this "
            "to decide which axes deserve a Stage-B 2D fine-tune.\n\n"
            "## Results\n\n"
            f"{_stage_a_section(trials)}\n\n"
            f"Raw per-trial JSON: `{raw_dir}`\n"
        )
    elif stage == "b":
        body = (
            f"# GLM-OCR stage-B 2D grid — {run_id}\n\n"
            f"**Completed:** {now}  \n"
            f"**Shape:** c={STAGE_B_CONCURRENCY}, N={STAGE_B_TOTAL}, "
            f"pool={pool_size}, seed={pool_seed}  \n"
            f"**Baseline:** `{baseline}`  \n\n"
            f"{_stage_b_section(trials)}\n\n"
            f"Raw per-trial JSON: `{raw_dir}`\n"
        )
    elif stage == "c":
        fixed = trials[0].knobs if trials else {}
        body = (
            f"# GLM-OCR stage-C c-curve — {run_id}\n\n"
            f"**Completed:** {now}  \n"
            f"**N per c:** {STAGE_C_TOTAL}, pool={pool_size}, seed={pool_seed}  \n"
            f"**Fixed config:** `{fixed}`  \n\n"
            f"{_stage_c_section(trials)}\n\n"
            f"Raw per-trial JSON: `{raw_dir}`\n"
        )
    else:
        # Legacy 2D grid + final — keep the v1 output format.
        body = _legacy_report(run_id, now, baseline, trials, pool_size, pool_seed)

    path.write_text(body, encoding="utf-8")


def _legacy_report(run_id, now, baseline, trials, pool_size, pool_seed) -> str:
    grid_trials = [t for t in trials if t.kind == "grid"]
    final_trials = [t for t in trials if t.kind.startswith("final")]
    if not grid_trials:
        return f"# {run_id}\n\n_(no trials)_\n"

    grid_trials_sorted = sorted(grid_trials, key=lambda x: x.score(), reverse=True)
    winner = grid_trials_sorted[0]

    sgl_vals = sorted({t.sgl for t in grid_trials if t.sgl is not None})
    omw_vals = sorted({t.omw for t in grid_trials if t.omw is not None})
    idx = {(t.sgl, t.omw): t for t in grid_trials}
    heatmap_lines = [
        "| SGL \\ OMW | " + " | ".join(str(o) for o in omw_vals) + " |",
        "|---:|" + "---:|" * len(omw_vals),
    ]
    for sgl in sgl_vals:
        row = [f"**{sgl}**"]
        for omw in omw_vals:
            t = idx.get((sgl, omw))
            if t is None:
                row.append("—")
                continue
            marker = " ⭐" if (t.sgl == winner.sgl and t.omw == winner.omw) else ""
            row.append(f"{t.rps:.3f} ({t.fail_rate:.0%}){marker}")
        heatmap_lines.append("| " + " | ".join(row) + " |")
    heatmap = "\n".join(heatmap_lines)

    rows = [
        "| OMW | SGL | ok | fail | fail% | rps | p50 ms | p95 ms | p99 ms | mean ms | wall s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in sorted(grid_trials, key=lambda x: (x.sgl, x.omw)):
        s = t.summary
        rows.append(
            f"| {t.omw} | {t.sgl} | {s.get('successes',0)} | {s.get('failures',0)} | "
            f"{t.fail_rate:.0%} | {_latency_cols(t)} |"
        )
    grid_table = "\n".join(rows)

    final_rows = [
        "| c | ok | fail | fail% | rps | p50 ms | p95 ms | p99 ms | mean ms | wall s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in final_trials:
        s = t.summary
        c = s.get("concurrency", "?")
        final_rows.append(
            f"| {c} | {s.get('successes',0)} | {s.get('failures',0)} | "
            f"{t.fail_rate:.0%} | {_latency_cols(t)} |"
        )
    final_table = "\n".join(final_rows)

    return f"""# GLM-OCR 2D tuning grid — {run_id}

**Completed:** {now}
**Dataset:** OmniDocBench ({pool_size}-image pool, seed={pool_seed})
**Trial shape:** {LEGACY_TRIAL_TOTAL} requests at client concurrency {LEGACY_TRIAL_CONCURRENCY} per cell
**Final verification:** {LEGACY_FINAL_TOTAL} requests at c ∈ {LEGACY_FINAL_CONCURRENCIES}
**Ranking rule:** maximise rps subject to `failure_rate < {FAIL_RATE_GATE:.0%}`

## Setup

| Var | Value |
|---|---|
| `OCR_MAX_WORKERS` swept over | {LEGACY_OMW_VALUES} |
| `SGL_MAX_RUNNING_REQUESTS` swept over | {LEGACY_SGL_VALUES} |
| `CPU_WORKERS` (held) | {baseline.get('CPU_WORKERS')} |
| `CPU_THREADS` (held) | {baseline.get('CPU_THREADS')} |

## Winner

**`OCR_MAX_WORKERS={winner.omw}`  `SGL_MAX_RUNNING_REQUESTS={winner.sgl}`** — rps={winner.rps:.3f}, fail={winner.fail_rate:.0%}, p95={winner.p95:.0f}ms, p99={winner.p99:.0f}ms

## Heatmap — rps (fail%)

⭐ marks the winning cell. Rows = SGL cap; columns = OMW (fan-out).

{heatmap}

## Grid details

{grid_table}

## Final verification at winner

Re-run with OMW={winner.omw}, SGL={winner.sgl} at {LEGACY_FINAL_CONCURRENCIES}:

{final_table}

## Observability pointers

- Grafana dashboard: <http://localhost:3000/d/glmocr-load>
- Alloy UI: <http://localhost:12345/graph>
"""


if __name__ == "__main__":
    sys.exit(main())
