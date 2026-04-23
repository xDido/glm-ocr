"""Post-hoc augmentation for omnidoc asyncio-matrix reports.

Reads an existing matrix markdown report, parses the per-trial
(concurrency, interval, wall_seconds) from its table, walks Prometheus
`glmocr_in_flight_requests` to identify each trial's time window by
matching its wall duration and in-flight signature, then queries SGLang
and worker gauges over those windows and appends a "Runtime signals
(retroactive, per trial)" section to the report.

This script exists because:
  1. The matrix wrapper doesn't run `runtime_probe_loop.py` per trial,
     so there's no probe.jsonl the probe-mode renderer could consume.
  2. Grafana annotations silently fail on this stack (Grafana 11 admin
     lacks `annotations:create` by default), so we can't look up per-
     trial windows via the annotation API either.
  3. Prometheus has been scraping `glmocr_in_flight_requests`,
     `sglang:num_running_reqs`, `sglang:num_queue_reqs` the whole time,
     and trial in-flight signatures (plateau heights + durations) are
     distinctive enough to match each report row to a window.

Phase decomposition PNG is NOT produced by this script — it requires
per-request probe samples captured during the run, which we don't have
retroactively. Re-run the matrix with per-trial probing to get that.

Usage:
    python scripts/augment_matrix_report.py \
        [--report loadtest/results/omnidoc-<TS>-asyncio-matrix.md] \
        [--prometheus-url http://localhost:9090]

If `--report` is omitted, the newest `*-asyncio-matrix.md` under
loadtest/results/ is used.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics
import sys
import time
import urllib.parse
import urllib.request
from typing import Optional


# Metric names to pull over each trial window.
METRIC_IN_FLIGHT = "glmocr_in_flight_requests"
METRIC_SGL_RUN = "sglang:num_running_reqs"
METRIC_SGL_QUEUE = "sglang:num_queue_reqs"

# Per-trial phase decomposition — CPU container histograms.
# Each entry: (label, metric base, optional label selector clause).
# Selectors are injected into histogram queries. Scope notes clarify
# whether the histogram counts HTTP requests or per-region calls.
CPU_PHASES: list[tuple[str, str, str, str]] = [
    (
        "Flask end-to-end",
        "flask_http_request_duration_seconds",
        'url_rule="/glmocr/parse",status="200"',
        "per HTTP request",
    ),
    (
        "Layout forward (ONNX + pre/post)",
        "glmocr_layout_seconds",
        "",
        "per HTTP request",
    ),
    (
        "OCR region call",
        "glmocr_ocr_region_seconds",
        "",
        "per region (N per HTTP request)",
    ),
]

# Per-trial phase decomposition — SGLang container histograms.
SGLANG_PHASES: list[tuple[str, str, str, str]] = [
    (
        "SGLang end-to-end",
        "sglang:e2e_request_latency_seconds",
        "",
        "per SGLang request",
    ),
    ("Queue wait (scheduler)", "sglang:queue_time_seconds", "", "per SGLang request"),
    (
        "Time-to-first-token (prefill+first decode)",
        "sglang:time_to_first_token_seconds",
        "",
        "per SGLang request",
    ),
    (
        "Inter-token latency (decode step)",
        "sglang:inter_token_latency_seconds",
        "",
        "per decoded token",
    ),
]


def _prom_range(
    prom_url: str, query: str, start: float, end: float, step: int = 1
) -> list[tuple[float, float]]:
    qs = urllib.parse.urlencode(
        {
            "query": query,
            "start": int(start),
            "end": int(end),
            "step": step,
        }
    )
    try:
        with urllib.request.urlopen(
            f"{prom_url}/api/v1/query_range?{qs}",
            timeout=10,
        ) as r:
            d = json.load(r)
    except Exception as exc:
        print(f"[augment] prometheus query failed: {exc!r}", file=sys.stderr)
        return []
    if d.get("status") != "success" or not d.get("data", {}).get("result"):
        return []
    # Matrix queries may have multiple series (e.g. one per worker pid
    # for `glmocr_in_flight_requests`). Sum them at each timestamp so
    # the signature reflects total in-flight across all CPU workers.
    series_list = d["data"]["result"]
    timeline: dict[int, float] = {}
    for series in series_list:
        for t, v in series.get("values", []):
            ts = int(float(t))
            try:
                fv = float(v)
            except ValueError:
                continue
            timeline[ts] = timeline.get(ts, 0.0) + fv
    return sorted(timeline.items())


def _stats(values: list[float]) -> tuple[float, float, float]:
    """Return (mean, p95, peak). Empty → (nan, nan, nan)."""
    if not values:
        return (math.nan, math.nan, math.nan)
    s = sorted(values)
    mean = statistics.fmean(values)
    p95 = s[min(len(s) - 1, int(len(s) * 0.95))]
    peak = s[-1]
    return (mean, p95, peak)


def _fmt(x: float) -> str:
    if x != x:  # NaN
        return "—"
    if abs(x - round(x)) < 1e-6:
        return f"{int(round(x))}"
    return f"{x:.1f}"


def _fmt_ms(x: float) -> str:
    """Render a value already in milliseconds with thousands-separator
    and no decimals for magnitudes > 1 ms; '—' for NaN/empty."""
    if x != x:
        return "—"
    if x >= 1:
        return f"{int(round(x)):,}"
    return f"{x:.2f}"


# ---------------------------------------------------------------------------
# Prometheus histogram queries over an arbitrary [start_ts, end_ts] window.
# ---------------------------------------------------------------------------


def _prom_instant(prom_url: str, query: str, at_ts: int) -> float:
    """Execute an instant Prometheus query at `at_ts`. Returns the first
    scalar value from the result, or NaN on any failure / empty result."""
    qs = urllib.parse.urlencode({"query": query, "time": at_ts})
    try:
        with urllib.request.urlopen(
            f"{prom_url}/api/v1/query?{qs}",
            timeout=10,
        ) as r:
            d = json.load(r)
    except Exception:
        return math.nan
    if d.get("status") != "success":
        return math.nan
    result = d.get("data", {}).get("result") or []
    if not result:
        return math.nan
    try:
        return float(result[0]["value"][1])
    except (KeyError, IndexError, ValueError, TypeError):
        return math.nan


def _selector(labels: str) -> str:
    """Return `{labels}` if labels non-empty, else empty string."""
    return f"{{{labels}}}" if labels else ""


def _histogram_stats(
    prom_url: str,
    metric: str,
    labels: str,
    start_ts: int,
    end_ts: int,
) -> tuple[float, float, float, float]:
    """Return (mean_ms, p50_ms, p95_ms, p99_ms) for a Prometheus histogram
    evaluated over [start_ts, end_ts]. NaN for any value that can't be
    computed (metric absent, count==0, etc.).

    Uses `increase()` over the exact window duration — one number per
    phase per trial. All returned values are in MILLISECONDS (input
    histograms are seconds)."""
    duration = max(1, end_ts - start_ts)
    base_sel = _selector(labels)
    # Mean: total_time / total_count over the window.
    sum_q = f"sum(increase({metric}_sum{base_sel}[{duration}s]))"
    count_q = f"sum(increase({metric}_count{base_sel}[{duration}s]))"
    total_s = _prom_instant(prom_url, sum_q, end_ts)
    total_n = _prom_instant(prom_url, count_q, end_ts)
    if total_s != total_s or total_n != total_n or total_n <= 0:
        mean_ms = math.nan
    else:
        mean_ms = (total_s / total_n) * 1000.0

    # Quantiles via histogram_quantile on the bucket increase.
    # `sum by (le)` aggregates across scraped instances.
    def _q(q: float) -> float:
        qexpr = (
            f"histogram_quantile({q}, sum by (le) "
            f"(increase({metric}_bucket{base_sel}[{duration}s])))"
        )
        v = _prom_instant(prom_url, qexpr, end_ts)
        return v * 1000.0 if v == v else math.nan

    return (mean_ms, _q(0.50), _q(0.95), _q(0.99))


# ---------------------------------------------------------------------------
# Phase decomposition rendering.
# ---------------------------------------------------------------------------


def render_phase_section(
    trial: dict,
    window: Optional[tuple[int, int]],
    prom_url: str,
) -> str:
    c = trial["concurrency"]
    interval = trial["interval"]
    label = f"c={c}, i={interval:g}s" if interval > 0 else f"c={c} (back-to-back)"
    header = f"#### {label} — Mean request phase decomposition"

    if window is None:
        return header + "\n\n_(no matching window — phase decomposition unavailable)_\n"

    start_ts, end_ts = window

    def _block(title: str, phases: list[tuple[str, str, str, str]]) -> str:
        rows = [
            f"**{title}**",
            "",
            "| Phase | scope | mean ms | p50 | p95 | p99 |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for name, metric, selector, scope in phases:
            m, p50, p95, p99 = _histogram_stats(
                prom_url,
                metric,
                selector,
                start_ts,
                end_ts,
            )
            rows.append(
                f"| {name} | {scope} | {_fmt_ms(m)} | "
                f"{_fmt_ms(p50)} | {_fmt_ms(p95)} | {_fmt_ms(p99)} |"
            )
        return "\n".join(rows) + "\n"

    cpu_block = _block("CPU container", CPU_PHASES)
    sgl_block = _block("SGLang container", SGLANG_PHASES)

    note = (
        "\n_Values aggregate across all requests that completed inside "
        "the trial window. Scope indicates whether the histogram counts "
        "whole HTTP requests, per-region OCR fan-out calls, or per-token "
        "decode steps — percentiles are computed within that scope._\n"
    )
    return "\n".join([header, "", cpu_block, "", sgl_block, note])


# ---------------------------------------------------------------------------
# Matrix report parsing.
# ---------------------------------------------------------------------------

# Matches a data row of the sweep-mode table, e.g.:
#   | 12 | — | 100 | 0 | 100.0% | 57.4 | 1.742 | 5,100 | ...
# We only need the first three numeric/cell fields (c, interval, wall s).
# The table shape after the sweep-mode patch is:
#   c | interval (s) | ok | fail | fail % | wall s | rps | mean | min | p50 | p90 | p95 | p99 | max
_ROW_RE = re.compile(
    r"^\|\s*(\d+)\s*\|\s*([0-9.]+|—)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*[0-9.%]+\s*\|"
    r"\s*([0-9]+\.[0-9]+)\s*\|"
)


def parse_matrix_report(report_path: pathlib.Path) -> list[dict]:
    """Extract per-trial metadata from the sweep-mode table."""
    trials = []
    for line in report_path.read_text(encoding="utf-8").splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        c = int(m.group(1))
        iv_raw = m.group(2)
        interval = 0.0 if iv_raw == "—" else float(iv_raw)
        wall = float(m.group(3))
        trials.append({"concurrency": c, "interval": interval, "wall": wall})
    return trials


# ---------------------------------------------------------------------------
# Window discovery — Prometheus fingerprint matching.
#
# We walk `glmocr_in_flight_requests` over the last few hours, segment
# active periods, and greedy-match each trial to the segment whose
# (duration, peak) best fits its (wall_seconds, concurrency). This is
# more robust than trusting a matrix-level start timestamp — it handles
# runs that span discontinuous executions (e.g. a matrix that crashed
# mid-way and was re-stitched from a partial rerun, which is exactly
# the path that created THIS plan's input data).
# ---------------------------------------------------------------------------


def _segment_active(
    timeline: list[tuple[int, float]],
    idle_threshold: float = 0.5,
    max_idle_gap_seconds: int = 20,
) -> list[tuple[int, int, float]]:
    """Return (start_ts, end_ts, peak_value) for every active segment in
    the timeline. A contiguous run of at most `max_idle_gap_seconds` of
    below-threshold samples is tolerated inside a segment — this lets
    paced c=1 trials (which oscillate 0↔1 with up to interval_seconds of
    idle) stay as one segment, while sharply separating distinct trials
    that have ≥20 s of quiet between them (typical inter-trial overhead
    for warmup + setup)."""
    if not timeline:
        return []
    step = max(1, (timeline[-1][0] - timeline[0][0]) // max(1, len(timeline) - 1))
    segments: list[tuple[int, int, float]] = []
    in_seg = False
    seg_start = seg_end = 0
    peak = 0.0
    idle_s = 0
    for ts, v in timeline:
        if v > idle_threshold:
            if not in_seg:
                in_seg = True
                seg_start = ts
                peak = v
            seg_end = ts
            if v > peak:
                peak = v
            idle_s = 0
        else:
            if in_seg:
                idle_s += step
                if idle_s > max_idle_gap_seconds:
                    segments.append((seg_start, seg_end, peak))
                    in_seg = False
                    idle_s = 0
                    peak = 0.0
    if in_seg:
        segments.append((seg_start, seg_end, peak))
    return segments


def _fingerprint_score(
    segment: tuple[int, int, float],
    trial: dict,
) -> float:
    """Lower = better fit. Combines duration-fit (|Δ| / expected) and a
    peak mismatch penalty scaled to the trial's target concurrency."""
    start, end, peak = segment
    dur = max(1, end - start)
    expected_dur = max(1.0, trial["wall"])
    dur_err = abs(dur - expected_dur) / expected_dur

    c = trial["concurrency"]
    # For loaded trials the peak should land near c (or slightly below
    # if worker allocation was uneven). For paced c=1 trials the peak
    # should be 1. We penalize relative mismatch.
    if c >= 2:
        # Loaded — peak should be in a band around c, allow ±50%
        target = float(c)
        peak_err = abs(peak - target) / target
    else:
        peak_err = abs(peak - 1.0) / 1.0  # c=1 should peak at exactly 1
    return dur_err + 0.5 * peak_err


def find_windows_by_fingerprint(
    trials: list[dict],
    timeline: list[tuple[int, float]],
) -> dict[int, tuple[int, int]]:
    """Greedy assignment: sort trials by discriminating-power (longer
    and higher-concurrency first, as they're most distinctive), and for
    each trial pick the best-scoring still-available segment. Returns
    a mapping from trial index (into `trials`) to (start_ts, end_ts).
    Trials without a plausible match are absent from the return dict."""
    segments = _segment_active(timeline)
    # Rank trials by wall_seconds desc so the most-distinctive (longest)
    # claim their segment first, avoiding accidental steals by shorter
    # trials with duplicate signatures.
    trial_order = sorted(range(len(trials)), key=lambda i: -trials[i]["wall"])
    available = list(range(len(segments)))
    windows: dict[int, tuple[int, int]] = {}
    for ti in trial_order:
        trial = trials[ti]
        if not available:
            break
        best = None
        best_score = float("inf")
        for si in available:
            score = _fingerprint_score(segments[si], trial)
            if score < best_score:
                best_score = score
                best = si
        # Reject matches that are clearly wrong (score > 1.0 means the
        # duration alone was off by more than 100%).
        if best is None or best_score > 1.0:
            print(
                f"[augment] warning: trial#{ti} c={trial['concurrency']} "
                f"i={trial['interval']} wall={trial['wall']:.0f}s — no "
                f"plausible Prometheus segment found (best score="
                f"{best_score:.2f})",
                file=sys.stderr,
            )
            continue
        s = segments[best]
        windows[ti] = (s[0], s[1])
        available.remove(best)
        print(
            f"[augment] trial#{ti} c={trial['concurrency']} "
            f"i={trial['interval']} -> segment [{s[0]}..{s[1]}] "
            f"dur={s[1] - s[0]}s peak={s[2]:.0f} (score={best_score:.2f})",
            file=sys.stderr,
        )
    return windows


# ---------------------------------------------------------------------------
# Augmentation rendering.
# ---------------------------------------------------------------------------


def render_trial_section(
    trial: dict,
    window: Optional[tuple[int, int]],
    prom_url: str,
) -> str:
    c = trial["concurrency"]
    interval = trial["interval"]
    label = f"c={c}, i={interval:g}s" if interval > 0 else f"c={c} (back-to-back)"
    header = f"### {label}"

    if window is None:
        return (
            header
            + "\n\n_(no matching Prometheus window found — trial may have been too short to scrape)_\n"
        )

    start_ts, end_ts = window
    # Expand the window by 2 seconds on each side so we don't clip the
    # ramp-up/drain samples at trial boundaries.
    start_ts = max(0, start_ts - 2)
    end_ts = end_ts + 2

    in_flight_series = _prom_range(prom_url, METRIC_IN_FLIGHT, start_ts, end_ts)
    sgl_run_series = _prom_range(prom_url, METRIC_SGL_RUN, start_ts, end_ts)
    sgl_queue_series = _prom_range(prom_url, METRIC_SGL_QUEUE, start_ts, end_ts)

    if_vals = [v for _, v in in_flight_series]
    run_vals = [v for _, v in sgl_run_series]
    q_vals = [v for _, v in sgl_queue_series]

    if_mean, if_p95, if_peak = _stats(if_vals)
    r_mean, r_p95, r_peak = _stats(run_vals)
    q_mean, q_p95, q_peak = _stats(q_vals)

    worker_md = (
        "**Worker concurrency (CPU container):**\n\n"
        "| Metric | Value |\n"
        "|---|---:|\n"
        f"| target concurrency (c) | {c} |\n"
        f"| in-flight mean | {_fmt(if_mean)} |\n"
        f"| in-flight p95  | {_fmt(if_p95)} |\n"
        f"| in-flight peak | {_fmt(if_peak)} |\n"
        f"| probe samples  | {len(if_vals)} |\n"
    )

    sgl_md = (
        "\n**SGLang state (from Prometheus, live gauges):**\n\n"
        "| Signal | mean | p95 | peak |\n"
        "|---|---:|---:|---:|\n"
        f"| sglang running (in-GPU batch) | {_fmt(r_mean)} | {_fmt(r_p95)} | {_fmt(r_peak)} |\n"
        f"| sglang queued (waiting for slot) | {_fmt(q_mean)} | {_fmt(q_p95)} | {_fmt(q_peak)} |\n"
    )

    window_md = (
        f"\n_Window: {start_ts}–{end_ts} (duration {end_ts - start_ts} s; step=1 s)_\n"
    )
    return "\n".join([header, "", worker_md, sgl_md, window_md])


def build_config_advisory(
    trials: list[dict],
    matched: list[Optional[tuple[int, int]]],
    prom_url: str,
) -> str:
    """Cross-trial advisory that compares observed peak usage against
    the configured SGLang limits and suggests whether to raise or lower
    each knob. Uses the union of all trial windows so the "peak" is
    the hottest moment across the full matrix run."""
    windows = [w for w in matched if w is not None]
    if not windows:
        return ""

    min_start = min(w[0] for w in windows)
    max_end = max(w[1] for w in windows)
    dur = max(1, max_end - min_start)

    def q(expr: str, at: int) -> float:
        qs = urllib.parse.urlencode({"query": expr, "time": at})
        try:
            with urllib.request.urlopen(
                f"{prom_url}/api/v1/query?{qs}",
                timeout=10,
            ) as r:
                d = json.load(r)
            res = d.get("data", {}).get("result") or []
            return float(res[0]["value"][1]) if res else math.nan
        except Exception:
            return math.nan

    # Effective configured ceilings (SGLang applies its own profiled
    # cap; we report both the .env value and what actually got used).
    cfg_max_total = q("sglang:max_total_num_tokens", max_end)

    # Peak KV usage observed across the matrix window
    peak_used = q(f"max_over_time(sum(sglang:num_used_tokens)[{dur}s:5s])", max_end)
    # Peak concurrent running requests
    peak_running = q(
        f"max_over_time(sum(sglang:num_running_reqs)[{dur}s:5s])",
        max_end,
    )
    # Peak prompt + generation tokens in any single SGLang request.
    # histogram_quantile(1.0, ...) gives the bucket upper-bound of the
    # worst-observed request.
    worst_prompt = q(
        f"histogram_quantile(1.0, sum by (le) "
        f"(increase(sglang:e2e_request_latency_seconds_bucket[{dur}s])))",
        max_end,
    )
    # For per-request token length, the best proxy we have is the
    # total-tokens / total-requests ratio — plus the max-over-time of
    # the running-KV / running-req ratio as an overestimate.
    total_prompt = q(f"sum(increase(sglang:prompt_tokens_total[{dur}s]))", max_end)
    total_gen = q(f"sum(increase(sglang:generation_tokens_total[{dur}s]))", max_end)
    total_req = q(f"sum(increase(sglang:num_requests_total[{dur}s]))", max_end)
    mean_prompt = total_prompt / total_req if total_req > 0 else math.nan
    mean_gen = total_gen / total_req if total_req > 0 else math.nan
    mean_total = mean_prompt + mean_gen if total_req > 0 else math.nan

    # ---- Advisory logic -------------------------------------------------
    kv_util = (
        peak_used / cfg_max_total
        if cfg_max_total and cfg_max_total == cfg_max_total
        else math.nan
    )

    max_total_advice: str
    if kv_util != kv_util:
        max_total_advice = "—"
    elif kv_util >= 0.90:
        max_total_advice = (
            f"**Raise** — peak used {peak_used:.0f}/{cfg_max_total:.0f} "
            f"({kv_util * 100:.0f}%). KV pool is nearly saturated, so new "
            f"requests are queueing. Increase VRAM fraction or drop the "
            f"context length to grow the effective ceiling."
        )
    elif kv_util <= 0.40:
        max_total_advice = (
            f"**Consider lowering** — peak used only {peak_used:.0f}/{cfg_max_total:.0f} "
            f"({kv_util * 100:.0f}%). The configured ceiling is well above "
            f"what the workload actually needs; you could reclaim VRAM "
            f"for CUDA graphs / activations."
        )
    else:
        max_total_advice = (
            f"**Hold** — peak used {peak_used:.0f}/{cfg_max_total:.0f} "
            f"({kv_util * 100:.0f}%). Comfortable middle band."
        )

    # SGL_CONTEXT_LENGTH advice: compare mean/max per-request total tokens
    # to the configured context length. We only know the MEAN from
    # histograms; the real max isn't directly observable without a
    # `prompt_tokens_buckets` label on the histogram. Flag with a floor
    # of 2x mean as a conservative required context.
    required_ctx = mean_total * 2 if mean_total == mean_total else math.nan
    # Note: SGL_CONTEXT_LENGTH isn't in Prometheus; we have to read it
    # from the environment of the SGLang container if possible.
    # For now, emit a recommendation based on observed mean.
    ctx_advice: str
    if required_ctx != required_ctx:
        ctx_advice = "—"
    elif required_ctx < 1024:
        ctx_advice = (
            f"**Consider 2048** — observed mean per-request total "
            f"= {mean_total:.0f} tok (prompt {mean_prompt:.0f} + gen "
            f"{mean_gen:.0f}). Even 2x mean ({required_ctx:.0f}) fits "
            f"2048 with margin; setting context lower than the current "
            f"4096 would grow the KV pool further."
        )
    elif required_ctx < 4096:
        ctx_advice = (
            f"**Hold at 4096** — observed mean {mean_total:.0f} tok × 2 "
            f"= {required_ctx:.0f}. Current ceiling has comfortable headroom."
        )
    else:
        ctx_advice = (
            f"**Raise** — observed mean {mean_total:.0f} tok × 2 = "
            f"{required_ctx:.0f} which approaches 4096. A single long "
            f"prompt may start hitting the ceiling. Bump to 8192."
        )

    return (
        "## Config tuning advisory\n"
        "\n"
        "_Tracks observed peak usage against configured SGLang limits "
        "over the full matrix window, and advises whether each knob "
        "should be raised, lowered, or held. Numbers are read live from "
        "Prometheus._\n"
        "\n"
        "| Knob | .env value | Observed | Advice |\n"
        "|---|---:|---:|---|\n"
        f"| `sglang:max_total_num_tokens` (effective) | {int(cfg_max_total) if cfg_max_total == cfg_max_total else '—'} | peak used {int(peak_used) if peak_used == peak_used else '—'} ({kv_util * 100:.0f}% util) | {max_total_advice} |\n"
        f"| `SGL_CONTEXT_LENGTH` | 4096 (per .env) | mean req {int(mean_total) if mean_total == mean_total else '—'} tok (prompt {int(mean_prompt) if mean_prompt == mean_prompt else '—'} + gen {int(mean_gen) if mean_gen == mean_gen else '—'}) | {ctx_advice} |\n"
        f"| `SGL_MAX_RUNNING_REQUESTS` | (from .env) | peak {int(peak_running) if peak_running == peak_running else '—'} concurrent | Hold unless peak approaches the configured cap for long stretches. |\n"
        "\n"
        "_KV-pool utilization > 90% = saturated (new requests queue); "
        "< 40% = over-provisioned (reclaim VRAM). Context length should "
        "sit ~2x the mean per-request total tokens — the current 4096 "
        "leaves a safety margin over typical OCR loads._\n"
    )


def build_augmentation_section(
    trials: list[dict],
    matched: list[Optional[tuple[int, int]]],
    prom_url: str,
) -> str:
    parts = [
        build_config_advisory(trials, matched, prom_url),
        "",
        "## Runtime signals + phase decomposition (retroactive, per trial)",
        "",
        "_Pulled from Prometheus post-hoc. Each trial window is "
        "reconstructed from the matrix start timestamp + cumulative "
        "wall times in execution order. For each window we report two "
        "things: live gauges (in-flight / SGLang running / SGLang queued) "
        "and per-phase histogram statistics on the CPU and SGLang "
        "pipelines. Phase percentiles are computed within the scope of "
        "that phase's histogram (per HTTP request, per region call, or "
        "per decoded token)._",
        "",
    ]
    for trial, window in zip(trials, matched):
        parts.append(render_trial_section(trial, window, prom_url))
        parts.append(render_phase_section(trial, window, prom_url))
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------


def _find_latest_matrix_report(results_dir: pathlib.Path) -> Optional[pathlib.Path]:
    candidates = sorted(
        results_dir.glob("*-asyncio-matrix.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--report",
        type=pathlib.Path,
        default=None,
        help="matrix markdown report to augment; defaults to "
        "the newest *-asyncio-matrix.md under "
        "loadtest/results/",
    )
    p.add_argument("--prometheus-url", default="http://localhost:9090")
    p.add_argument(
        "--results-dir", type=pathlib.Path, default=pathlib.Path("loadtest/results")
    )
    p.add_argument(
        "--windows-json",
        type=pathlib.Path,
        default=None,
        help="Explicit per-trial windows. JSON file shaped as "
        '[{"c": 12, "interval": 0, "start": 1776778532, '
        '"end": 1776778820}, ...]. Used when Prometheus '
        "segment fingerprinting can't disambiguate trials "
        "(e.g. multiple paced c=1 trials in one coalesced "
        "segment). Overrides fingerprint matching for any "
        "(c, interval) pair listed here; fingerprint "
        "matching still applies to trials not in the file.",
    )
    args = p.parse_args()

    report = args.report or _find_latest_matrix_report(args.results_dir)
    if not report or not report.exists():
        print(
            f"[augment] no matrix report found under {args.results_dir}",
            file=sys.stderr,
        )
        return 1
    print(f"[augment] augmenting {report}")

    trials = parse_matrix_report(report)
    if not trials:
        print("[augment] no trial rows found in report table", file=sys.stderr)
        return 2
    print(f"[augment] parsed {len(trials)} trials from report")

    # Pull in-flight timeline over a wide window. 4 h is enough to cover
    # any sensible matrix execution plus rerun stitching. step=5 s keeps
    # the point count under Prometheus's default 11k-point cap while
    # preserving enough resolution for segment detection (segments are
    # typically ≥60 s).
    now = int(time.time())
    window_start = now - 4 * 60 * 60
    timeline = _prom_range(
        args.prometheus_url, METRIC_IN_FLIGHT, window_start, now, step=5
    )
    if not timeline:
        print(
            "[augment] no in-flight samples from Prometheus — is it up?",
            file=sys.stderr,
        )
        return 3

    # Fingerprint-match each trial to a Prometheus segment by
    # (duration, peak) signature. Robust to discontinuous executions.
    idx_windows = find_windows_by_fingerprint(trials, timeline)
    matched = [idx_windows.get(i) for i in range(len(trials))]

    # Explicit windows from JSON override fingerprint matches. This is
    # the escape hatch when segments collide (e.g. 2 paced c=1 trials
    # running back-to-back look like one segment to the fingerprinter).
    if args.windows_json:
        explicit = json.loads(args.windows_json.read_text(encoding="utf-8"))
        for entry in explicit:
            key = (int(entry["c"]), float(entry["interval"]))
            for i, t in enumerate(trials):
                if (t["concurrency"], t["interval"]) == key:
                    matched[i] = (int(entry["start"]), int(entry["end"]))
                    print(
                        f"[augment] explicit window override for "
                        f"c={key[0]} i={key[1]}: "
                        f"[{entry['start']}..{entry['end']}]",
                        file=sys.stderr,
                    )
                    break

    n_matched = sum(1 for w in matched if w is not None)
    print(f"[augment] matched {n_matched}/{len(trials)} trials to Prometheus windows")
    augmentation = build_augmentation_section(trials, matched, args.prometheus_url)

    # Append the new section before the final "## Observability pointers"
    # section if present, else append at the end.
    content = report.read_text(encoding="utf-8")
    marker = "## Observability pointers"
    if marker in content:
        head, tail = content.split(marker, 1)
        new_content = head + augmentation + "\n" + marker + tail
    else:
        new_content = content + "\n" + augmentation
    report.write_text(new_content, encoding="utf-8")
    print(f"[augment] appended runtime signals section to {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
