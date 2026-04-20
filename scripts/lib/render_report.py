"""Render a self-contained markdown report from load-test artifacts.

One invocation = one .md file. Intermediate JSON/JSONL files are NOT
kept by the driver scripts (orchestrators treat them as ephemeral), with
one exception: the staged sweep (tune_params.py --stage) writes a
per-run `_trials.json` plus per-trial raw JSONs under
`loadtest/results/raw/<run-id>/`, which this tool consumes in the
`stage` mode.

Four modes:

  simple  — one bench.json -> one .md (used by omnidoc_asyncio.sh)
  sweep   — many bench.json -> one .md with a comparison table
  probe   — bench.json + probe.jsonl + Prometheus backfill -> one .md
            with correlated CPU/SGLang signals during the run window.
  stage   — _trials.json from tune_params.py --stage -> one .md plus
            matplotlib PNGs. Handles stage a (1D scans), b (2D grid),
            and c (c-curve verification) based on trial `kind`.

The live queries (/runtime/summary, /metrics, Prometheus range) are
best-effort; a failure just yields a "(unavailable)" row rather than
aborting the report. matplotlib is an optional dep — if missing, the
stage mode still emits the markdown without inline charts.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

# Local-relative import — both `python scripts/lib/render_report.py` and
# `python -m scripts.lib.render_report` need to find parse_metrics.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import parse_metrics as pm  # noqa: E402


# ---------------------------------------------------------------------------
# HTTP helpers (best-effort — failures become "(unavailable)" in the report).
# ---------------------------------------------------------------------------

def _get_json(url: str, timeout: float = 3.0) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return {}


def _prom_range(prom_url: str, query: str, start: float, end: float,
                step: int = 30) -> list[tuple[float, float]]:
    qs = urllib.parse.urlencode({
        "query": query, "start": int(start), "end": int(end), "step": step,
    })
    try:
        with urllib.request.urlopen(
            f"{prom_url}/api/v1/query_range?{qs}", timeout=5,
        ) as r:
            d = json.load(r)
    except Exception:
        return []
    if d.get("status") != "success" or not d.get("data", {}).get("result"):
        return []
    return [
        (float(t), float(v))
        for t, v in d["data"]["result"][0].get("values", [])
    ]


# ---------------------------------------------------------------------------
# Formatting helpers.
# ---------------------------------------------------------------------------

def _pct(ok: int, total: int) -> str:
    return f"{100 * ok / total:.1f}%" if total else "—"


def _lat_rows(lat: dict) -> list[tuple[str, str]]:
    order = ("min", "p50", "p90", "p95", "p99", "max", "mean")
    rows = []
    for k in order:
        v = lat.get(k)
        rows.append((k, f"{v:,.0f}" if isinstance(v, (int, float)) else "—"))
    return rows


def _fmt_table(rows: list[tuple[str, str]], headers: tuple[str, str]) -> str:
    lines = [
        f"| {headers[0]} | {headers[1]} |",
        "|---|---:|",
    ]
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def _server_table(cpu_url: str) -> str:
    """Query /runtime/summary and format a side-by-side env/actual table.

    Fields exposed by runtime_app._install(_summary_) with the keys used
    in the Grafana dashboard's config-knobs section.
    """
    sm = _get_json(f"{cpu_url}/runtime/summary")
    if not sm:
        return "_(/runtime/summary unavailable — CPU container not responding)_"

    def row(label: str, block_key: str, sub_env: str = "env",
            sub_actual: str | tuple[str, ...] = "actual") -> tuple[str, str, str]:
        b = sm.get(block_key) or {}
        env_v = b.get(sub_env)
        if isinstance(sub_actual, tuple):
            actual_v = next((b.get(k) for k in sub_actual if k in b), None)
        else:
            actual_v = b.get(sub_actual)
        return (
            label,
            "—" if env_v is None else str(env_v),
            "—" if actual_v is None else str(actual_v),
        )

    data = [
        row("CPU_WORKERS", "cpu_workers"),
        row("CPU_THREADS (per worker)", "cpu_threads_per_worker",
            sub_actual="actual_per_worker"),
        row("OCR_MAX_WORKERS", "ocr_max_workers", sub_actual="config"),
        row("SGL_MAX_RUNNING_REQUESTS", "sglang_max_running",
            sub_actual=("runtime", "live_running")),
        row("SGL_MAX_TOTAL_TOKENS", "sglang_batch_tokens",
            sub_env="env_total", sub_actual="runtime_total"),
        row("SGL_MAX_PREFILL_TOKENS", "sglang_batch_tokens",
            sub_env="env_prefill", sub_actual="runtime_prefill"),
        row("SGL_DTYPE", "sglang_dtype"),
        row("SGL_TP_SIZE", "sglang_tp_size"),
        row("SGL_MEM_FRACTION_STATIC", "sglang_mem_fraction"),
        row("SGL_MODEL_PATH", "sglang_model"),
    ]

    lines = [
        "| Var | env (.env) | actual (live) |",
        "|---|---|---|",
    ]
    for label, env_v, actual_v in data:
        lines.append(f"| `{label}` | {env_v} | {actual_v} |")
    return "\n".join(lines)


def _errors_section(summary: dict) -> str:
    failures = summary.get("failures", 0)
    samples = summary.get("error_samples") or []
    if failures == 0:
        return "No failures. ✓"
    lines = [f"**{failures} failures.** First {len(samples)} samples:", ""]
    for s in samples:
        lines.append(f"- `{s}`")
    return "\n".join(lines)


def _headline_table(summary: dict) -> str:
    ok = summary.get("successes", 0)
    fail = summary.get("failures", 0)
    total = summary.get("total", ok + fail)
    wall = summary.get("wall_seconds", 0.0)
    rps = summary.get("throughput_rps", 0.0)
    rows = [
        ("Successes", f"{ok:,} / {total:,} ({_pct(ok, total)})"),
        ("Failures", f"{fail:,}"),
        ("Wall time", f"{wall:,.1f} s"),
        ("Throughput (successful)", f"{rps:.3f} req/s"),
    ]
    return _fmt_table(rows, ("Metric", "Value"))


def _observability_section(run_id: str) -> str:
    return "\n".join([
        "- Grafana dashboard: <http://localhost:3000/d/glmocr-load>",
        f"- Pushgateway: <http://localhost:9091/metrics> (job=`glmocr_asyncio`, run_id=`{run_id}`)",
        "- Alloy UI: <http://localhost:12345/graph>",
    ])


# ---------------------------------------------------------------------------
# Mode: simple — single bench run.
# ---------------------------------------------------------------------------

def render_simple(args: argparse.Namespace) -> None:
    summary = json.loads(pathlib.Path(args.bench).read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    md = f"""# GLM-OCR asyncio load test — {args.run_id}

**Completed:** {now}
**Driver:** asyncio (`loadtest/asyncio/bench.py`)
**Endpoint:** `{summary.get("host", "")}{summary.get("endpoint", "")}`
**Dataset:** OmniDocBench ({args.pool_size}-image pool)

## Test parameters

{_fmt_table([
    ("Concurrency", str(summary.get("concurrency", "—"))),
    ("Total requests", str(summary.get("total", "—"))),
    ("Image pool", str(args.pool_size)),
    ("Warmup", "2"),
    ("Per-request timeout", "300 s"),
], ("Knob", "Value"))}

## Server runtime

{_server_table(args.cpu_url)}

## Headline

{_headline_table(summary)}

## Latency (successful requests only, milliseconds)

{_fmt_table(_lat_rows(summary.get("latency_ms") or {}), ("Percentile", "Latency (ms)"))}

## Errors

{_errors_section(summary)}

## Observability pointers

{_observability_section(args.run_id)}
"""
    pathlib.Path(args.out).write_text(md, encoding="utf-8")
    print(f"[render_report] wrote {args.out}")


# ---------------------------------------------------------------------------
# Mode: sweep — multiple bench runs, one .md with a comparison table.
# ---------------------------------------------------------------------------

def render_sweep(args: argparse.Namespace) -> None:
    bench_paths = [pathlib.Path(p) for p in args.bench]
    summaries = []
    for p in sorted(bench_paths, key=lambda pp: json.loads(pp.read_text())["concurrency"]):
        try:
            summaries.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"[render_report] skipping {p}: {exc!r}", file=sys.stderr)

    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    host = summaries[0].get("host", "") if summaries else ""

    # Comparison table rows. Mean is shown because it is the stable
    # complement to wall (wall is max-over-slots and noise-dominated at
    # small N; mean is population-average and converges faster).
    hdr = ("| c | ok | fail | fail % | wall s | rps | mean | min | p50 | p90 | p95 | p99 | max |")
    sep = ("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    rows = [hdr, sep]
    for s in summaries:
        ok = s["successes"]
        fail = s["failures"]
        tot = s["total"]
        lat = s["latency_ms"]
        rows.append(
            f"| {s['concurrency']} | {ok} | {fail} | {_pct(ok, tot)} | "
            f"{s['wall_seconds']:.1f} | {s['throughput_rps']:.3f} | "
            f"{lat.get('mean', float('nan')):,.0f} | "
            f"{lat['min']:,.0f} | {lat['p50']:,.0f} | {lat['p90']:,.0f} | "
            f"{lat['p95']:,.0f} | {lat['p99']:,.0f} | {lat['max']:,.0f} |"
        )
    table = "\n".join(rows)

    # Per-level error samples.
    err_sections = []
    for s in summaries:
        c = s["concurrency"]
        failures = s.get("failures", 0)
        samples = s.get("error_samples") or []
        if failures == 0:
            err_sections.append(f"- **c={c}**: no failures ✓")
        else:
            err_sections.append(
                f"- **c={c}** ({failures} failures): "
                + ", ".join(f"`{x}`" for x in samples[:3])
            )
    errs = "\n".join(err_sections)

    concurrencies = ", ".join(str(s["concurrency"]) for s in summaries)

    md = f"""# GLM-OCR asyncio concurrency sweep — {args.run_id}

**Completed:** {now}
**Driver:** asyncio (`loadtest/asyncio/bench.py`)
**Endpoint:** `{host}{summaries[0].get("endpoint", "") if summaries else ""}`
**Dataset:** OmniDocBench ({args.pool_size}-image pool)
**Concurrency levels:** {concurrencies}

## Server runtime

{_server_table(args.cpu_url)}

## Results — all levels

_Latencies in ms; rps counts successes only. The **ceiling** is the
concurrency where rps stops rising — past that, extra concurrency
just adds queuing and failures._

{table}

## Errors per level

{errs}

## Observability pointers

{_observability_section(args.run_id)}
"""
    pathlib.Path(args.out).write_text(md, encoding="utf-8")
    print(f"[render_report] wrote {args.out}")


# ---------------------------------------------------------------------------
# Mode: probe — bench.json + probe.jsonl + Prometheus backfill correlation.
# ---------------------------------------------------------------------------

def _summarize_series(values: list[float]) -> tuple[float, float, float, int]:
    if not values:
        return (float("nan"), float("nan"), float("nan"), 0)
    s = sorted(values)
    mean = statistics.fmean(values)
    mx = max(values)
    p95 = s[min(len(s) - 1, int(len(s) * 0.95))]
    return (mean, p95, mx, len(values))


def render_probe(args: argparse.Namespace) -> None:
    summary = json.loads(pathlib.Path(args.bench).read_text(encoding="utf-8"))
    probe_lines = pathlib.Path(args.probe).read_text(encoding="utf-8").splitlines()
    probe = [json.loads(l) for l in probe_lines if l.strip()]

    # Pull the run window from probe timestamps.
    run_start = probe[0]["ts"] if probe else time.time() - summary.get("wall_seconds", 0)
    run_end = probe[-1]["ts"] if probe else time.time()

    # Probe-side signals (in_flight is reliable; sglang_* may or may not
    # be populated depending on whether SGLang's /metrics is reachable).
    in_flight = [p["in_flight"] for p in probe if isinstance(p.get("in_flight"), (int, float))]
    sgl_run_probe = [p["sglang_running"] for p in probe if isinstance(p.get("sglang_running"), (int, float))]
    sgl_queue_probe = [p["sglang_queued"] for p in probe if isinstance(p.get("sglang_queued"), (int, float))]

    # Backfill sglang from Prometheus if the probe didn't capture it.
    if not sgl_run_probe:
        sgl_run = [v for _, v in _prom_range(
            args.prom_url,
            'sglang:num_running_reqs{model_name="glm-ocr"}',
            run_start, run_end,
        )]
    else:
        sgl_run = sgl_run_probe

    if not sgl_queue_probe:
        sgl_queue = [v for _, v in _prom_range(
            args.prom_url,
            'sglang:num_queue_reqs{model_name="glm-ocr"}',
            run_start, run_end,
        )]
    else:
        sgl_queue = sgl_queue_probe

    def stats_row(name: str, vals: list[float]) -> str:
        mean, p95, mx, n = _summarize_series(vals)
        if n == 0:
            return f"| {name} | — | — | — | 0 |"
        return f"| {name} | {mean:.2f} | {p95:.1f} | {mx:.1f} | {n} |"

    signal_table = "\n".join([
        "| Signal | mean | p95 | max | samples |",
        "|---|---:|---:|---:|---:|",
        stats_row("CPU `glmocr_in_flight_requests`", in_flight),
        stats_row("SGLang `sglang:num_running_reqs`", sgl_run),
        stats_row("SGLang `sglang:num_queue_reqs`", sgl_queue),
    ])

    # Derived ratios if both series exist.
    derived = ""
    if in_flight and sgl_run:
        m_if = statistics.fmean(in_flight)
        m_sg = statistics.fmean(sgl_run)
        derived = (
            f"\n**Fan-out ratio** (mean sglang_running ÷ mean in_flight) "
            f"≈ **{m_sg / m_if:.2f}** — each client request produces ~this "
            f"many parallel SGLang calls.\n"
        )

    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    md = f"""# GLM-OCR asyncio probe run — {args.run_id}

**Completed:** {now}
**Driver:** asyncio (`loadtest/asyncio/bench.py`) + runtime probe
**Endpoint:** `{summary.get("host", "")}{summary.get("endpoint", "")}`
**Dataset:** OmniDocBench ({args.pool_size}-image pool)

## Test parameters

{_fmt_table([
    ("Concurrency", str(summary.get("concurrency", "—"))),
    ("Total requests", str(summary.get("total", "—"))),
    ("Image pool", str(args.pool_size)),
    ("Probe interval", "2 s"),
    ("Warmup", "2"),
], ("Knob", "Value"))}

## Server runtime

{_server_table(args.cpu_url)}

## Headline

{_headline_table(summary)}

## Latency (successful requests only, milliseconds)

{_fmt_table(_lat_rows(summary.get("latency_ms") or {}), ("Percentile", "Latency (ms)"))}

## Correlated signals during the run window
{derived}
{signal_table}

## Errors

{_errors_section(summary)}

## Observability pointers

{_observability_section(args.run_id)}
"""
    pathlib.Path(args.out).write_text(md, encoding="utf-8")
    print(f"[render_report] wrote {args.out}")


# ---------------------------------------------------------------------------
# Mode: stage — reads _trials.json from tune_params.py --stage and emits
# a stage-specific markdown + inline matplotlib PNGs.
# ---------------------------------------------------------------------------

# SGLang/OCR knob glossary surfaced in every stage report so readers can
# interpret the axes without bouncing to the README.
KNOB_GLOSSARY = [
    ("SGL_SPECULATIVE",
     "NEXTN/MTP speculative decoding (shipped with GLM-OCR weights). "
     "Claimed ~2–4× decode throughput at the cost of KV-cache + "
     "draft-model memory."),
    ("SGL_SPEC_NUM_STEPS",
     "Draft-model token lookahead depth per verification step. Higher "
     "= more aggressive speculation, more wasted work on rejects."),
    ("SGL_SPEC_NUM_DRAFT_TOKENS",
     "Tokens committed per accepted draft. Too low → spec overhead "
     "not amortized; too high → more frequent rejects."),
    ("SGL_SPEC_EAGLE_TOPK",
     "EAGLE draft candidate fan-out. Usually 1 for NEXTN; higher only "
     "pays off with true-EAGLE draft models."),
    ("SGL_MEM_FRACTION_STATIC",
     "Fraction of GPU memory reserved for model weights + activations. "
     "The rest (1 - this) is KV cache + spec draft cache. With spec "
     "on, lower values give more cache headroom."),
    ("SGL_MAX_RUNNING_REQUESTS",
     "Max requests SGLang batches simultaneously on the GPU. Too low → "
     "client queue; too high → KV cache thrash."),
    ("SGL_MAX_PREFILL_TOKENS",
     "Tokens processed per prefill step. Too low → prompts chunked "
     "across more steps; too high → blocks decode longer."),
    ("SGL_MAX_TOTAL_TOKENS",
     "Total KV-cache slots across all in-flight requests. Too low → "
     "evictions; too high → steals VRAM from weights."),
    ("SGL_CHUNKED_PREFILL",
     "Interleave prefill with decode. Usually a win for mixed "
     "concurrency; verify with the ablation."),
    ("SGL_SCHEDULE_POLICY",
     "`lpm` = longest-prefix-match (helps shared-prompt workloads); "
     "`fcfs` = first-come-first-serve. OCR has unique images so `lpm` "
     "≈ `fcfs`."),
    ("OCR_CONN_POOL",
     "aiohttp pool size between the CPU container and SGLang. Must be "
     "≥ CPU_THREADS × OCR_MAX_WORKERS. Undersized → 503s; oversized → "
     "wasted RAM."),
    ("OCR_MAX_WORKERS",
     "Parallel SGLang calls per document (region fan-out). The CPU "
     "side's answer to how many regions to submit at once."),
]


def _knob_glossary_md() -> str:
    lines = [
        "| Knob | Effect |",
        "|---|---|",
    ]
    for k, desc in KNOB_GLOSSARY:
        lines.append(f"| `{k}` | {desc} |")
    return "\n".join(lines)


def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _trial_mean(t: dict) -> float:
    return float((t.get("summary", {}).get("latency_ms") or {}).get("mean", float("nan")))


def _trial_rps(t: dict) -> float:
    return float(t.get("summary", {}).get("throughput_rps", 0.0))


def _trial_fail_rate(t: dict) -> float:
    # Aborted cells stopped before hitting `total`, so use the actual
    # attempted count as the denominator to match bench.py's abort
    # logic (fails/attempted, not fails/target-N).
    s = t.get("summary", {})
    attempted = s.get("requests_attempted") or s.get("total") or 1
    return float(s.get("failures", 0)) / attempted


# Production SLO targets — any cell whose p99 or fail-rate exceeds these
# is NOT a candidate for winner selection, regardless of rps. Override via
# RENDER_SLO_P99_MS / RENDER_SLO_MAX_FAIL env vars if your service tier
# differs.
SLO_P99_MS = 120_000.0
SLO_MAX_FAIL = 0.10


def _trial_meets_slo(t: dict) -> bool:
    s = t.get("summary", {})
    if s.get("aborted"):
        return False
    lat = s.get("latency_ms") or {}
    p99 = float(lat.get("p99") or 0.0)
    return p99 <= SLO_P99_MS and _trial_fail_rate(t) <= SLO_MAX_FAIL


def _trial_p95(t: dict) -> float:
    return float((t.get("summary", {}).get("latency_ms") or {}).get("p95", float("nan")))


def _trial_p50(t: dict) -> float:
    return float((t.get("summary", {}).get("latency_ms") or {}).get("p50", float("nan")))


def _trial_p99(t: dict) -> float:
    return float((t.get("summary", {}).get("latency_ms") or {}).get("p99", float("nan")))


def _trial_max(t: dict) -> float:
    return float((t.get("summary", {}).get("latency_ms") or {}).get("max", float("nan")))


def _trial_wall(t: dict) -> float:
    return float(t.get("summary", {}).get("wall_seconds", 0.0))


def _render_stage_a_chart(plt, trials_by_knob: dict[str, list[dict]],
                          png_dir: pathlib.Path) -> dict[str, str]:
    """One PNG per knob: rps + mean latency on twin-y axes."""
    rel_paths: dict[str, str] = {}
    for knob, trials in trials_by_knob.items():
        # Trials may store the value under t["knobs"][knob].
        xs_raw = [t["knobs"].get(knob, "") for t in trials]
        # Attempt numeric x-axis; fall back to string/categorical.
        try:
            xs: list = [float(x) for x in xs_raw]
            categorical = False
        except (TypeError, ValueError):
            xs = list(xs_raw)
            categorical = True

        rps = [_trial_rps(t) for t in trials]
        mean_s = [_trial_mean(t) / 1000.0 for t in trials]

        fig, ax1 = plt.subplots(figsize=(6, 3.4))
        if categorical:
            ax1.plot(range(len(xs)), rps, "o-", color="#1f77b4", label="rps")
            ax1.set_xticks(range(len(xs)))
            ax1.set_xticklabels(xs)
        else:
            ax1.plot(xs, rps, "o-", color="#1f77b4", label="rps")
        ax1.set_xlabel(knob)
        ax1.set_ylabel("rps (successes)", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        if categorical:
            ax2.plot(range(len(xs)), mean_s, "s--", color="#d62728", label="mean latency (s)")
        else:
            ax2.plot(xs, mean_s, "s--", color="#d62728", label="mean latency (s)")
        ax2.set_ylabel("mean latency (s)", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")

        ax1.set_title(f"Stage A — {knob}")
        fig.tight_layout()
        out_png = png_dir / f"stage-a-{knob}.png"
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        rel_paths[knob] = out_png.name
    return rel_paths


def _render_stage_b_heatmaps(plt, trials: list[dict], png_dir: pathlib.Path) -> dict[str, str]:
    """Three heatmaps: rps, p95, fail%. Expects every trial to share the
    same two knob axes."""
    if not trials:
        return {}
    axis_keys = list(trials[0]["knobs"].keys())
    if len(axis_keys) != 2:
        return {}
    x_knob, y_knob = axis_keys
    x_vals = sorted({t["knobs"][x_knob] for t in trials}, key=_maybe_num)
    y_vals = sorted({t["knobs"][y_knob] for t in trials}, key=_maybe_num)
    idx = {(t["knobs"][x_knob], t["knobs"][y_knob]): t for t in trials}

    def grid(f) -> list[list[float]]:
        return [[f(idx[(xv, yv)]) if (xv, yv) in idx else float("nan")
                 for xv in x_vals] for yv in y_vals]

    # Track aborted cells so we can stamp them on the heatmap.
    aborted_grid = [[bool(idx[(xv, yv)]["summary"].get("aborted"))
                     if (xv, yv) in idx else False
                     for xv in x_vals] for yv in y_vals]

    rel_paths: dict[str, str] = {}
    for metric_name, fn, fmt in [
        ("rps",   _trial_rps,        "{:.3f}"),
        ("p95",   _trial_p95,        "{:.0f}"),
        ("fail%", _trial_fail_rate,  "{:.0%}"),
    ]:
        values = grid(fn)
        fig, ax = plt.subplots(figsize=(max(4, len(x_vals) + 2), max(3, len(y_vals) + 1)))
        im = ax.imshow(values, aspect="auto", cmap="viridis" if metric_name != "fail%" else "Reds")
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(x_vals)
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels(y_vals)
        ax.set_xlabel(x_knob)
        ax.set_ylabel(y_knob)
        ax.set_title(f"Stage B — {metric_name}  (✖ = aborted early)")
        for yi in range(len(y_vals)):
            for xi in range(len(x_vals)):
                v = values[yi][xi]
                if v != v:  # NaN: empty cell
                    continue
                label = fmt.format(v)
                if aborted_grid[yi][xi]:
                    label += "\n✖"
                ax.text(xi, yi, label, ha="center", va="center",
                        color="white", fontsize=9)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        safe = metric_name.replace("%", "pct")
        out_png = png_dir / f"stage-b-{safe}.png"
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        rel_paths[metric_name] = out_png.name
    return rel_paths


def _maybe_num(v):
    try:
        return (0, float(v))
    except (TypeError, ValueError):
        return (1, str(v))


def _render_stage_c_chart(plt, trials: list[dict], png_dir: pathlib.Path) -> str:
    """rps + p50/p95/p99 + wall vs concurrency. Two subplots so the
    latency lines don't get squashed by the wall magnitude."""
    c = [int(t["summary"].get("concurrency", 0)) for t in trials]
    rps = [_trial_rps(t) for t in trials]
    p50 = [_trial_p50(t) / 1000.0 for t in trials]
    p95 = [_trial_p95(t) / 1000.0 for t in trials]
    p99 = [_trial_p99(t) / 1000.0 for t in trials]
    wall = [_trial_wall(t) for t in trials]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.plot(c, rps, "o-", color="#1f77b4", label="rps")
    ax1_r = ax1.twinx()
    ax1_r.plot(c, wall, "s--", color="#2ca02c", label="wall (s)")
    ax1.set_xlabel("concurrency")
    ax1.set_ylabel("rps", color="#1f77b4")
    ax1_r.set_ylabel("wall (s)", color="#2ca02c")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Throughput + wall vs concurrency")

    ax2.plot(c, p50, "o-", label="p50")
    ax2.plot(c, p95, "s-", label="p95")
    ax2.plot(c, p99, "^-", label="p99")
    ax2.set_xlabel("concurrency")
    ax2.set_ylabel("latency (s)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Latency percentiles vs concurrency")

    fig.tight_layout()
    out_png = png_dir / "stage-c-curve.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png.name


def _render_resource_chart(plt, probe_path: pathlib.Path,
                           png_dir: pathlib.Path) -> str | None:
    """Time-series PNG for RAM / VRAM / CPU% / GPU%. Skips if the probe
    file is missing fields (older probe runs)."""
    try:
        lines = probe_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    samples = [json.loads(l) for l in lines if l.strip()]
    if not samples:
        return None
    has_resources = any(
        s.get("cpu_cores_cpu") is not None or s.get("vram_used_mb") is not None
        for s in samples
    )
    if not has_resources:
        return None

    t0 = samples[0]["ts"]
    ts_rel = [s["ts"] - t0 for s in samples]

    def series(key: str) -> list[float | None]:
        return [s.get(key) for s in samples]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)

    def plot_one(ax, keys_labels, ylabel, title):
        for key, label in keys_labels:
            vals = series(key)
            xs = [x for x, v in zip(ts_rel, vals) if v is not None]
            ys = [v for v in vals if v is not None]
            if xs:
                ax.plot(xs, ys, label=label)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plot_one(axes[0][0], [("cpu_cores_cpu", "cpu ctr"),
                          ("cpu_cores_sglang", "sglang ctr")],
             "cores", "Container CPU (rate of container_cpu_usage_seconds_total)")
    plot_one(axes[0][1], [("mem_rss_cpu_mb", "cpu ctr"),
                          ("mem_rss_sglang_mb", "sglang ctr")],
             "MiB", "Container RSS")
    plot_one(axes[1][0], [("vram_used_mb", "VRAM used (MiB)")],
             "MiB", "GPU VRAM (DCGM_FI_DEV_FB_USED)")
    plot_one(axes[1][1], [("gpu_util_pct", "GPU util %")],
             "%", "GPU utilization (DCGM_FI_DEV_GPU_UTIL)")
    axes[1][0].set_xlabel("seconds since probe start")
    axes[1][1].set_xlabel("seconds since probe start")
    fig.tight_layout()
    out_png = png_dir / "resource-usage.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png.name


def _resource_summary_table(probe_path: pathlib.Path) -> str:
    try:
        lines = probe_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return "_(no probe file — resource summary unavailable)_"
    samples = [json.loads(l) for l in lines if l.strip()]
    if not samples:
        return "_(probe file empty)_"

    def stats(key: str, scale: float = 1.0):
        vals = [s.get(key) for s in samples if isinstance(s.get(key), (int, float))]
        if not vals:
            return None
        vals = [v * scale for v in vals]
        return (statistics.fmean(vals), max(vals), len(vals))

    rows = [("CPU container cores", stats("cpu_cores_cpu"), "{:.2f}"),
            ("SGLang container cores", stats("cpu_cores_sglang"), "{:.2f}"),
            ("CPU container RSS MiB", stats("mem_rss_cpu_mb"), "{:.0f}"),
            ("SGLang container RSS MiB", stats("mem_rss_sglang_mb"), "{:.0f}"),
            ("GPU VRAM used MiB", stats("vram_used_mb"), "{:.0f}"),
            ("GPU utilization %", stats("gpu_util_pct"), "{:.0f}")]

    md = ["| Signal | mean | max | samples |", "|---|---:|---:|---:|"]
    for label, st, fmt in rows:
        if st is None:
            md.append(f"| {label} | — | — | 0 |")
        else:
            mean, mx, n = st
            md.append(f"| {label} | {fmt.format(mean)} | {fmt.format(mx)} | {n} |")
    return "\n".join(md)


def _stage_type_from_trials(trials: list[dict]) -> str:
    kinds = {t.get("kind", "") for t in trials}
    if any(k.startswith("stage-a:") for k in kinds):
        return "a"
    if "stage-b" in kinds:
        return "b"
    if any(k.startswith("stage-c (") for k in kinds):
        return "c"
    if "stage-d" in kinds:
        return "d"
    if any(k.startswith("stage-e (") for k in kinds):
        return "e"
    if "stage-g" in kinds:
        return "g"
    return "unknown"


def render_stage(args: argparse.Namespace) -> None:
    trials = json.loads(pathlib.Path(args.trials).read_text(encoding="utf-8"))
    stage = args.stage or _stage_type_from_trials(trials)
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    out_md = pathlib.Path(args.out)
    png_dir = out_md.parent / f"{out_md.stem}.d"
    png_dir.mkdir(parents=True, exist_ok=True)
    plt = _try_import_matplotlib()
    if plt is None:
        print("[render_report] matplotlib missing — markdown will omit PNG charts "
              "(pip install matplotlib to enable)", file=sys.stderr)

    resource_png = None
    resource_table = ""
    if args.probe:
        probe_path = pathlib.Path(args.probe)
        resource_table = _resource_summary_table(probe_path)
        if plt is not None:
            resource_png = _render_resource_chart(plt, probe_path, png_dir)

    if stage == "a":
        body = _render_stage_a(args, trials, plt, png_dir, now,
                               resource_png, resource_table)
    elif stage == "b":
        body = _render_stage_b(args, trials, plt, png_dir, now,
                               resource_png, resource_table)
    elif stage == "c":
        body = _render_stage_c(args, trials, plt, png_dir, now,
                               resource_png, resource_table)
    elif stage == "d":
        body = _render_stage_d(args, trials, plt, png_dir, now,
                               resource_png, resource_table)
    elif stage == "e":
        body = _render_stage_e(args, trials, plt, png_dir, now,
                               resource_png, resource_table)
    elif stage == "g":
        body = _render_stage_g(args, trials, plt, png_dir, now,
                               resource_png, resource_table)
    else:
        body = f"# {args.run_id}\n\n_(unrecognized stage in {args.trials})_\n"

    out_md.write_text(body, encoding="utf-8")
    print(f"[render_report] wrote {out_md}")


def _render_stage_a(args, trials, plt, png_dir, now,
                    resource_png, resource_table) -> str:
    # Group trials by which knob they scanned.
    by_knob: dict[str, list[dict]] = {}
    for t in trials:
        kind = t.get("kind", "")
        if not kind.startswith("stage-a:"):
            continue
        knob = kind.split(":", 1)[1]
        by_knob.setdefault(knob, []).append(t)

    chart_section = ""
    if plt is not None and by_knob:
        rel = _render_stage_a_chart(plt, by_knob, png_dir)
        lines = []
        for knob, rel_path in rel.items():
            lines.append(f"### {knob}\n\n![{knob}]({png_dir.name}/{rel_path})")
        chart_section = "\n\n".join(lines)

    # Per-knob table.
    table_blocks = []
    for knob, ts in by_knob.items():
        rows = [
            f"#### `{knob}`\n",
            "| value | ok | fail | attempted | fail% | rps | p50 | p95 | p99 | mean ms | wall s | aborted | SLO |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for t in ts:
            s = t["summary"]
            attempted = s.get("requests_attempted", s.get("total", 0))
            aborted = "✖ yes" if s.get("aborted") else ""
            slo = "✅" if _trial_meets_slo(t) else "❌"
            rows.append(
                f"| {t['knobs'].get(knob, '')} | {s.get('successes', 0)} | "
                f"{s.get('failures', 0)} | {attempted} | "
                f"{_trial_fail_rate(t):.0%} | "
                f"{_trial_rps(t):.3f} | {_trial_p50(t):.0f} | "
                f"{_trial_p95(t):.0f} | {_trial_p99(t):.0f} | "
                f"{_trial_mean(t):.0f} | {_trial_wall(t):.1f} | {aborted} | {slo} |"
            )
        table_blocks.append("\n".join(rows))
    tables_md = "\n\n".join(table_blocks)

    # SLO winners section — every cell that met p99 ≤ 120s AND fail ≤ 10%,
    # sorted by rps. Surfaces the candidate configs worth taking to Stage B/C.
    slo_trials = [t for trials in by_knob.values() for t in trials if _trial_meets_slo(t)]
    slo_trials.sort(key=lambda t: _trial_rps(t), reverse=True)
    if slo_trials:
        slo_rows = [
            f"SLO gate: `p99 ≤ {SLO_P99_MS/1000:.0f}s` AND `fail% ≤ {SLO_MAX_FAIL:.0%}`. "
            f"{len(slo_trials)} of {sum(len(v) for v in by_knob.values())} "
            "cells meet it. Sorted by rps descending.",
            "",
            "| knob | value | rps | p99 ms | fail% | mean ms |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for t in slo_trials:
            k = next(iter(t["knobs"]))
            slo_rows.append(
                f"| `{k}` | {t['knobs'][k]} | {_trial_rps(t):.3f} | "
                f"{_trial_p99(t):.0f} | {_trial_fail_rate(t):.0%} | "
                f"{_trial_mean(t):.0f} |"
            )
        slo_md = "\n".join(slo_rows)
    else:
        slo_md = (f"_No cell met the SLO gate (p99 ≤ {SLO_P99_MS/1000:.0f}s, "
                  f"fail ≤ {SLO_MAX_FAIL:.0%}). Either narrow the pool to "
                  "shorter documents, reduce client concurrency, or widen "
                  "the SLO._")

    resource_md = ""
    if resource_table:
        resource_md = f"\n## Resource usage (probe window)\n\n{resource_table}\n"
        if resource_png:
            resource_md += f"\n![resource usage]({png_dir.name}/{resource_png})\n"

    return f"""# GLM-OCR stage-A 1D scans — {args.run_id}

**Completed:** {now}  \\
**Source:** `{args.trials}`

One knob varies per scan; all others held at baseline. Use the charts
to spot where throughput flattens (saturation) or latency inflects
(onset of queueing).

## SLO-compliant candidates

{slo_md}

## Knob glossary

{_knob_glossary_md()}

## Charts

{chart_section or "_(matplotlib unavailable — tables below only)_"}

## Tables

{tables_md}
{resource_md}
## Observability pointers

{_observability_section(args.run_id)}
"""


def _render_stage_b(args, trials, plt, png_dir, now,
                    resource_png, resource_table) -> str:
    b_trials = [t for t in trials if t.get("kind") == "stage-b"]
    axis_keys = list(b_trials[0]["knobs"].keys()) if b_trials else []

    chart_section = ""
    if plt is not None and b_trials and len(axis_keys) == 2:
        rel = _render_stage_b_heatmaps(plt, b_trials, png_dir)
        lines = [f"### {name}\n\n![{name}]({png_dir.name}/{p})"
                 for name, p in rel.items()]
        chart_section = "\n\n".join(lines)

    rows = [
        "| " + " | ".join(axis_keys) + " | ok | fail | attempted | fail% | rps | p50 | p95 | p99 | mean | wall s | aborted |",
        "|" + "|".join(["---"] * len(axis_keys)) + "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in b_trials:
        s = t["summary"]
        vals = [str(t["knobs"].get(k, "")) for k in axis_keys]
        attempted = s.get("requests_attempted", s.get("total", 0))
        aborted = "✖ yes" if s.get("aborted") else ""
        rows.append(
            "| " + " | ".join(vals) +
            f" | {s.get('successes', 0)} | {s.get('failures', 0)} | "
            f"{attempted} | {_trial_fail_rate(t):.0%} | "
            f"{_trial_rps(t):.3f} | {_trial_p50(t):.0f} | "
            f"{_trial_p95(t):.0f} | {_trial_p99(t):.0f} | "
            f"{_trial_mean(t):.0f} | {_trial_wall(t):.1f} | {aborted} |"
        )
    table_md = "\n".join(rows)

    resource_md = ""
    if resource_table:
        resource_md = f"\n## Resource usage (probe window)\n\n{resource_table}\n"
        if resource_png:
            resource_md += f"\n![resource usage]({png_dir.name}/{resource_png})\n"

    return f"""# GLM-OCR stage-B 2D grid — {args.run_id}

**Completed:** {now}  \\
**Source:** `{args.trials}`  \\
**Axes:** {' × '.join(f'`{k}`' for k in axis_keys)}

## Knob glossary

{_knob_glossary_md()}

## Heatmaps

{chart_section or "_(matplotlib unavailable — see table below)_"}

## Grid details

{table_md}
{resource_md}
## Observability pointers

{_observability_section(args.run_id)}
"""


def _render_stage_c_bar_panels(plt, c_trials: list[dict],
                               png_dir: pathlib.Path) -> dict[str, str]:
    """Stage-G-style panels adapted for a 1-D c-sweep. One PNG per
    metric with one bar per concurrency cell, labelled with the value.
    Mirrors `_render_stage_g_heatmaps` visually but is a simple bar
    chart since only `c` varies."""
    if not c_trials:
        return {}
    cs = sorted({int(t["summary"].get("concurrency", 0) or 0) for t in c_trials})
    by_c = {
        int(t["summary"].get("concurrency", 0) or 0): t for t in c_trials
    }
    rel: dict[str, str] = {}
    for metric_name, fn, fmt, color in [
        ("rps", _trial_rps, "{:.2f}", "#2a9d8f"),
        ("p99 ms", _trial_p99, "{:.0f}", "#e76f51"),
        ("fail%", _trial_fail_rate, "{:.0%}", "#f4a261"),
    ]:
        values = [fn(by_c[c]) for c in cs]
        fig, ax = plt.subplots(figsize=(max(4, len(cs) * 1.2 + 2), 3.2))
        bars = ax.bar([str(c) for c in cs], values, color=color)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    fmt.format(v),
                    ha="center", va="bottom", fontsize=10)
        ax.set_xlabel("concurrency")
        ax.set_title(f"Stage C — {metric_name}")
        ax.margins(y=0.15)
        fig.tight_layout()
        safe = metric_name.replace(" ", "_").replace("%", "pct")
        out_png = png_dir / f"stage-c-{safe}.png"
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        rel[metric_name] = out_png.name
    return rel


def _render_stage_c(args, trials, plt, png_dir, now,
                    resource_png, resource_table) -> str:
    c_trials = [t for t in trials if t.get("kind", "").startswith("stage-c")]
    # Strip non-knob metadata before rendering "Fixed config" so the
    # cell-varying fields (concurrency, bottleneck) don't clutter it.
    fixed_raw = dict(c_trials[0]["knobs"]) if c_trials else {}
    for ephemeral in ("concurrency", "bottleneck", "endpoint"):
        fixed_raw.pop(ephemeral, None)

    # Heatmap-equivalent panels (1-D bar charts): rps / p99 / fail%.
    panels_section = ""
    if plt is not None and c_trials:
        rels = _render_stage_c_bar_panels(plt, c_trials, png_dir)
        panels_section = "\n\n".join(
            f"### {name}\n\n![{name}]({png_dir.name}/{p})"
            for name, p in rels.items()
        )

    # Legacy c-curve line chart (rps + p99 vs c on twin axes). Useful
    # alongside the bar panels when the sweep has many points.
    curve_section = ""
    if plt is not None and c_trials:
        rel = _render_stage_c_chart(plt, c_trials, png_dir)
        curve_section = f"![c-curve]({png_dir.name}/{rel})"

    # Rich grid mirroring stage-g / stage-e: utilization + bottleneck +
    # SLO verdict. Data comes from `summary.utilization` and
    # `summary.bottleneck` which stage_c attaches per cell.
    rows = [
        "| # | c | ok | fail | fail% | rps | p50 | p95 | p99 | mean ms | wall s | GPU% | VRAM% | batch% | SLO | bottleneck |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    slo_winners = []
    for i, t in enumerate(c_trials, 1):
        s = t["summary"]
        util = s.get("utilization") or {}
        c = int(s.get("concurrency", 0)) or int(t.get("knobs", {}).get("concurrency", 0))
        slo_ok = _trial_meets_slo(t)
        if slo_ok:
            slo_winners.append((i, t))
        rows.append(
            f"| {i} | {c} | {s.get('successes', 0)} | {s.get('failures', 0)} | "
            f"{_trial_fail_rate(t):.0%} | {_trial_rps(t):.3f} | "
            f"{_trial_p50(t):.0f} | {_trial_p95(t):.0f} | {_trial_p99(t):.0f} | "
            f"{_trial_mean(t):.0f} | {_trial_wall(t):.1f} | "
            f"{util.get('gpu_compute', 0) * 100:.0f} | "
            f"{util.get('gpu_memory', 0) * 100:.0f} | "
            f"{util.get('sgl_batch', 0) * 100:.0f} | "
            f"{'✅' if slo_ok else '❌'} | {s.get('bottleneck', '-')} |"
        )
    table_md = "\n".join(rows)

    # Winner: SLO-compliant cell with highest rps.
    winner_md = "_(no SLO-compliant cell)_"
    if slo_winners:
        wi, wt = max(slo_winners, key=lambda x: _trial_rps(x[1]))
        ws = wt["summary"]
        wc = int(ws.get("concurrency", 0)) or int(wt.get("knobs", {}).get("concurrency", 0))
        wu = ws.get("utilization") or {}
        winner_md = (
            f"**Cell {wi}** — `concurrency={wc}` — "
            f"rps={_trial_rps(wt):.3f}, p99={_trial_p99(wt):.0f}ms, "
            f"mean={_trial_mean(wt):.0f}ms, fail={_trial_fail_rate(wt):.0%}, "
            f"GPU%={wu.get('gpu_compute', 0) * 100:.0f}, "
            f"VRAM%={wu.get('gpu_memory', 0) * 100:.0f}."
        )

    fixed_md = "\n".join(f"- `{k}` = `{v}`" for k, v in fixed_raw.items()) or "_(baseline env)_"

    resource_md = ""
    if resource_table:
        resource_md = f"\n## Resource usage (probe window)\n\n{resource_table}\n"
        if resource_png:
            resource_md += f"\n![resource usage]({png_dir.name}/{resource_png})\n"

    return f"""# GLM-OCR stage-C c-curve — {args.run_id}

**Completed:** {now}  \\
**Source:** `{args.trials}`  \\
**Shape:** {len(c_trials)} cells  \\
**Axes:** `c`

## Winner

{winner_md}

## Fixed config

{fixed_md}

## Knob glossary

{_knob_glossary_md()}

## Panels

{panels_section or "_(matplotlib unavailable — see grid table below)_"}

## Grid

{table_md}

## Curve

{curve_section or "_(matplotlib unavailable — see table above)_"}

The classic c-curve view: throughput (rps, left axis) and tail
latency (p99, right axis) plotted against concurrency. The knee is
where p99 starts rising faster than rps — past that point, adding
concurrency only grows the queue without adding throughput.

## How to read

- **rps** is the production-throughput metric. Higher = more docs/sec per replica.
- **p99** / **mean** show latency tail vs typical. p99 ≤ 120s is the SLO.
- **batch%** is SGLang's continuous-batching fill rate = running_reqs / cap. Low batch% doesn't necessarily mean bad throughput — spec decoding compresses time each slot spends running.
- **VRAM%** near 80–95% = KV cache is at capacity. This is normal on 8 GB hardware.
- **SLO ✅** = p99 ≤ 120s AND fail ≤ 10% AND not aborted.
{resource_md}
## Observability pointers

{_observability_section(args.run_id)}
"""


# ---------------------------------------------------------------------------
# Stage E: saturation c-curve at fixed winner config with per-cell
# bottleneck + utilization attribution.
# ---------------------------------------------------------------------------

def _render_stage_e(args, trials, plt, png_dir, now,
                    resource_png, resource_table) -> str:
    e_trials = [t for t in trials if t.get("kind", "").startswith("stage-e")]
    if not e_trials:
        return f"# {args.run_id}\n\n_(no stage-e trials found)_\n"

    # Fixed config is the same across every cell — pull from the first.
    fixed = {k: v for k, v in e_trials[0]["knobs"].items()
             if k not in ("concurrency", "bottleneck", "endpoint")}
    endpoint = e_trials[0]["knobs"].get("endpoint", "/glmocr/parse")

    # c-curve chart (rps/p99 vs c) reused from stage-c rendering.
    chart_section = ""
    if plt is not None:
        rel = _render_stage_c_chart(plt, e_trials, png_dir)
        chart_section = f"![c-curve]({png_dir.name}/{rel})"

    # Grid table with utilization columns so we can eyeball where GPU
    # actually climbs and where SLO breaks.
    rows = [
        "| c | ok | fail | att | fail% | rps | p50 | p95 | p99 | mean | wall | GPU% | batch% | CPU% | VRAM% | SLO | bottleneck |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in e_trials:
        s = t["summary"]
        util = s.get("utilization") or {}
        c = int(s.get("concurrency", 0)) or int(t["knobs"].get("concurrency", 0))
        att = s.get("requests_attempted", s.get("total", 0))
        slo = "OK" if _trial_meets_slo(t) else "miss"
        rows.append(
            f"| {c} | {s.get('successes', 0)} | {s.get('failures', 0)} | {att} | "
            f"{_trial_fail_rate(t):.0%} | {_trial_rps(t):.3f} | "
            f"{_trial_p50(t):.0f} | {_trial_p95(t):.0f} | {_trial_p99(t):.0f} | "
            f"{_trial_mean(t):.0f} | {_trial_wall(t):.0f} | "
            f"{util.get('gpu_compute', 0) * 100:.0f} | "
            f"{util.get('sgl_batch', 0) * 100:.0f} | "
            f"{util.get('cpu_container', 0) * 100:.0f} | "
            f"{util.get('gpu_memory', 0) * 100:.0f} | "
            f"{slo} | {s.get('bottleneck', '-')} |"
        )
    table_md = "\n".join(rows)

    # Find the saturation knee: highest c where SLO still holds.
    winners = [t for t in e_trials if _trial_meets_slo(t)]
    if winners:
        best = max(winners, key=lambda t: _trial_rps(t))
        c_best = int(best["summary"].get("concurrency", 0)) or int(best["knobs"].get("concurrency", 0))
        gpu_best = (best["summary"].get("utilization") or {}).get("gpu_compute", 0)
        winner_md = (
            f"- **Best c under SLO** = **{c_best}** — "
            f"rps={_trial_rps(best):.3f}, p99={_trial_p99(best):.0f}ms, "
            f"GPU util={gpu_best * 100:.0f}%, "
            f"bottleneck={best['summary'].get('bottleneck', '-')}"
        )
    else:
        winner_md = "- _No cell met the SLO gate — every c-level violated p99 or fail threshold._"

    # Fixed config as a bulleted list.
    fixed_md = "\n".join(f"- `{k}` = `{v}`" for k, v in fixed.items())

    resource_md = ""
    if resource_table:
        resource_md = f"\n## Resource usage (global probe window)\n\n{resource_table}\n"
        if resource_png:
            resource_md += f"\n![resource usage]({png_dir.name}/{resource_png})\n"

    return f"""# GLM-OCR stage-E saturation c-curve - {args.run_id}

**Completed:** {now}  \\
**Endpoint:** `{endpoint}`  \\
**Source:** `{args.trials}`

## Fixed config

{fixed_md}

## SLO-compliant winner

{winner_md}

## Knob glossary

{_knob_glossary_md()}

## Curve

{chart_section or "_(matplotlib unavailable - see table below)_"}

## Grid + utilization

{table_md}

## How to read this

- **GPU%** climbs toward saturation as we raise c. The knee of the curve
  is where we want production to operate (high util, still under SLO).
- **batch%** is running requests / SGL cap. It's what "continuous
  batching fill rate" looks like. Higher c forces more concurrent
  region calls → higher batch%.
- **SLO OK** means p99 <= 120s AND fail <= 10% AND not aborted.
- Compare runs with endpoint=`/glmocr/parse` vs `/glmocr/parse-async`
  at the same c to measure the pipeline-overlap uplift.
{resource_md}
## Observability pointers

{_observability_section(args.run_id)}
"""


# ---------------------------------------------------------------------------
# Stage D: focused combo grid with bottleneck attribution.
# ---------------------------------------------------------------------------

# Axes swept in Stage D. Keep in sync with scripts/tune_params.py
# STAGE_D_COMBOS — if the combo structure changes there, update here.
STAGE_D_AXES = ["CPU_WORKERS", "CPU_THREADS", "OCR_MAX_WORKERS",
                "SGL_MAX_RUNNING_REQUESTS"]


def _peak_util(t: dict) -> tuple[str, float]:
    """Return (resource_name, util_ratio) for the most-saturated resource
    in this cell. Used to surface where head-room actually is."""
    util = t.get("summary", {}).get("utilization") or {}
    if not util:
        return ("?", 0.0)
    name, val = max(util.items(), key=lambda kv: kv[1])
    return name, float(val)


def _render_stage_d(args, trials, plt, png_dir, now,
                    resource_png, resource_table) -> str:
    d_trials = [t for t in trials if t.get("kind") == "stage-d"]

    # ---- Bottleneck tally -------------------------------------------------
    from collections import Counter
    tally = Counter(t["summary"].get("bottleneck", "unknown") for t in d_trials)
    tally_lines = ["| Bottleneck | Cells |", "|---|---:|"]
    for label, n in tally.most_common():
        tally_lines.append(f"| {label} | {n} |")
    tally_md = "\n".join(tally_lines)

    # ---- Main grid table --------------------------------------------------
    headers = (
        ["# "] + [k.replace("_", " ") for k in STAGE_D_AXES] +
        ["ok", "fail", "att", "fail%", "rps", "p50", "p95", "p99", "mean", "wall",
         "GPU%", "VRAM%", "batch%", "CPU%", "abort?", "SLO", "bottleneck"]
    )
    sep = ["---" for _ in headers[:5]] + ["---:" for _ in headers[5:]]
    rows = ["| " + " | ".join(headers) + " |",
            "|" + "|".join(sep) + "|"]

    for i, t in enumerate(d_trials, start=1):
        s = t["summary"]
        knobs = t.get("knobs", {})
        util = s.get("utilization") or {}
        attempted = s.get("requests_attempted", s.get("total", 0))
        aborted = "✖" if s.get("aborted") else ""
        slo = "✅" if _trial_meets_slo(t) else "❌"
        cells = (
            [str(i)] +
            [str(knobs.get(k, "")) for k in STAGE_D_AXES] +
            [str(s.get("successes", 0)), str(s.get("failures", 0)), str(attempted),
             f"{_trial_fail_rate(t):.0%}", f"{_trial_rps(t):.3f}",
             f"{_trial_p50(t):.0f}", f"{_trial_p95(t):.0f}",
             f"{_trial_p99(t):.0f}", f"{_trial_mean(t):.0f}",
             f"{_trial_wall(t):.0f}",
             f"{util.get('gpu_compute', 0) * 100:.0f}",
             f"{util.get('gpu_memory', 0) * 100:.0f}",
             f"{util.get('sgl_batch', 0) * 100:.0f}",
             f"{util.get('cpu_container', 0) * 100:.0f}",
             aborted, slo, s.get("bottleneck", "—")]
        )
        rows.append("| " + " | ".join(cells) + " |")
    grid_md = "\n".join(rows)

    # ---- Utilization PNG (one row per cell) -------------------------------
    chart_section = ""
    if plt is not None and d_trials:
        chart_section = _render_stage_d_util_chart(plt, d_trials, png_dir)

    # ---- SLO winners ------------------------------------------------------
    slo_winners = sorted(
        [t for t in d_trials if _trial_meets_slo(t)],
        key=lambda t: _trial_rps(t), reverse=True,
    )
    if slo_winners:
        def _fmt_slo_winner(t: dict) -> str:
            idx1 = d_trials.index(t) + 1
            knobs = t.get("knobs", {})
            knob_str = ", ".join(f"{k}={knobs.get(k, '')}" for k in STAGE_D_AXES)
            bneck = t["summary"].get("bottleneck", "")
            return (f"- **cell {idx1}** (`{knob_str}`) — "
                    f"rps={_trial_rps(t):.3f}, p99={_trial_p99(t):.0f}ms, "
                    f"fail={_trial_fail_rate(t):.0%}, bottleneck={bneck}")
        slo_md = "\n".join(_fmt_slo_winner(t) for t in slo_winners)
    else:
        slo_md = (f"_No cell met the SLO gate (`p99 ≤ {SLO_P99_MS/1000:.0f}s`, "
                  f"`fail ≤ {SLO_MAX_FAIL:.0%}`, not aborted). The `bottleneck` "
                  f"column below tells you what's pinned._")

    resource_md = ""
    if resource_table:
        resource_md = f"\n## Resource usage (global probe window)\n\n{resource_table}\n"
        if resource_png:
            resource_md += f"\n![resource usage]({png_dir.name}/{resource_png})\n"

    return f"""# GLM-OCR stage-D combo grid — {args.run_id}

**Completed:** {now}  \\
**Source:** `{args.trials}`  \\
**Shape:** {len(d_trials)} cells, c=8, N=200 per cell, pool_seed=42

## SLO-compliant candidates

{slo_md}

## Bottleneck tally

_Counts how many of the {len(d_trials)} cells hit each classification.
If every cell is GPU-bound the CPU axes don't matter; if ⚠ UNDER-UTILIZED
appears, that cell could handle more load._

{tally_md}

## Utilization (per-cell resource headroom)

_For each cell the four ratios below (also in the grid table) are
`avg_usage / capacity`. A row where all four are well below 1.0 means
the stack is under-utilized — nothing is pinned._

{chart_section or "_(matplotlib unavailable — see GPU%/VRAM%/batch%/CPU% columns in the grid below)_"}

## Grid details

{grid_md}

## How to read this

- **GPU%** = avg DCGM gpu_util ÷ 100
- **VRAM%** = avg FB_USED ÷ 16 GB (T4)
- **batch%** = avg sglang:num_running_reqs ÷ SGL_MAX_RUNNING_REQUESTS
- **CPU%** = avg container CPU cores ÷ (CPU_WORKERS × CPU_THREADS)
- A cell is **SLO ✅** only if p99 ≤ 120s AND fail ≤ 10% AND it wasn't
  aborted.
- The **bottleneck** column applies the priority rules documented in
  the v4 plan: VRAM → GPU compute (batch-full vs under-subscribed) →
  CPU container → GPU queue → CPU ingress → under-utilization.
{resource_md}
## Observability pointers

{_observability_section(args.run_id)}
"""


def _render_stage_d_util_chart(plt, d_trials: list[dict], png_dir: pathlib.Path) -> str:
    """Horizontal bars, one row per cell, four grouped bars per row:
    GPU, VRAM, batch, CPU. Red dashed line at 0.85 = saturation."""
    labels = [f"cell {i+1}" for i in range(len(d_trials))]
    util_keys = ["gpu_compute", "gpu_memory", "sgl_batch", "cpu_container"]
    util_names = ["GPU", "VRAM", "batch", "CPU ctr"]
    colors = ["#d62728", "#ff7f0e", "#9467bd", "#1f77b4"]

    import numpy as np
    n = len(d_trials)
    fig, ax = plt.subplots(figsize=(10, max(3, 0.7 * n + 1)))
    bar_height = 0.18
    y_pos = np.arange(n)

    for k_idx, (key, name, color) in enumerate(zip(util_keys, util_names, colors)):
        values = [float(((t.get("summary", {}).get("utilization") or {}).get(key) or 0.0))
                  for t in d_trials]
        offset = (k_idx - 1.5) * bar_height
        ax.barh(y_pos + offset, values, height=bar_height,
                label=name, color=color, alpha=0.9)

    ax.axvline(0.85, color="red", linestyle="--", linewidth=1,
               label="saturation (0.85)")
    ax.set_xlim(0, max(1.05, max([(t.get("summary", {}).get("utilization") or {}).get(k, 0)
                                   for t in d_trials for k in util_keys] + [1.0]) + 0.05))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # cell 1 on top
    ax.set_xlabel("Utilization (fraction of capacity)")
    ax.set_title("Stage D — per-cell resource utilization")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_png = png_dir / "stage-d-utilization.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return f"![utilization]({png_dir.name}/{out_png.name})"


# ---------------------------------------------------------------------------
# Stage G: OCR_MAX_WORKERS × SGL_MAX_TOTAL_TOKENS 2D grid at fixed c.
# ---------------------------------------------------------------------------

def _render_stage_g(args, trials, plt, png_dir, now,
                    resource_png, resource_table) -> str:
    g_trials = [t for t in trials if t.get("kind") == "stage-g"]
    if not g_trials:
        return f"# {args.run_id}\n\n_(no stage-g trials)_\n"

    # Detect which axes actually vary. Legacy sweeps varied OMW × MAX_TOKENS
    # at fixed c; the reshape varies MAX_TOKENS × concurrency at fixed OMW.
    # Pick whichever of {OMW, concurrency} has more than one unique value.
    omw_values = {int(t["summary"].get("omw", 0)) for t in g_trials}
    c_values   = {int(t["summary"].get("concurrency", 0) or 0) for t in g_trials}
    inner_is_c = len(c_values) > 1

    inner_label = "c" if inner_is_c else "OMW"
    inner_key   = "concurrency" if inner_is_c else "omw"

    rows = [
        f"| # | {inner_label} | MAX_TOKENS | rps | fail% | p50 | p95 | p99 | mean ms | GPU% | VRAM% | batch% | SLO | bottleneck |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    slo_winners = []
    for i, t in enumerate(g_trials, 1):
        s = t["summary"]
        util = s.get("utilization") or {}
        slo_ok = _trial_meets_slo(t)
        if slo_ok:
            slo_winners.append((i, t))
        rows.append(
            f"| {i} | {s.get(inner_key, '?')} | {s.get('max_total_tokens', '?')} | "
            f"{_trial_rps(t):.3f} | {_trial_fail_rate(t):.0%} | "
            f"{_trial_p50(t):.0f} | {_trial_p95(t):.0f} | {_trial_p99(t):.0f} | "
            f"{_trial_mean(t):.0f} | "
            f"{util.get('gpu_compute', 0)*100:.0f} | "
            f"{util.get('gpu_memory', 0)*100:.0f} | "
            f"{util.get('sgl_batch', 0)*100:.0f} | "
            f"{'✅' if slo_ok else '❌'} | {s.get('bottleneck', '-')} |"
        )
    grid_md = "\n".join(rows)

    # Heatmaps: one for rps, one for p99. Axes auto-detected above.
    chart_section = ""
    if plt is not None and slo_winners:
        rel = _render_stage_g_heatmaps(plt, g_trials, png_dir, inner_key, inner_label)
        chart_section = "\n\n".join(
            f"### {name}\n\n![{name}]({png_dir.name}/{p})"
            for name, p in rel.items()
        )

    # Winner.
    winner_md = "_(no SLO-compliant cell)_"
    if slo_winners:
        wi, wt = max(slo_winners, key=lambda x: _trial_rps(x[1]))
        s = wt["summary"]
        inner_desc = (
            f"concurrency={s.get('concurrency')}"
            if inner_is_c else f"OCR_MAX_WORKERS={s.get('omw')}"
        )
        winner_md = (
            f"**Cell {wi}** — `{inner_desc}`, "
            f"`SGL_MAX_TOTAL_TOKENS={s.get('max_total_tokens')}` — "
            f"rps={_trial_rps(wt):.3f}, p99={_trial_p99(wt):.0f}ms, "
            f"mean={_trial_mean(wt):.0f}ms, fail={_trial_fail_rate(wt):.0%}, "
            f"GPU%={(wt['summary'].get('utilization') or {}).get('gpu_compute', 0)*100:.0f}."
        )

    fixed_md = (
        "- `CPU_WORKERS=4`, `CPU_THREADS=8`\n"
        "- `CPU container = 8 vCPU / 20 GB limit`\n"
        "- `SGL_MAX_RUNNING_REQUESTS=32`\n"
        "- `SGL_MEM_FRACTION_STATIC=0.814`\n"
        "- `SGL_SPECULATIVE=true` (NEXTN, 4 draft tokens, topk=1)\n"
        "- `OCR_CONN_POOL=192`\n"
        "- `OCR_REQUEST_TIMEOUT=60`, `OCR_RETRY_MAX=1`\n"
        "- Pool: 128 images from `OmniDocBench/images/` only (deterministic, alphabetical)\n"
        + (
            f"- `OCR_MAX_WORKERS={next(iter(omw_values))}` (fixed)\n"
            if inner_is_c else ""
        )
        + "- `N=100` per cell, abort gate 15%"
    )

    resource_md = ""
    if resource_table:
        resource_md = f"\n## Resource usage (global probe window)\n\n{resource_table}\n"
        if resource_png:
            resource_md += f"\n![resource usage]({png_dir.name}/{resource_png})\n"

    return f"""# GLM-OCR stage-G sweep — {args.run_id}

**Completed:** {now}  \\
**Source:** `{args.trials}`  \\
**Shape:** {len(g_trials)} cells  \\
**Axes:** `{inner_label}` × `SGL_MAX_TOTAL_TOKENS`

## Winner

{winner_md}

## Fixed config

{fixed_md}

## Knob glossary

{_knob_glossary_md()}

## Heatmaps

{chart_section or "_(matplotlib unavailable — see grid table below)_"}

## Grid

{grid_md}

## How to read

- **rps** is the production-throughput metric. Higher = more docs/sec per replica.
- **p99** / **mean** show latency tail vs typical. p99 ≤ 120s is the SLO.
- **batch%** is SGLang's continuous-batching fill rate = running_reqs / cap. Low batch% doesn't necessarily mean bad throughput — spec decoding compresses time each slot spends running.
- **VRAM%** near 80–95% = KV cache is at capacity. This is normal on 8 GB hardware.
- **SLO ✅** = p99 ≤ 120s AND fail ≤ 10% AND not aborted.
{resource_md}
## Observability pointers

{_observability_section(args.run_id)}
"""


def _render_stage_g_heatmaps(plt, g_trials: list[dict], png_dir: pathlib.Path,
                             inner_key: str = "omw",
                             inner_label: str = "OCR_MAX_WORKERS") -> dict[str, str]:
    """Two heatmaps: rps (higher better) and p99 (lower better). X axis is
    the varying inner axis (OMW or concurrency); Y axis is MAX_TOKENS."""
    if not g_trials:
        return {}
    inners = sorted({int(t["summary"].get(inner_key, 0) or 0) for t in g_trials})
    mts    = sorted({int(t["summary"].get("max_total_tokens", 0)) for t in g_trials})
    idx = {
        (int(t["summary"].get(inner_key, 0) or 0),
         int(t["summary"].get("max_total_tokens", 0))): t
        for t in g_trials
    }
    rel_paths: dict[str, str] = {}

    for metric_name, fn, fmt, cmap in [
        ("rps", _trial_rps, "{:.2f}", "viridis"),
        ("p99 ms", _trial_p99, "{:.0f}", "Reds_r"),
        ("fail%", _trial_fail_rate, "{:.0%}", "Reds"),
    ]:
        values = [[fn(idx[(o, m)]) if (o, m) in idx else float("nan")
                   for o in inners] for m in mts]
        fig, ax = plt.subplots(figsize=(max(4, len(inners) + 2), max(3, len(mts) + 1)))
        im = ax.imshow(values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(inners)))
        ax.set_xticklabels(inners)
        ax.set_yticks(range(len(mts)))
        ax.set_yticklabels([f"{m:,}" for m in mts])
        ax.set_xlabel(inner_label)
        ax.set_ylabel("SGL_MAX_TOTAL_TOKENS")
        ax.set_title(f"Stage G — {metric_name}")
        for yi in range(len(mts)):
            for xi in range(len(inners)):
                v = values[yi][xi]
                if v == v:
                    ax.text(xi, yi, fmt.format(v), ha="center", va="center",
                            color="white", fontsize=9)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        safe = metric_name.replace(" ", "_").replace("%", "pct")
        out_png = png_dir / f"stage-g-{safe}.png"
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        rel_paths[metric_name] = out_png.name
    return rel_paths


# ---------------------------------------------------------------------------
# Stage CPU: CPU-container-focused single-cell report. Request phase
# decomposition + container resource time-series + in-flight concurrency.
# ---------------------------------------------------------------------------


def _load_histograms_diff(pre_path: pathlib.Path | None,
                          post_path: pathlib.Path) -> dict:
    """Parse `/metrics` snapshots and return the post-pre diff. If pre is
    missing/unreadable, the post is returned as-is (caller is responsible
    for the force-recreate-before-run invariant)."""
    post_samples = pm.parse_prom_text(post_path.read_text(encoding="utf-8"))
    post = pm.collect_histograms(post_samples)
    if pre_path is None:
        return post
    try:
        pre_samples = pm.parse_prom_text(pre_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return post
    pre = pm.collect_histograms(pre_samples)
    return pm.diff_histograms(post, pre)


def _fmt_q(val: float | None, overflow_label: str | None) -> str:
    """Render a histogram quantile result as `1.23s` or `">20s"`."""
    if overflow_label is not None:
        return overflow_label
    if val is None:
        return "—"
    return f"{val:.2f}s"


def _fmt_max(val: float | None, overflow_label: str | None) -> str:
    """Render histogram max as `≤1.23s` (bucket upper bound) or `">20s"`.

    The `≤` prefix signals that the true max is ≤ this bucket edge, not
    an exact value — Prometheus histograms don't expose exact extrema.
    """
    if overflow_label is not None:
        return overflow_label
    if val is None:
        return "—"
    return f"≤{val:.2f}s"


def _in_flight_stats(probe_path: pathlib.Path | None) -> dict[str, float | None]:
    """Reduce `in_flight` samples from probe JSONL. Returns mean/peak/p95."""
    if probe_path is None:
        return {"mean": None, "peak": None, "p95": None, "samples": 0}
    try:
        lines = probe_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {"mean": None, "peak": None, "p95": None, "samples": 0}
    vals: list[float] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            s = json.loads(line)
        except ValueError:
            continue
        v = s.get("in_flight")
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return {"mean": None, "peak": None, "p95": None, "samples": 0}
    vals_sorted = sorted(vals)
    idx_p95 = max(0, min(len(vals_sorted) - 1, int(0.95 * (len(vals_sorted) - 1))))
    return {
        "mean": statistics.fmean(vals),
        "peak": max(vals),
        "p95": vals_sorted[idx_p95],
        "samples": len(vals),
    }


def _render_phase_bar_chart(plt, decomp: dict, png_dir: pathlib.Path) -> str | None:
    """Horizontal stacked-bar: layout / ocr fan-out wall / other,
    with ideal-parallel floor as a dashed reference."""
    if plt is None:
        return None
    segments = [
        ("layout (CPU)",   decomp.get("layout", 0.0), "#4c72b0"),
        ("ocr fan-out",    decomp.get("ocr_wall", 0.0), "#dd8452"),
        ("other overhead", decomp.get("other", 0.0), "#55a868"),
    ]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    left = 0.0
    for label, width, color in segments:
        if width <= 0:
            continue
        ax.barh(0, width, left=left, color=color, edgecolor="white",
                linewidth=1.5, label=f"{label}: {width:.2f}s")
        ax.text(left + width / 2, 0, f"{width:.2f}s", ha="center",
                va="center", color="white", fontsize=9, fontweight="bold")
        left += width
    # Ideal-parallel floor — the OCR slice if fan-out were perfectly parallel.
    floor = decomp.get("ideal_floor")
    if floor is not None and floor > 0:
        ax.axvline(decomp.get("layout", 0.0) + floor, linestyle="--",
                   color="#c44e52", linewidth=1.5,
                   label=f"ideal-parallel floor: layout + {floor:.2f}s")
    ax.set_yticks([])
    ax.set_xlabel("seconds (mean per /glmocr/parse request)")
    ax.set_title("Mean request phase decomposition")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out = png_dir / "phase-decomposition.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out.name


def _render_stage_cpu(args, trials, plt, png_dir, now,
                      resource_png, resource_table) -> str:
    # Pick the single stage-e trial (c=12 cell). If multiple are present
    # we take the first — this renderer is intended for one cell.
    if not trials:
        return f"# {args.run_id}\n\n_(no trials — cannot render)_\n"
    t = trials[0]
    s = t.get("summary", {})
    knobs = t.get("knobs", {})

    # ---- CPU container /metrics histograms (post-pre diff) ---------------
    hists = _load_histograms_diff(
        pathlib.Path(args.metrics_pre) if args.metrics_pre else None,
        pathlib.Path(args.metrics_post),
    )

    layout_h = pm.find_histogram(hists, "glmocr_layout_seconds")
    ocr_h = pm.find_histogram(hists, "glmocr_ocr_region_seconds")
    # Flask histogram is grouped by url_rule; scope to /glmocr/parse.
    parse_h = pm.find_histogram(
        hists, "flask_http_request_duration_seconds",
        match_labels={"url_rule": "/glmocr/parse"},
    )

    # ---- SGLang /metrics histograms (optional) ---------------------------
    sgl_hists = {}
    if getattr(args, "sglang_metrics_post", None):
        sgl_hists = _load_histograms_diff(
            pathlib.Path(args.sglang_metrics_pre) if args.sglang_metrics_pre else None,
            pathlib.Path(args.sglang_metrics_post),
        )

    # Metric names use the colon prefix `sglang:...` on the server we run.
    sgl_e2e = pm.find_histogram(sgl_hists, "sglang:e2e_request_latency_seconds")
    sgl_queue = pm.find_histogram(sgl_hists, "sglang:queue_time_seconds")
    sgl_ttft = pm.find_histogram(sgl_hists, "sglang:time_to_first_token_seconds")
    sgl_itl = pm.find_histogram(sgl_hists, "sglang:inter_token_latency_seconds")

    def hrow(label: str, container: str, h: pm.Histogram | None) -> str:
        if h is None or h.count == 0:
            return f"| {label} | {container} | — | — | — | — | — | 0 |"
        p50_v, p50_o = h.quantile(0.50)
        p95_v, p95_o = h.quantile(0.95)
        p99_v, p99_o = h.quantile(0.99)
        max_v, max_o = h.max_approx()
        mean_s = h.mean()
        return (
            f"| {label} | {container} | "
            f"{_fmt_q(p50_v, p50_o)} | "
            f"{_fmt_q(p95_v, p95_o)} | "
            f"{_fmt_q(p99_v, p99_o)} | "
            f"{(mean_s if mean_s is not None else 0.0):.2f}s | "
            f"{_fmt_max(max_v, max_o)} | "
            f"{int(h.count)} |"
        )

    # ---- End-to-end comparison: client-observed vs server-observed -------
    # Client-observed = what the driver measured (bench.py). Server-observed
    # = flask_http_request_duration_seconds scoped to /glmocr/parse. If they
    # diverge significantly, there's network / async dispatch overhead on
    # the driver side.
    def _ms_to_s(v: float) -> float:
        return v / 1000.0

    client_row = (
        f"| Client-observed (bench.py) | driver | "
        f"{_ms_to_s(_trial_p50(t)):.2f}s | "
        f"{_ms_to_s(_trial_p95(t)):.2f}s | "
        f"{_ms_to_s(_trial_p99(t)):.2f}s | "
        f"{_ms_to_s(_trial_mean(t)):.2f}s | "
        f"{_ms_to_s(_trial_max(t)):.2f}s | "
        f"{int(s.get('requests_attempted') or s.get('total') or 0)} |"
    )

    e2e_table = "\n".join([
        "| Signal | container | p50 | p95 | p99 | mean | max | samples |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        client_row,
        hrow("Server-observed /glmocr/parse", "cpu", parse_h),
    ])

    # ---- Pipeline phase breakdown ----------------------------------------
    # Every timed phase, tagged with which container it runs in. OCR
    # region = one SGLang call, so the phase is measured on the cpu
    # container (aiohttp round-trip including SGLang server processing)
    # but the heavy lifting happens on the gpu container.
    phase_table = "\n".join([
        "| Phase | container | p50 | p95 | p99 | mean | max | samples |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        hrow("Layout inference (PP-DocLayoutV3)", "cpu", layout_h),
        hrow("OCR region RTT (client side)", "cpu→gpu", ocr_h),
        hrow("SGLang server E2E (per call)", "gpu", sgl_e2e),
        hrow("SGLang scheduler queue time", "gpu", sgl_queue),
        hrow("SGLang time-to-first-token", "gpu", sgl_ttft),
        hrow("SGLang inter-token latency", "gpu", sgl_itl),
    ])

    # ---- Decomposition (mean values) -------------------------------------
    mean_total = parse_h.mean() if parse_h and parse_h.count else None
    mean_layout = layout_h.mean() if layout_h and layout_h.count else None
    n_regions = int(ocr_h.count) if ocr_h else 0
    n_requests = int(parse_h.count) if parse_h else 0
    # Sum of per-region OCR time, divided by OCR_MAX_WORKERS, is the ideal
    # parallel floor for one request's OCR phase if all regions fan out.
    omw = _safe_int(knobs.get("OCR_MAX_WORKERS")) or 16
    ideal_floor = None
    if ocr_h and n_requests > 0 and omw > 0:
        ideal_floor = (ocr_h.sum / n_requests) / omw

    decomp: dict[str, float | None] = {}
    if mean_total is not None and mean_layout is not None:
        # Empirical ocr wall = mean_total − mean_layout − tiny fixed overhead.
        # We don't have "other" directly; we assume any residual is
        # preprocessing + serialization. Clamp to >= 0.
        ocr_wall = max(0.0, mean_total - mean_layout)
        # If the "ideal floor" is smaller than ocr_wall, the rest is
        # "other"; otherwise fan-out is already at the floor and other=0.
        if ideal_floor is not None and ideal_floor < ocr_wall:
            other = ocr_wall - ideal_floor
            ocr_wall_bar = ideal_floor
        else:
            other = 0.0
            ocr_wall_bar = ocr_wall
        decomp = {
            "layout": mean_layout,
            "ocr_wall": ocr_wall_bar,
            "other": other,
            "ideal_floor": ideal_floor,
        }

    phase_png = _render_phase_bar_chart(plt, decomp, png_dir) if decomp else None
    phase_md = (
        f"![phase decomposition]({png_dir.name}/{phase_png})\n"
        if phase_png else "_(no phase decomposition — metrics unavailable)_\n"
    )

    # ---- Worker concurrency ---------------------------------------------
    if_stats = _in_flight_stats(pathlib.Path(args.probe) if args.probe else None)

    def _fmt_f(v: float | None) -> str:
        return f"{v:.1f}" if v is not None else "—"

    in_flight_md = (
        "| Metric | Value |\n"
        "|---|---:|\n"
        f"| target concurrency (c) | {knobs.get('concurrency', '?')} |\n"
        f"| in-flight mean | {_fmt_f(if_stats['mean'])} |\n"
        f"| in-flight p95  | {_fmt_f(if_stats['p95'])} |\n"
        f"| in-flight peak | {_fmt_f(if_stats['peak'])} |\n"
        f"| probe samples  | {if_stats['samples']} |"
    )

    # ---- Summary header --------------------------------------------------
    slo_ok = _trial_meets_slo(t)
    pname, pval = _peak_util(t)
    summary_line = (
        f"**c={knobs.get('concurrency', '?')}**, "
        f"**N={int(s.get('total') or s.get('requests_attempted') or 0)}** — "
        f"rps={_trial_rps(t):.3f}, "
        f"p50={_trial_p50(t):.0f}ms, "
        f"p95={_trial_p95(t):.0f}ms, "
        f"p99={_trial_p99(t):.0f}ms, "
        f"fail={_trial_fail_rate(t):.0%}, "
        f"SLO={'✅' if slo_ok else '❌'}. "
        f"Most-saturated resource during cell: **{pname}** ({pval*100:.0f}%)."
    )

    # ---- Derived fraction (for prose) -----------------------------------
    frac_line = ""
    if mean_total and mean_layout:
        pct = (mean_layout / mean_total) * 100.0
        frac_line = (
            f"\nLayout consumed **{pct:.0f}%** of mean request wall time "
            f"({mean_layout:.2f}s of {mean_total:.2f}s)."
        )

    resource_png_md = ""
    if resource_png:
        resource_png_md = f"\n![resource usage]({png_dir.name}/{resource_png})\n"

    regions_per_req = (n_regions / n_requests) if n_requests else 0.0

    # SGLang gauge summary from probe JSONL (running / queued averaged).
    sgl_gauge_md = _sglang_gauge_summary(pathlib.Path(args.probe) if args.probe else None)

    # SGLang-side note about how to read the table.
    sgl_note = (
        "" if sgl_e2e else
        "\n_(SGLang histograms absent — pass `--sglang-metrics-pre/--sglang-metrics-post` "
        "to populate the gpu-side rows.)_\n"
    )

    return f"""# GLM-OCR system-wide latency — {args.run_id}

**Completed:** {now}  \\
**Source:** `{args.trials}` (+ probe + cpu /metrics + sglang /metrics snapshots)

## Summary

{summary_line}{frac_line}

## End-to-end latency (client-observed vs server-observed)

Two measurements of the same requests:
- **Client-observed** — bench.py wall-time from request fire to response received.
- **Server-observed** — Flask request-duration histogram inside the cpu container.

A large gap between them = asyncio dispatch / network overhead on the driver.

{e2e_table}

## Pipeline phase breakdown (where each request spends its time)

All phases that are individually timed. Each row is tagged with which
container owns the work. "cpu→gpu" means the measurement is recorded on
the cpu container but the wall time includes SGLang server processing.
{sgl_note}
{phase_table}

- **Total /glmocr/parse requests:** {n_requests}  (warmup excluded via pre-snapshot after warmup)
- **OCR region calls:** {n_regions}  (≈ {regions_per_req:.1f} regions / request)

## Mean request phase decomposition (visual)

{phase_md}

## Worker concurrency (CPU container)

{in_flight_md}

## SGLang state (from probe, live gauges)

{sgl_gauge_md}

## Container resource usage (during probe)

Both containers are measured from cAdvisor (`container_cpu_usage_seconds_total`
rate, `container_memory_rss`); GPU is measured from DCGM. See
`{png_dir.name}/resource-usage.png` for the time-series view.

{resource_table or "_(no probe file)_"}
{resource_png_md}
## Knob glossary

{_knob_glossary_md()}

## Observability pointers

{_observability_section(args.run_id)}
"""


def _safe_int(x: Any) -> int | None:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _sglang_gauge_summary(probe_path: pathlib.Path | None) -> str:
    """Summarize sglang_running / sglang_queued from probe samples.

    runtime_probe_loop samples these every 2s from SGLang /metrics.
    This table shows how full SGLang's scheduler was during the cell
    — the headline answer to 'did we saturate the gpu server?'.
    """
    if probe_path is None:
        return "_(no probe file)_"
    try:
        lines = probe_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return "_(no probe file)_"
    running: list[float] = []
    queued: list[float] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        r = d.get("sglang_running")
        q = d.get("sglang_queued")
        if isinstance(r, (int, float)):
            running.append(float(r))
        if isinstance(q, (int, float)):
            queued.append(float(q))

    def _stat(vals: list[float]) -> tuple[str, str, str]:
        if not vals:
            return ("—", "—", "—")
        vs = sorted(vals)
        idx_p95 = max(0, min(len(vs) - 1, int(0.95 * (len(vs) - 1))))
        return (
            f"{statistics.fmean(vals):.1f}",
            f"{vs[idx_p95]:.1f}",
            f"{max(vals):.0f}",
        )

    r_mean, r_p95, r_peak = _stat(running)
    q_mean, q_p95, q_peak = _stat(queued)
    return (
        "| Signal | mean | p95 | peak |\n"
        "|---|---:|---:|---:|\n"
        f"| sglang running (in-GPU batch) | {r_mean} | {r_p95} | {r_peak} |\n"
        f"| sglang queued (waiting for slot) | {q_mean} | {q_p95} | {q_peak} |"
    )


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def render_stage_cpu(args: argparse.Namespace) -> None:
    trials = json.loads(pathlib.Path(args.trials).read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    out_md = pathlib.Path(args.out)
    png_dir = out_md.parent / f"{out_md.stem}.d"
    png_dir.mkdir(parents=True, exist_ok=True)
    plt = _try_import_matplotlib()

    resource_png = None
    resource_table = ""
    if args.probe:
        probe_path = pathlib.Path(args.probe)
        resource_table = _resource_summary_table(probe_path)
        if plt is not None:
            resource_png = _render_resource_chart(plt, probe_path, png_dir)

    body = _render_stage_cpu(args, trials, plt, png_dir, now,
                             resource_png, resource_table)
    out_md.write_text(body, encoding="utf-8")
    print(f"[render_report] wrote {out_md}")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    common_cpu = dict(default="http://localhost:5002")

    ps = sub.add_parser("simple")
    ps.add_argument("--bench", required=True)
    ps.add_argument("--out", required=True)
    ps.add_argument("--run-id", required=True)
    ps.add_argument("--pool-size", default="?")
    ps.add_argument("--cpu-url", **common_cpu)

    pw = sub.add_parser("sweep")
    pw.add_argument("--bench", required=True, nargs="+")
    pw.add_argument("--out", required=True)
    pw.add_argument("--run-id", required=True)
    pw.add_argument("--pool-size", default="?")
    pw.add_argument("--cpu-url", **common_cpu)

    pp = sub.add_parser("probe")
    pp.add_argument("--bench", required=True)
    pp.add_argument("--probe", required=True)
    pp.add_argument("--out", required=True)
    pp.add_argument("--run-id", required=True)
    pp.add_argument("--pool-size", default="?")
    pp.add_argument("--cpu-url", **common_cpu)
    pp.add_argument("--prom-url", default="http://localhost:9090")

    pt = sub.add_parser("stage")
    pt.add_argument("--trials", required=True,
                    help="loadtest/results/raw/<run-id>/_trials.json from "
                         "tune_params.py --stage")
    pt.add_argument("--out", required=True)
    pt.add_argument("--run-id", required=True)
    pt.add_argument("--probe", default=None,
                    help="optional probe.jsonl to include resource-usage "
                         "summary + chart")
    pt.add_argument("--stage", choices=["a", "b", "c"], default=None,
                    help="override stage inferred from trial kinds")

    # CPU-focus single-cell report. Different from `stage` in that it
    # consumes pre/post /metrics snapshots and produces a phase-breakdown
    # PNG instead of throughput-vs-knob charts.
    pc = sub.add_parser("stage-cpu")
    pc.add_argument("--trials", required=True,
                    help="loadtest/results/raw/<run-id>/_trials.json")
    pc.add_argument("--probe", required=True,
                    help="runtime_probe_loop JSONL for resource time-series")
    pc.add_argument("--metrics-pre", default=None,
                    help="/metrics snapshot BEFORE the run (pairs with "
                         "--metrics-post for histogram diff). Optional if "
                         "the cpu container was force-recreated immediately "
                         "before the run.")
    pc.add_argument("--metrics-post", required=True,
                    help="/metrics snapshot AFTER the run")
    pc.add_argument("--sglang-metrics-pre", default=None,
                    help="SGLang /metrics snapshot BEFORE the run — unlocks "
                         "gpu-side percentile rows (sglang:e2e_*, "
                         "sglang:queue_time, sglang:time_to_first_token).")
    pc.add_argument("--sglang-metrics-post", default=None,
                    help="SGLang /metrics snapshot AFTER the run.")
    pc.add_argument("--out", required=True)
    pc.add_argument("--run-id", required=True)

    args = parser.parse_args()

    if args.mode == "simple":
        render_simple(args)
    elif args.mode == "sweep":
        render_sweep(args)
    elif args.mode == "probe":
        render_probe(args)
    elif args.mode == "stage":
        render_stage(args)
    elif args.mode == "stage-cpu":
        render_stage_cpu(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
