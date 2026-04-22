"""Render GLM-OCR architecture diagrams as PNGs.

Produces four PNGs next to this script:
    system_overview.png       -> docs/ARCHITECTURE.md §2
    request_lifecycle.png     -> docs/ARCHITECTURE.md §3
    layout_forwarding.png     -> docs/ARCHITECTURE.md §5
    observability_flow.png    -> docs/ARCHITECTURE.md §7

Uses matplotlib only — no additional installs.

    python docs/diagrams/render.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).parent

COLOR_CPU    = "#E3F2FD"
COLOR_GPU    = "#FFEBEE"
COLOR_OBS    = "#E8F5E9"
COLOR_EXT    = "#F5F5F5"
COLOR_WARN   = "#FFF8E1"
BORDER       = "#263238"
BORDER_LIGHT = "#78909C"
ARROW        = "#546E7A"
TEXT         = "#212121"


def _box(ax, x, y, w, h, fill, body, *,
         title=None, fontsize=9, title_fontsize=11,
         align="center", mono=False, border=BORDER, lw=1.2,
         title_pad=0.22, body_pad_top=0.58):
    """Rounded rectangle with optional bold title.

    - Title is drawn near the top, va='top'.
    - Body text is drawn below the title (va='top'), so box height must fit
      all body lines.  Use align='left' to left-justify the body.
    """
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw,
        edgecolor=border,
        facecolor=fill,
    )
    ax.add_patch(patch)

    body_family = "monospace" if mono else "sans-serif"

    if title is not None:
        ax.text(
            x + w / 2, y + h - title_pad,
            title,
            ha="center", va="top",
            fontsize=title_fontsize, fontweight="bold", color=TEXT,
            family="sans-serif",
        )
        body_x = x + 0.18 if align == "left" else x + w / 2
        ax.text(
            body_x, y + h - body_pad_top,
            body,
            ha=align, va="top",
            fontsize=fontsize, color=TEXT,
            family=body_family, linespacing=1.35,
        )
    else:
        body_x = x + 0.18 if align == "left" else x + w / 2
        ax.text(
            body_x, y + h / 2,
            body,
            ha=align, va="center",
            fontsize=fontsize, color=TEXT,
            family=body_family, linespacing=1.35,
        )


def _arrow(ax, x1, y1, x2, y2, label=None, *,
           color=ARROW, lw=1.5, style="->", label_pos=(None, None),
           label_fontsize=8, rad=0.0):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    if label:
        lx = label_pos[0] if label_pos[0] is not None else (x1 + x2) / 2
        ly = label_pos[1] if label_pos[1] is not None else (y1 + y2) / 2
        ax.text(
            lx, ly, label,
            ha="center", va="center",
            fontsize=label_fontsize, color=TEXT,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="none", alpha=0.92),
        )


def _setup(w_units, h_units, figw=14, dpi=160):
    figh = figw * h_units / w_units
    fig, ax = plt.subplots(figsize=(figw, figh), dpi=dpi)
    ax.set_xlim(0, w_units)
    ax.set_ylim(0, h_units)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _save(fig, name):
    out = OUT_DIR / name
    fig.savefig(out, bbox_inches="tight", dpi=160, facecolor="white",
                pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


# -----------------------------------------------------------------------------
# 1. System overview
# -----------------------------------------------------------------------------

def render_system_overview():
    fig, ax = _setup(14, 11)

    ax.text(7, 10.5, "GLM-OCR — system overview",
            ha="center", va="center",
            fontsize=16, fontweight="bold", color=TEXT)

    # External client
    _box(ax, 5.0, 9.3, 4.0, 0.85, COLOR_EXT,
         "POST  /glmocr/parse   (PDF or image bytes)",
         title="External client", fontsize=9, title_fontsize=10,
         title_pad=0.20, body_pad_top=0.55)

    # CPU container
    cpu_body = (
        "wsgi.py  ──▶  glmocr Pipeline\n"
        "               ├─ LayoutDetector.process()   — ONNX + numpy post-proc\n"
        "               └─ OCRClient.process() × N    — ThreadPoolExecutor (32)\n"
        "\n"
        "runtime_app.py instrumentation:\n"
        "  • Histograms:  glmocr_layout_seconds,  glmocr_ocr_region_seconds\n"
        "  • Gauges:      glmocr_in_flight_requests,  glmocr_pipeline_up\n"
        "  • Gauges:      glmocr_config_*   (every .env knob)\n"
        "  • Endpoints:   /runtime,  /runtime/summary,  /metrics"
    )
    _box(ax, 0.3, 4.8, 8.5, 3.9, COLOR_CPU, cpu_body,
         title="glmocr-cpu    (Flask + gunicorn,  4 workers × 16 gthread = 64 slots,  :5002)",
         fontsize=9, title_fontsize=11, align="left", mono=True,
         title_pad=0.24, body_pad_top=0.62)

    # GPU container
    gpu_body = (
        "sglang launch flags:\n"
        "  --max-running-requests 64\n"
        "  --max-total-tokens 200000\n"
        "  --chunked-prefill-size 8192\n"
        "  --speculative-algorithm NEXTN\n"
        "  --schedule-policy lpm\n"
        "  --context-length 24576\n"
        "  --mem-fraction-static 0.95\n"
        "\n"
        "Exposes:\n"
        "  /metrics,  /get_server_info,\n"
        "  /health"
    )
    _box(ax, 9.2, 4.8, 4.5, 3.9, COLOR_GPU, gpu_body,
         title="glmocr-sglang    (1× GPU,  :30000)",
         fontsize=8.5, title_fontsize=10.5, align="left", mono=True,
         title_pad=0.24, body_pad_top=0.62)

    # Prometheus
    _box(ax, 0.3, 2.7, 6.5, 1.6, COLOR_OBS,
         "scrape_interval:  5s     retention:  6h\n"
         "scrapes:  sglang,  cpu,  pushgateway,  locust-exporter,\n"
         "          cadvisor,  dcgm-exporter",
         title="Prometheus   (local,  :9090)",
         fontsize=8.5, title_fontsize=10, align="left", mono=True,
         title_pad=0.22, body_pad_top=0.58)

    # Alloy
    _box(ax, 7.2, 2.7, 6.5, 1.6, COLOR_OBS,
         "scrapes same targets,  applies cardinality allowlist:\n"
         "  glmocr_.* | sglang[:_].* | DCGM_FI_.* |\n"
         "  container_(cpu|memory|network)_.* |\n"
         "  locust_.* | flask_http_.* | up",
         title="Grafana Alloy   — cloud egress",
         fontsize=8, title_fontsize=10, align="left", mono=True,
         title_pad=0.22, body_pad_top=0.58)

    # Local Grafana
    _box(ax, 0.3, 0.5, 6.5, 1.8, COLOR_OBS,
         "Dashboards:\n"
         "  • glmocr-load             (RPS,  latency,  SGLang batching,  GPU, ...)\n"
         "  • glmocr-sweep-progress   (cells,  SLO,  bottleneck classifier)",
         title="Grafana   (local,  :3000)",
         fontsize=8.5, title_fontsize=10, align="left", mono=True,
         title_pad=0.22, body_pad_top=0.58)

    # Cloud Mimir
    _box(ax, 7.2, 0.5, 6.5, 1.8, COLOR_OBS,
         "long-term metric storage\n"
         "+ 2nd datasource wired back into local Grafana\n"
         "  for cross-run / historical analysis",
         title="Grafana Cloud Mimir",
         fontsize=8.5, title_fontsize=10, align="left",
         title_pad=0.22, body_pad_top=0.58)

    # Arrows
    _arrow(ax, 7.0, 9.3, 4.5, 8.7,
           label=None)  # client -> cpu
    _arrow(ax, 8.8, 6.8, 9.2, 6.8,
           label="HTTP\n/v1/chat/completions\nfan-out ×32",
           label_pos=(9.0, 7.9),
           label_fontsize=7)

    # CPU -> Prometheus
    _arrow(ax, 2.0, 4.8, 2.0, 4.3,
           label="/metrics  5s",
           label_pos=(2.75, 4.55),
           label_fontsize=7)
    # GPU -> Prometheus
    _arrow(ax, 10.5, 4.8, 5.0, 4.3,
           label=None, rad=-0.05)
    # CPU -> Alloy
    _arrow(ax, 6.0, 4.8, 9.5, 4.3,
           label=None, rad=0.08)
    # GPU -> Alloy
    _arrow(ax, 11.5, 4.8, 11.5, 4.3,
           label="/metrics  5s",
           label_pos=(12.4, 4.55),
           label_fontsize=7)

    # Prometheus -> Grafana local
    _arrow(ax, 3.0, 2.7, 3.0, 2.3,
           label="PromQL",
           label_pos=(3.7, 2.5),
           label_fontsize=7)
    # Alloy -> Grafana Cloud Mimir
    _arrow(ax, 10.5, 2.7, 10.5, 2.3,
           label="remote_write",
           label_pos=(11.5, 2.5),
           label_fontsize=7)
    # Cloud -> local Grafana (backlink)
    _arrow(ax, 7.2, 1.4, 6.8, 1.4,
           label=None, style="<-", lw=1.0, color=BORDER_LIGHT)

    _save(fig, "system_overview.png")


# -----------------------------------------------------------------------------
# 2. Request lifecycle
# -----------------------------------------------------------------------------

def render_request_lifecycle():
    fig, ax = _setup(14, 12)

    ax.text(7, 11.5, "Request lifecycle — one page end-to-end",
            ha="center", va="center",
            fontsize=16, fontweight="bold", color=TEXT)

    steps = [
        ("1.  Client   POST  /glmocr/parse",
         "PDF or image bytes.  Client-side latency starts here.",
         COLOR_EXT, 0.9),
        ("2.  gunicorn accepts on one of 64 slots",
         "CPU_WORKERS × CPU_THREADS  =  4 × 16.   gthread worker class.",
         COLOR_CPU, 0.9),
        ("3.  Flask  →  glmocr Pipeline",
         "Pre-loaded at fork time via wsgi.py's manual pipeline.start().\n"
         "Model weights already in worker RAM — no cold-load on request path.",
         COLOR_CPU, 1.15),
        ("4.  Layout forward   (ONNX + numpy)",
         "PDF → PIL  →  ORT.InferenceSession.run(pixel_values)  (1 intra-op thread)\n"
         "numpy post-proc:  sigmoid,  top-K,  box decode,  reading-order,  NMS,  polygon\n"
         "Batcher coalesces up to 4 concurrent callers within a 20 ms window.\n"
         "Observed by  glmocr_layout_seconds  Histogram.",
         COLOR_WARN, 1.55),
        ("5.  OCR region fan-out   (up to 32 parallel)",
         "ThreadPoolExecutor fires one /v1/chat/completions per region\n"
         "via a 2048-entry HTTP connection pool.\n"
         "Observed by  glmocr_ocr_region_seconds  Histogram.",
         COLOR_CPU, 1.4),
        ("6.  SGLang batches on GPU",
         "Up to 64 concurrent.   NEXTN speculative decoding  (~2-4×).\n"
         "lpm scheduling  +  chunked prefill  (8192 tokens).\n"
         "sglang_num_running_reqs /  _num_queue_reqs /  _token_usage update live.",
         COLOR_GPU, 1.4),
        ("7.  Pipeline assembles response   →   Flask returns",
         "glmocr_in_flight_requests  decrements.\n"
         "flask_http_request_duration_seconds  Histogram observes end-to-end.",
         COLOR_CPU, 1.15),
    ]

    y = 10.6
    gap = 0.1
    for title, body, fill, h in steps:
        _box(ax, 0.4, y - h, 13.2, h, fill, body,
             title=title, fontsize=8.5, title_fontsize=10.5,
             align="left", mono=True,
             title_pad=0.24, body_pad_top=0.60)
        # arrow to next
        if steps[-1][0] != title:
            _arrow(ax, 7.0, y - h - 0.02, 7.0, y - h - gap + 0.02,
                   label=None, lw=1.0, color=BORDER_LIGHT)
        y -= (h + gap)

    ax.text(
        7, 0.25,
        "Ratio   glmocr_layout_seconds  /  flask_http_request_duration_seconds\n"
        "tells you whether the bottleneck is layout (CPU) or OCR fan-out (GPU).",
        ha="center", va="center", fontsize=8.5, style="italic", color=TEXT,
        family="monospace",
    )

    _save(fig, "request_lifecycle.png")


# -----------------------------------------------------------------------------
# 3. Layout forwarding stages
# -----------------------------------------------------------------------------

def render_layout_forwarding():
    fig, ax = _setup(17, 11)

    ax.text(8.5, 10.4, "Layout forwarding — optimization stages",
            ha="center", va="center",
            fontsize=16, fontweight="bold", color=TEXT)

    # column positions and widths (wider than before to prevent clipping)
    col_x = [0.2, 3.5, 7.0, 10.5, 13.5]
    col_w = [3.2, 3.4, 3.4, 2.9, 3.3]
    headers = ["", "Forward pass", "Post-processing", "Batching", "Impact"]

    header_y = 9.6
    for i, h in enumerate(headers):
        if not h:
            continue
        ax.text(col_x[i] + col_w[i] / 2, header_y, h,
                ha="center", va="center",
                fontsize=11, fontweight="bold", color=TEXT)

    rows = [
        ("Stock\n(upstream glmocr)",
         "torch eager\nHF PPDocLayoutV3",
         "torch\npost_process_object_detection",
         "—",
         "baseline",
         "#ECEFF1"),
        ("Stage 2\nONNX forward",
         "ORT shim\n_OnnxLayoutModel",
         "torch\npost_process_object_detection",
         "—",
         "≈ 1.76× on CPU",
         "#E1F5FE"),
        ("Stage 3\nnumpy post-proc",
         "ORT shim\n_OnnxLayoutModel",
         "numpy port\nlayout_postprocess.py",
         "—",
         "+30–50%\ntorch off request path",
         "#E8F5E9"),
        ("Stage 4\nfused graph\n(tested, not default)",
         "ORT fused graph\n(sigmoid, topK,\nbox decode, order\ndecoder in ONNX)",
         "numpy tail\n(mask, NMS, polygon)",
         "—",
         "IoU 1.0000\nΔ 1.19e-07\n(30 OmniDocBench pages)",
         "#FFF8E1"),
        ("Current\n(.env committed)",
         "ORT shim (raw graph)\n_OnnxLayoutModel",
         "numpy port\nlayout_postprocess.py",
         "queue + single\nbatcher thread\nmax=4, window=20 ms",
         "~ 5× rps\nvs Stage 3 alone",
         "#FCE4EC"),
    ]

    y = 8.9
    row_h = 1.55
    gap = 0.08
    for name, fwd, post, batch, impact, fill in rows:
        y_bot = y - row_h
        for i, text in enumerate([name, fwd, post, batch, impact]):
            mono = i != 0  # left column is sans-serif, rest are monospace
            _box(ax, col_x[i], y_bot, col_w[i], row_h, fill, text,
                 fontsize=9 if i == 0 else 8.5,
                 title=None, align="center", mono=mono)
        y -= (row_h + gap)

    ax.text(
        8.5, 0.35,
        "All five variants coexist behind env flags:\n"
        "LAYOUT_BACKEND ∈ {torch, onnx}     LAYOUT_POSTPROC ∈ {torch, numpy}     "
        "LAYOUT_GRAPH ∈ {raw, fused}     LAYOUT_BATCH_ENABLED ∈ {false, true}\n"
        "Rollback to any earlier stage:  one-line .env edit  +  container restart.",
        ha="center", va="center",
        fontsize=8.5, color=TEXT, family="monospace",
    )

    _save(fig, "layout_forwarding.png")


# -----------------------------------------------------------------------------
# 4. Observability flow
# -----------------------------------------------------------------------------

def render_observability_flow():
    fig, ax = _setup(14, 10)

    ax.text(7, 9.5, "Observability — metric data flow",
            ha="center", va="center",
            fontsize=16, fontweight="bold", color=TEXT)

    # Emitters: two columns of three
    emitters = [
        ("sglang   :30000/metrics",
         "num_running_reqs, num_queue_reqs,\ntoken_usage, cache_hit_rate,\n"
         "gen_throughput",
         COLOR_GPU, 0.2, 6.9),
        ("glmocr-cpu   :5002/metrics",
         "glmocr_layout_seconds,\nglmocr_ocr_region_seconds,\n"
         "glmocr_in_flight_requests,\nglmocr_config_*  (env snapshot),\nflask_http_*",
         COLOR_CPU, 0.2, 4.5),
        ("pushgateway   :9091",
         "asyncio bench summaries,\nsweep_progress_* gauges\n(honor_labels=true)",
         COLOR_EXT, 0.2, 2.1),
        ("dcgm-exporter   (GPU)",
         "GPU util, VRAM,\ntemp, power, SM clocks",
         COLOR_GPU, 4.3, 6.9),
        ("cadvisor",
         "per-container CPU, RSS,\nnetwork, disk",
         COLOR_EXT, 4.3, 4.5),
        ("locust-exporter",
         "live Locust web API state",
         COLOR_EXT, 4.3, 2.1),
    ]

    em_w, em_h = 3.9, 1.6
    for title, body, fill, x, y in emitters:
        _box(ax, x, y, em_w, em_h, fill, body,
             title=title, fontsize=7.5, title_fontsize=9,
             align="left", mono=True,
             title_pad=0.22, body_pad_top=0.55)

    # Prometheus
    _box(ax, 9.2, 6.5, 4.5, 2.0, COLOR_OBS,
         "scrape_interval:  5s\nretention:  6h\n"
         "enable-remote-write-receiver\n  (k6 --out targets this)",
         title="Prometheus   :9090",
         fontsize=8.5, title_fontsize=10.5, align="left", mono=True,
         title_pad=0.22, body_pad_top=0.60)

    # Alloy
    _box(ax, 9.2, 3.9, 4.5, 2.0, COLOR_OBS,
         "scrapes same targets\ncardinality allowlist regex\n"
         "prometheus.remote_write",
         title="Grafana Alloy",
         fontsize=8.5, title_fontsize=10.5, align="left", mono=True,
         title_pad=0.22, body_pad_top=0.60)

    # Local Grafana
    _box(ax, 9.2, 1.5, 4.5, 1.9, COLOR_OBS,
         "glmocr-load dashboard\n"
         "glmocr-sweep-progress dashboard\n"
         "driver start/end annotations",
         title="Grafana   :3000",
         fontsize=8.5, title_fontsize=10.5, align="left", mono=True,
         title_pad=0.22, body_pad_top=0.60)

    # Arrows from each emitter to Prometheus + Alloy
    em_anchors = [(em[3] + em_w, em[4] + em_h / 2) for em in emitters]
    for ax_x, ax_y in em_anchors:
        # to Prometheus
        _arrow(ax, ax_x, ax_y, 9.2, 7.5, label=None, lw=0.9,
               color=BORDER_LIGHT, rad=0.03)
        # to Alloy
        _arrow(ax, ax_x, ax_y, 9.2, 4.9, label=None, lw=0.9,
               color=BORDER_LIGHT, rad=-0.03)

    # Prometheus -> Grafana local
    _arrow(ax, 11.4, 6.5, 11.4, 3.4, label="PromQL",
           label_pos=(12.0, 4.9), label_fontsize=8, rad=0.12)
    # Alloy -> Grafana Cloud (to the right / off-canvas — draw as label)
    # Add a label box noting egress
    ax.text(13.5, 4.9, "──▶  Grafana Cloud\n       (remote_write)",
            ha="left", va="center", fontsize=9, color=TEXT,
            family="monospace")

    ax.text(
        7, 0.4,
        "Prometheus and Alloy scrape the same targets in parallel.\n"
        "Prometheus feeds local dashboards;   Alloy handles cardinality-controlled cloud egress.",
        ha="center", va="center",
        fontsize=8.5, color=TEXT, style="italic",
    )

    _save(fig, "observability_flow.png")


def main() -> int:
    render_system_overview()
    render_request_lifecycle()
    render_layout_forwarding()
    render_observability_flow()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
