"""
Runtime-integrity observability for the CPU container.

Exposes GET /runtime with three nested views so you can verify that the
knobs set via .env actually took effect inside the running process:

  env_claimed     - raw environment variables (what .env declared)
  config_loaded   - values read from /app/config.yaml (what glmocr sees)
  runtime_actual  - live measurements: gunicorn master, worker PIDs, threads,
                    RSS; glmocr Pipeline introspection if reachable
  sglang          - SGLang's own /get_server_info + /metrics (live batching)

Also exposes GET /metrics in Prometheus text format via
prometheus-flask-exporter. Auto HTTP counters/histograms are emitted by the
exporter; custom pipeline gauges (queue depth, pipeline_up) are attached via
a dedicated collector when GLMOCR_PIPELINE_METRICS=true.

This module is deliberately side-effect-free with respect to glmocr: it only
reads.
"""

from __future__ import annotations

import json as _json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Any

import yaml
from flask import Blueprint, jsonify

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore

bp = Blueprint("glmocr_runtime", __name__)

CONFIG_PATH = os.environ.get("GLMOCR_CONFIG", "/app/config.yaml")

ENV_KEYS = (
    "CPU_WORKERS",
    "CPU_THREADS",
    "GUNICORN_TIMEOUT",
    "GLMOCR_PORT",
    "OCR_MAX_WORKERS",
    "OCR_CONNECT_TIMEOUT",
    "OCR_REQUEST_TIMEOUT",
    "OCR_RETRY_MAX",
    "OCR_RETRY_BACKOFF_BASE",
    "OCR_RETRY_BACKOFF_MAX",
    "OCR_CONN_POOL",
    "OCR_MODEL_NAME",
    "LAYOUT_ENABLED",
    "LAYOUT_DEVICE",
    "LAYOUT_USE_POLYGON",
    "SGLANG_HOST",
    "SGLANG_PORT",
    "SGLANG_SCHEME",
    "SGL_MODEL_PATH",
    "SGL_SERVED_MODEL_NAME",
    "SGL_TP_SIZE",
    "SGL_DTYPE",
    "SGL_MAX_RUNNING_REQUESTS",
    "SGL_MAX_PREFILL_TOKENS",
    "SGL_MAX_TOTAL_TOKENS",
    "SGL_MEM_FRACTION_STATIC",
    "SGL_CHUNKED_PREFILL",
    "SGL_SCHEDULE_POLICY",
)


def _env_claimed() -> dict[str, str | None]:
    return {k: os.environ.get(k) for k in ENV_KEYS}


def _loaded_config() -> dict[str, Any]:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception as exc:
        return {"_error": repr(exc), "_path": CONFIG_PATH}

    pipeline = cfg.get("pipeline", {}) if isinstance(cfg, dict) else {}
    ocr_api = pipeline.get("ocr_api", {}) if isinstance(pipeline, dict) else {}
    layout = pipeline.get("layout", {}) if isinstance(pipeline, dict) else {}
    return {
        "path": CONFIG_PATH,
        "pipeline.max_workers": pipeline.get("max_workers"),
        "pipeline.ocr_api.connection_pool_size": ocr_api.get("connection_pool_size"),
        "pipeline.ocr_api.request_timeout": ocr_api.get("request_timeout"),
        "pipeline.ocr_api.connect_timeout": ocr_api.get("connect_timeout"),
        "pipeline.ocr_api.retry_max_attempts": ocr_api.get("retry_max_attempts"),
        "pipeline.ocr_api.api_host": ocr_api.get("api_host"),
        "pipeline.ocr_api.api_port": ocr_api.get("api_port"),
        "pipeline.ocr_api.model": ocr_api.get("model"),
        "pipeline.layout.device": layout.get("device"),
        "pipeline.layout.use_polygon": layout.get("use_polygon"),
        "pipeline.maas.enabled": pipeline.get("maas", {}).get("enabled")
        if isinstance(pipeline, dict)
        else None,
    }


def _runtime_actual() -> dict[str, Any]:
    """Live view of gunicorn master + siblings from inside this worker."""
    if psutil is None:
        return {"_error": "psutil not installed"}

    me_pid = os.getpid()
    ppid = os.getppid()
    me = psutil.Process(me_pid)
    try:
        master = psutil.Process(ppid)
    except psutil.NoSuchProcess:
        master = None

    workers: list[dict[str, Any]] = []
    if master is not None:
        for child in master.children(recursive=False):
            try:
                with child.oneshot():
                    workers.append(
                        {
                            "pid": child.pid,
                            "ppid": child.ppid(),
                            "threads": child.num_threads(),
                            "status": child.status(),
                            "rss_mb": round(child.memory_info().rss / 1_048_576, 1),
                            "started_at": round(child.create_time(), 2),
                            "name": child.name(),
                        }
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    return {
        "hostname": socket.gethostname(),
        "host_cpu_count": os.cpu_count(),
        "master": {
            "pid": ppid,
            "cmdline": master.cmdline() if master else None,
            "threads": master.num_threads() if master else None,
        },
        "this_worker": {
            "pid": me_pid,
            "threads": me.num_threads(),
            "rss_mb": round(me.memory_info().rss / 1_048_576, 1),
        },
        "workers": workers,
        "worker_count": len(workers),
        "total_worker_threads": sum(w["threads"] for w in workers),
    }


def _http_get_json(url: str, timeout: float = 3.0) -> dict[str, Any]:
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {
                "status": resp.status,
                "body": _json.loads(resp.read().decode("utf-8")),
            }
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "error": exc.reason, "body": None}
    except Exception as exc:
        return {"status": None, "error": repr(exc), "body": None}


def _http_get_text(url: str, timeout: float = 3.0) -> dict[str, Any]:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {
                "status": resp.status,
                "text": resp.read().decode("utf-8", errors="replace"),
            }
    except Exception as exc:
        return {"status": None, "error": repr(exc), "text": None}


def _filter_sglang_metrics(raw: str | None) -> dict[str, float | int]:
    """Pick out the batching-relevant lines from SGLang's Prometheus exposition.

    Older SGLang builds export metric names with colons (`sglang:num_running_reqs`,
    recording-rule-style). Newer builds switched to underscore (`sglang_num_running_reqs`).
    Accept both so /runtime/summary keeps working across upgrades.
    """
    if not raw:
        return {}
    _bases = (
        "num_running_reqs",
        "num_queue_reqs",
        "num_used_tokens",
        "token_usage",
        "cache_hit_rate",
        "max_running_requests",
        "max_total_num_tokens",
        "gen_throughput",
    )
    interesting_prefixes = tuple(
        f"sglang{sep}{base}" for sep in (":", "_") for base in _bases
    )
    out: dict[str, float | int] = {}
    for line in raw.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if not any(line.startswith(p) for p in interesting_prefixes):
            continue
        try:
            name, value = line.rsplit(" ", 1)
            out[name] = float(value)
        except ValueError:
            continue
    return out


def _sglang_info() -> dict[str, Any]:
    host = os.environ.get("SGLANG_HOST", "sglang")
    port = os.environ.get("SGLANG_PORT", "30000")
    scheme = os.environ.get("SGLANG_SCHEME", "http")
    base = f"{scheme}://{host}:{port}"

    server_info = _http_get_json(f"{base}/get_server_info")
    models = _http_get_json(f"{base}/v1/models")
    metrics_raw = _http_get_text(f"{base}/metrics")

    return {
        "base_url": base,
        "server_info": server_info,
        "v1_models": models,
        "metrics_live": _filter_sglang_metrics(metrics_raw.get("text")),
        "metrics_status": metrics_raw.get("status"),
    }


@bp.route("/runtime", methods=["GET"])
def runtime() -> Any:
    return jsonify(
        {
            "generated_at": round(time.time(), 2),
            "cpu": {
                "env_claimed": _env_claimed(),
                "config_loaded": _loaded_config(),
                "runtime_actual": _runtime_actual(),
            },
            "sglang": _sglang_info(),
        }
    )


@bp.route("/runtime/summary", methods=["GET"])
def runtime_summary() -> Any:
    """Terse, human-oriented integrity summary."""
    env = _env_claimed()
    cfg = _loaded_config()
    actual = _runtime_actual()
    sgl = _sglang_info()

    sgl_info_body = sgl["server_info"].get("body") or {}
    sgl_metrics = sgl.get("metrics_live") or {}

    def _sgl_any(*names: str) -> float | int | None:
        for n in names:
            if n in sgl_metrics:
                return sgl_metrics[n]
        return None

    return jsonify(
        {
            "cpu_workers": {
                "env": env.get("CPU_WORKERS"),
                "actual": actual.get("worker_count"),
            },
            "cpu_threads_per_worker": {
                "env": env.get("CPU_THREADS"),
                "actual_per_worker": [w["threads"] for w in actual.get("workers", [])],
            },
            "ocr_max_workers": {
                "env": env.get("OCR_MAX_WORKERS"),
                "config": cfg.get("pipeline.max_workers"),
            },
            "sglang_max_running": {
                "env": env.get("SGL_MAX_RUNNING_REQUESTS"),
                "runtime": sgl_info_body.get("max_running_requests"),
                "live_running": _sgl_any(
                    "sglang:num_running_reqs", "sglang_num_running_reqs"
                ),
                "live_queued": _sgl_any(
                    "sglang:num_queue_reqs", "sglang_num_queue_reqs"
                ),
            },
            "sglang_batch_tokens": {
                "env_prefill": env.get("SGL_MAX_PREFILL_TOKENS"),
                "env_total": env.get("SGL_MAX_TOTAL_TOKENS"),
                "runtime_prefill": sgl_info_body.get("max_prefill_tokens"),
                "runtime_total": sgl_info_body.get("max_total_tokens"),
            },
            "sglang_dtype": {
                "env": env.get("SGL_DTYPE"),
                "runtime": sgl_info_body.get("dtype"),
            },
            "sglang_tp_size": {
                "env": env.get("SGL_TP_SIZE"),
                "runtime": sgl_info_body.get("tp_size"),
            },
            "sglang_mem_fraction": {
                "env": env.get("SGL_MEM_FRACTION_STATIC"),
                "runtime": sgl_info_body.get("mem_fraction_static"),
            },
            "sglang_chunked": {
                "env": env.get("SGL_CHUNKED_PREFILL"),
                "runtime": sgl_info_body.get("chunked_prefill_size")
                or sgl_info_body.get("enable_chunked_prefill"),
            },
            "sglang_model": {
                "env": env.get("SGL_MODEL_PATH"),
                "runtime": sgl_info_body.get("model_path"),
            },
        }
    )


# Histogram bucket boundaries (seconds) for auto-generated Flask request
# latency metrics. Default prometheus_flask_exporter buckets top out at
# 10 s — every request slower than that collapses into +Inf and all
# histogram_quantile() values get pinned to 10, so p50/p95/p99 render as
# a single flat line. Under real OCR load a cold-start request can easily
# exceed a minute, so extend the tail.
_FLASK_LATENCY_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    30.0,
    60.0,
    90.0,
    120.0,
    180.0,
)


def _install_prometheus(app):
    """Attach a /metrics endpoint to the Flask app. Returns the metrics
    instance (or None on failure)."""
    multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    try:
        if multiproc_dir:
            # GunicornInternalPrometheusMetrics serves /metrics on the main
            # Flask app (same port as /runtime etc). Its sibling
            # GunicornPrometheusMetrics is for a separate metrics HTTP
            # server — it deliberately suppresses the Flask route, so
            # using it here yields 404 on /metrics.
            from prometheus_flask_exporter.multiprocess import (  # type: ignore
                GunicornInternalPrometheusMetrics,
            )

            return GunicornInternalPrometheusMetrics(
                app,
                path="/metrics",
                group_by="url_rule",
                buckets=_FLASK_LATENCY_BUCKETS,
            )
        from prometheus_flask_exporter import PrometheusMetrics  # type: ignore

        return PrometheusMetrics(
            app,
            path="/metrics",
            group_by="url_rule",
            buckets=_FLASK_LATENCY_BUCKETS,
        )
    except Exception as exc:  # pragma: no cover
        print(
            f"[runtime_app] prometheus-flask-exporter init failed: {exc!r}",
            file=sys.stderr,
        )
        return None


def _install_pipeline_gauges(app):
    """Add glmocr-level Gauges backed by prometheus_client's multiprocess
    files (works with gunicorn + multiple workers).

    We use Gauge primitives (not a custom Collector) because in multiproc
    mode PrometheusMetrics.generate_metrics builds a fresh registry with
    only MultiProcessCollector attached, so custom Collectors on
    self.registry are ignored. Gauges write to files in
    PROMETHEUS_MULTIPROC_DIR which MultiProcessCollector reads.
    """
    try:
        from prometheus_client import Gauge  # type: ignore
    except ImportError:
        return

    try:
        pipeline_up = Gauge(
            "glmocr_pipeline_up",
            "1 if the worker's Pipeline has started and loaded the layout model.",
            multiprocess_mode="max",
        )
        in_flight = Gauge(
            "glmocr_in_flight_requests",
            "Number of /glmocr/parse requests currently being processed by this worker.",
            multiprocess_mode="livesum",
        )
    except ValueError:
        # Already registered (module re-imported in the same process).
        return

    # Mark pipeline up when install() runs (wsgi.py calls install AFTER
    # _pipeline.start(), so by definition the layout detector is loaded).
    pipeline = app.config.get("pipeline")
    if pipeline is not None:
        ld = getattr(pipeline, "layout_detector", None)
        if ld is not None and getattr(ld, "_model", None) is not None:
            pipeline_up.set(1)
        else:
            pipeline_up.set(0)
    else:
        pipeline_up.set(0)

    def _track_parse_entry():
        try:
            if request.endpoint == "parse":  # /glmocr/parse handler name
                in_flight.inc()
        except Exception:  # pragma: no cover
            pass

    def _track_parse_exit(response):
        try:
            if request.endpoint == "parse":
                in_flight.dec()
        except Exception:  # pragma: no cover
            pass
        return response

    from flask import request  # local import — only needed when enabled

    app.before_request(_track_parse_entry)
    app.after_request(_track_parse_exit)


def install(app) -> None:
    """
    Attach the runtime blueprint + Prometheus metrics to the glmocr Flask app.

    Falls back to WSGI dispatch if the upstream app isn't a Flask instance,
    so /runtime still works without relying on glmocr internals. /metrics is
    only installed when we have a real Flask app (prometheus-flask-exporter
    hooks into Flask's request lifecycle).
    """
    try:
        app.register_blueprint(bp)
    except AttributeError:
        from flask import Flask
        from werkzeug.middleware.dispatcher import DispatcherMiddleware  # type: ignore

        side = Flask("glmocr_runtime_side")
        side.register_blueprint(bp)
        return DispatcherMiddleware(app, {"/runtime": side})

    _install_prometheus(app)

    if os.environ.get("GLMOCR_PIPELINE_METRICS", "false").lower() in (
        "true",
        "1",
        "yes",
    ):
        _install_pipeline_gauges(app)

    _install_config_gauges()

    return app


# Per-stage timing histograms. The end-to-end request histogram already
# comes from prometheus-flask-exporter; these break that total down into
# the two stages that dominate wall time (layout inference on CPU, OCR
# region calls to SGLang) so we can see where the request is spending
# its time without a profiler.
_LAYOUT_LATENCY_BUCKETS = (
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    8.0,
    12.0,
    20.0,
)
_OCR_REGION_LATENCY_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    10.0,
    30.0,
    60.0,
)


def instrument_pipeline(pipeline) -> None:
    """Wrap the Pipeline's layout + OCR entry points with Histograms so
    /metrics exposes p50/p95/p99 for each stage in isolation.

    Called from wsgi.py after pipeline.start() has loaded the layout
    model. Safe to call multiple times per process — prometheus_client
    raises on duplicate registration and we swallow it. NOT safe to call
    from multiple processes simultaneously at import time; gunicorn fork
    model means each worker calls it once post-fork, which is fine.
    """
    if pipeline is None:
        return
    try:
        from prometheus_client import Histogram  # type: ignore
    except ImportError:
        return

    try:
        layout_hist = Histogram(
            "glmocr_layout_seconds",
            "Wall time spent inside LayoutDetector.process() per call.",
            buckets=_LAYOUT_LATENCY_BUCKETS,
        )
        ocr_hist = Histogram(
            "glmocr_ocr_region_seconds",
            "Wall time spent inside OCRClient.process() per region call.",
            buckets=_OCR_REGION_LATENCY_BUCKETS,
        )
    except ValueError:
        # Already registered (re-import); nothing to do.
        return

    # Layout: one call per page-batch. We only care about wall time; the
    # batch size is visible in other gauges and lots of batches are size 1
    # in this benchmark anyway.
    ld = getattr(pipeline, "layout_detector", None)
    if ld is not None and hasattr(ld, "process"):
        # Two optional layers stack on top of upstream glmocr's default
        # torch-eager layout path:
        #
        #   LAYOUT_BACKEND=onnx     replaces the forward pass with an
        #                           onnxruntime InferenceSession; ld._model
        #                           becomes a SimpleNamespace-returning shim.
        #   LAYOUT_POSTPROC=numpy   replaces ld.process entirely with a
        #                           numpy-only pipeline — no torch tensors
        #                           ever constructed on the request path.
        #                           Requires LAYOUT_BACKEND=onnx.
        #
        # The ONNX graph lives at
        # ${HF_HOME}/glmocr-layout-onnx/pp_doclayout_v3.onnx, produced once
        # by docker/cpu/export_layout_onnx.py at first boot.
        layout_backend = os.environ.get("LAYOUT_BACKEND", "torch").lower()
        layout_postproc = os.environ.get("LAYOUT_POSTPROC", "torch").lower()
        layout_graph = os.environ.get("LAYOUT_GRAPH", "raw").lower()
        # LAYOUT_VARIANT=paddle2onnx swaps the whole layout backend for
        # the alex-dinh/PP-DocLayoutV3-ONNX graph (Paddle2ONNX export of
        # the same weights). The torch export bakes batch=1 into several
        # Reshape initializers — see docs/OPTIMIZATIONS.md. The Paddle
        # export has no baked dims and batch>1 works cleanly. Takes
        # precedence over LAYOUT_BACKEND / LAYOUT_POSTPROC when set.
        layout_variant = os.environ.get("LAYOUT_VARIANT", "torch").lower()
        _numpy_path_installed = False

        if layout_variant == "paddle2onnx":
            try:
                import onnxruntime as _ort
                from layout_paddle2onnx import run_paddle_layout_pipeline

                hf_home = os.environ.get("HF_HOME") or "/root/.cache/huggingface"
                paddle_onnx_path = os.path.join(
                    hf_home,
                    "glmocr-layout-onnx",
                    "pp_doclayout_v3_paddle2onnx.onnx",
                )
                if not os.path.exists(paddle_onnx_path):
                    raise FileNotFoundError(
                        f"Paddle2ONNX graph missing at {paddle_onnx_path}. "
                        "Fetch with: curl -fsSL -o "
                        f"{paddle_onnx_path} "
                        "https://huggingface.co/alex-dinh/PP-DocLayoutV3-ONNX/"
                        "resolve/main/PP-DocLayoutV3.onnx"
                    )

                _paddle_opts = _ort.SessionOptions()
                _paddle_opts.intra_op_num_threads = int(
                    os.environ.get("LAYOUT_ONNX_THREADS", "1")
                )
                _paddle_sess = _ort.InferenceSession(
                    paddle_onnx_path,
                    _paddle_opts,
                    providers=["CPUExecutionProvider"],
                )

                # glmocr uses a collapsed label vocabulary
                # (display_formula + inline_formula → formula, etc.) that
                # its label_task_mapping router knows. The Paddle2ONNX
                # config.json uses the more granular original names.
                # Class IDs are identical (same weights), so reusing
                # glmocr's dict is the correct paddle→torch alignment.
                _ld_id2label = dict(ld._model.config.id2label)
                _ld_label_task_mapping = ld.label_task_mapping
                _ld_threshold = ld.threshold
                _ld_threshold_by_class = ld.threshold_by_class or {}
                _ld_layout_nms = ld.layout_nms
                _ld_layout_unclip_ratio = ld.layout_unclip_ratio
                _ld_layout_merge_bboxes_mode = ld.layout_merge_bboxes_mode
                _ld_batch_size = max(1, int(ld.batch_size))

                # Same sentinel dance as the torch+numpy path — drop the
                # torch model so gunicorn doesn't pay its RAM, while
                # passing glmocr.start()'s not-None check.
                class _PaddleSentinel:
                    def __call__(self, *_a, **_kw):
                        raise RuntimeError(
                            "_PaddleSentinel invoked — LAYOUT_VARIANT=paddle2onnx "
                            "bypasses ld._model; check ld.process override."
                        )

                del ld._model
                import gc as _gc
                _gc.collect()
                ld._model = _PaddleSentinel()

                def _paddle_process(
                    images,
                    save_visualization=False,
                    global_start_idx=0,
                    use_polygon=False,
                ):
                    pil_images = [
                        img.convert("RGB") if img.mode != "RGB" else img
                        for img in images
                    ]
                    all_results = run_paddle_layout_pipeline(
                        pil_images,
                        _paddle_sess,
                        label_task_mapping=_ld_label_task_mapping,
                        id2label=_ld_id2label,
                        threshold=_ld_threshold,
                        threshold_by_class=_ld_threshold_by_class,
                        layout_nms=_ld_layout_nms,
                        layout_unclip_ratio=_ld_layout_unclip_ratio,
                        layout_merge_bboxes_mode=_ld_layout_merge_bboxes_mode,
                        batch_size=_ld_batch_size,
                    )
                    # Visualization path intentionally not implemented —
                    # batch bench doesn't exercise it. Add if needed.
                    return all_results, {}

                ld.process = _paddle_process
                _numpy_path_installed = True
                print(
                    f"[layout] paddle2onnx backend enabled "
                    f"(path={paddle_onnx_path}, "
                    f"intra_op={_paddle_opts.intra_op_num_threads}); "
                    f"torch model weights released; "
                    f"batch>1 works — LAYOUT_BATCH_ENABLED can be true",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[layout] paddle2onnx backend unavailable, "
                    f"falling back: {type(e).__name__}: {e}",
                    flush=True,
                )

        if not _numpy_path_installed and layout_backend == "onnx" and layout_postproc == "numpy":
            try:
                import numpy as _np_mod
                import onnxruntime as _ort
                from layout_postprocess import (
                    compute_paddle_format_results,
                    compute_paddle_format_results_from_fused,
                    paddle_to_all_results,
                )

                hf_home = os.environ.get("HF_HOME") or "/root/.cache/huggingface"
                graph_filename = (
                    "pp_doclayout_v3_fused.onnx"
                    if layout_graph == "fused"
                    else "pp_doclayout_v3.onnx"
                )
                onnx_path = os.path.join(
                    hf_home,
                    "glmocr-layout-onnx",
                    graph_filename,
                )
                if not os.path.exists(onnx_path):
                    raise FileNotFoundError(
                        f"ONNX graph missing at {onnx_path}; "
                        "entrypoint should have produced it"
                    )

                ort_opts = _ort.SessionOptions()
                ort_opts.intra_op_num_threads = int(
                    os.environ.get("LAYOUT_ONNX_THREADS", "1")
                )
                _sess = _ort.InferenceSession(
                    onnx_path,
                    ort_opts,
                    providers=["CPUExecutionProvider"],
                )

                # Capture detector config + processor reference before we
                # release the torch model. We'll use these in every call.
                _ld_id2label = dict(ld._model.config.id2label)
                _ld_label_task_mapping = ld.label_task_mapping
                _ld_threshold = ld.threshold
                _ld_threshold_by_class = ld.threshold_by_class or {}
                _ld_layout_nms = ld.layout_nms
                _ld_layout_unclip_ratio = ld.layout_unclip_ratio
                _ld_layout_merge_bboxes_mode = ld.layout_merge_bboxes_mode
                _ld_batch_size = max(1, int(ld.batch_size))
                _ld_processor = ld._image_processor
                _ld_processor_size = dict(_ld_processor.size)

                # Drop the torch model. After this, ld._model holds a
                # sentinel so upstream _validate_runtime_config() still
                # passes (it only checks for not-None), but no torch
                # tensors are ever constructed on the request path.
                class _NumpySentinel:
                    """Not a torch module. Prevents glmocr.start() from
                    thinking the detector is uninitialized, while making
                    sure anyone who accidentally calls it fails loudly."""

                    def __call__(self, *_a, **_kw):
                        raise RuntimeError(
                            "_NumpySentinel invoked — LAYOUT_POSTPROC=numpy "
                            "bypasses ld._model; check ld.process override."
                        )

                del ld._model
                import gc as _gc

                _gc.collect()
                ld._model = _NumpySentinel()

                if layout_graph == "fused":

                    def _ort_run_fused(np_pixel_values, np_target_sizes):
                        return _sess.run(
                            None,
                            {
                                "pixel_values": np_pixel_values,
                                "target_sizes": np_target_sizes,
                            },
                        )

                    _chunk_orchestrator = lambda pixel_values, img_sizes_wh: (
                        compute_paddle_format_results_from_fused(
                            pixel_values=pixel_values,
                            ort_run_fused=_ort_run_fused,
                            img_sizes_wh=img_sizes_wh,
                            id2label=_ld_id2label,
                            threshold=_ld_threshold,
                            threshold_by_class=_ld_threshold_by_class,
                            layout_nms=_ld_layout_nms,
                            layout_unclip_ratio=_ld_layout_unclip_ratio,
                            layout_merge_bboxes_mode=_ld_layout_merge_bboxes_mode,
                            processor_size=_ld_processor_size,
                        )
                    )
                else:

                    def _ort_run(np_pixel_values):
                        return _sess.run(None, {"pixel_values": np_pixel_values})

                    _chunk_orchestrator = lambda pixel_values, img_sizes_wh: (
                        compute_paddle_format_results(
                            pixel_values=pixel_values,
                            ort_run=_ort_run,
                            img_sizes_wh=img_sizes_wh,
                            id2label=_ld_id2label,
                            threshold=_ld_threshold,
                            threshold_by_class=_ld_threshold_by_class,
                            layout_nms=_ld_layout_nms,
                            layout_unclip_ratio=_ld_layout_unclip_ratio,
                            layout_merge_bboxes_mode=_ld_layout_merge_bboxes_mode,
                            processor_size=_ld_processor_size,
                        )
                    )

                def _numpy_process(
                    images,
                    save_visualization=False,
                    global_start_idx=0,
                    use_polygon=False,
                ):
                    # Match glmocr.layout.PPDocLayoutDetector.process contract:
                    # returns (list[list[dict]], dict[int, Image]).
                    pil_images = [
                        img.convert("RGB") if img.mode != "RGB" else img
                        for img in images
                    ]
                    all_paddle_format: list = []
                    for chunk_start in range(0, len(pil_images), _ld_batch_size):
                        chunk = pil_images[chunk_start : chunk_start + _ld_batch_size]
                        inputs = _ld_processor(images=chunk, return_tensors="np")
                        pixel_values = inputs["pixel_values"]
                        img_sizes_wh = [img.size for img in chunk]
                        paddle_chunk = _chunk_orchestrator(
                            pixel_values,
                            img_sizes_wh,
                        )
                        all_paddle_format.extend(paddle_chunk)

                    img_sizes_wh_full = [img.size for img in pil_images]
                    all_results = paddle_to_all_results(
                        all_paddle_format,
                        img_sizes_wh_full,
                        _ld_label_task_mapping,
                    )

                    vis_images: dict = {}
                    if save_visualization:
                        from glmocr.utils.visualization_utils import (
                            draw_layout_boxes,
                        )

                        for idx, img_results in enumerate(all_paddle_format):
                            vis_images[global_start_idx + idx] = draw_layout_boxes(
                                image=_np_mod.array(pil_images[idx]),
                                boxes=img_results,
                                use_polygon=use_polygon,
                            )
                    return all_results, vis_images

                ld.process = _numpy_process
                _numpy_path_installed = True
                print(
                    f"[layout] numpy postproc enabled "
                    f"(graph={layout_graph}, path={onnx_path}, "
                    f"intra_op={ort_opts.intra_op_num_threads}); "
                    f"torch model weights released; request path is numpy-only",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[layout] numpy postproc unavailable, "
                    f"falling back to torch postproc: {e}",
                    flush=True,
                )

        if not _numpy_path_installed and layout_backend == "onnx":
            # Torch-postproc path: swap ld._model with an ORT shim that
            # returns HF-style torch tensors so upstream post_process_*
            # keeps working unchanged. Measured ~1.76× vs eager torch on
            # single-threaded CPU. intra_op_num_threads=1 matches the
            # production OMP/MKL throttle so concurrent workers don't
            # oversubscribe cores.
            try:
                import onnxruntime as _ort
                from types import SimpleNamespace as _SNS
                import torch as _torch

                hf_home = os.environ.get("HF_HOME") or "/root/.cache/huggingface"
                onnx_path = os.path.join(
                    hf_home,
                    "glmocr-layout-onnx",
                    "pp_doclayout_v3.onnx",
                )
                if not os.path.exists(onnx_path):
                    raise FileNotFoundError(
                        f"ONNX graph missing at {onnx_path}; "
                        "entrypoint should have produced it"
                    )

                ort_opts = _ort.SessionOptions()
                ort_opts.intra_op_num_threads = int(
                    os.environ.get("LAYOUT_ONNX_THREADS", "1")
                )
                _sess = _ort.InferenceSession(
                    onnx_path,
                    ort_opts,
                    providers=["CPUExecutionProvider"],
                )
                _orig_model = ld._model
                _orig_config = getattr(_orig_model, "config", None)

                class _OnnxLayoutModel:
                    """Drop-in replacement for PPDocLayoutV3ForObjectDetection.

                    glmocr calls `self._model(**inputs)` then reads
                    `out.logits` + `out.pred_boxes` from the returned HF
                    output object; a `SimpleNamespace` with those fields
                    is sufficient. We keep `.config` pointing at the
                    original model's config so `id2label` lookups still
                    work unchanged.
                    """

                    config = _orig_config

                    def __call__(self, pixel_values=None, **_kw):
                        np_in = pixel_values.detach().cpu().numpy()
                        logits, pred_boxes, order_logits, out_masks, last_hs = (
                            _sess.run(None, {"pixel_values": np_in})
                        )
                        return _SNS(
                            logits=_torch.from_numpy(logits),
                            pred_boxes=_torch.from_numpy(pred_boxes),
                            order_logits=_torch.from_numpy(order_logits),
                            out_masks=_torch.from_numpy(out_masks),
                            last_hidden_state=_torch.from_numpy(last_hs),
                        )

                    def to(self, _device):
                        return self

                    def eval(self):
                        return self

                ld._model = _OnnxLayoutModel()
                del _orig_model
                import gc as _gc

                _gc.collect()
                print(
                    f"[layout] onnxruntime backend enabled "
                    f"(path={onnx_path}, intra_op={ort_opts.intra_op_num_threads}); "
                    f"torch model weights released",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[layout] onnxruntime backend unavailable, "
                    f"falling back to torch: {e}",
                    flush=True,
                )

        # Optional torch.compile on the underlying HF model. Only meaningful
        # on the torch-postproc path — the numpy path replaces ld._model with
        # a sentinel, so torch.compile would be a no-op (or worse, error).
        # Measured regression on the current 4-worker CPU setup (see .env),
        # so this is off by default anyway.
        if (
            not _numpy_path_installed
            and os.environ.get("LAYOUT_COMPILE", "false").lower() == "true"
        ):
            model = getattr(ld, "_model", None)
            if model is not None:
                try:
                    import torch  # local import: runtime_app must not hard-depend on torch

                    ld._model = torch.compile(
                        model,
                        mode="reduce-overhead",
                        dynamic=True,
                    )
                    print(
                        "[layout] torch.compile enabled (mode=reduce-overhead, dynamic=True)",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[layout] torch.compile skipped: {e}", flush=True)

        original_layout = ld.process

        # Optional cross-request coalescer. Concurrent callers submit single
        # images; one background thread pulls up to LAYOUT_BATCH_MAX of them
        # within LAYOUT_BATCH_WINDOW_MS and calls process(images=[...]) once.
        # The HF detector's `process()` already runs a batched forward pass
        # internally, so coalescing N callers into one call amortizes the
        # per-call preprocessor + postprocessor overhead and lets the model
        # exploit batch-wise matmul efficiency.
        if os.environ.get("LAYOUT_BATCH_ENABLED", "false").lower() == "true":
            import queue as _queue
            import threading as _threading
            from concurrent.futures import Future as _Future

            batch_max = max(1, int(os.environ.get("LAYOUT_BATCH_MAX", "4")))
            window_s = max(
                0.0, float(os.environ.get("LAYOUT_BATCH_WINDOW_MS", "20")) / 1000.0
            )
            q: "_queue.Queue" = _queue.Queue()

            def _batcher_loop():
                while True:
                    first = q.get()
                    if first is None:  # shutdown sentinel
                        return
                    batch = [first]
                    deadline = time.monotonic() + window_s
                    while len(batch) < batch_max:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        try:
                            item = q.get(timeout=remaining)
                            if item is None:
                                return
                            batch.append(item)
                        except _queue.Empty:
                            break
                    images = [b[0] for b in batch]
                    use_poly = batch[0][1]  # visualization-only; safe to share
                    try:
                        all_results, _vis = original_layout(
                            images,
                            save_visualization=False,
                            global_start_idx=0,
                            use_polygon=use_poly,
                        )
                        for (_, _, fut), result_i in zip(batch, all_results):
                            # Caller submitted a single image and expects
                            # the per-single-image return shape: a list of
                            # length 1 of detection lists + empty vis dict.
                            fut.set_result(([result_i], {}))
                    except Exception as exc:
                        for _, _, fut in batch:
                            fut.set_exception(exc)

            t = _threading.Thread(
                target=_batcher_loop,
                name="layout-batcher",
                daemon=True,
            )
            t.start()
            print(
                f"[layout] batcher enabled (max={batch_max}, "
                f"window_ms={int(window_s * 1000)})",
                flush=True,
            )

            def _batched_layout(*args, **kwargs):
                # Extract positional + keyword arguments matching
                # glmocr's PPDocLayoutDetector.process signature.
                if args:
                    images = args[0]
                    save_viz = (
                        args[1]
                        if len(args) > 1
                        else kwargs.get("save_visualization", False)
                    )
                    gsi = (
                        args[2] if len(args) > 2 else kwargs.get("global_start_idx", 0)
                    )
                    use_poly = (
                        args[3] if len(args) > 3 else kwargs.get("use_polygon", False)
                    )
                else:
                    images = kwargs["images"]
                    save_viz = kwargs.get("save_visualization", False)
                    gsi = kwargs.get("global_start_idx", 0)
                    use_poly = kwargs.get("use_polygon", False)
                # Coalesce only single-image, no-vis calls — the common
                # production path. Anything else passes straight through.
                if len(images) != 1 or save_viz:
                    return original_layout(
                        images,
                        save_visualization=save_viz,
                        global_start_idx=gsi,
                        use_polygon=use_poly,
                    )
                fut: "_Future" = _Future()
                q.put((images[0], use_poly, fut))
                return fut.result()

            batched_entry = _batched_layout
        else:
            batched_entry = original_layout

        def _timed_layout(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return batched_entry(*args, **kwargs)
            finally:
                layout_hist.observe(time.perf_counter() - t0)

        ld.process = _timed_layout

    # OCR: one call per region. recognition_worker uses a ThreadPoolExecutor
    # of size `pipeline.max_workers`, so these observations interleave across
    # threads — Histogram is thread-safe.
    oc = getattr(pipeline, "ocr_client", None)
    if oc is not None and hasattr(oc, "process"):
        original_ocr = oc.process

        def _timed_ocr(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original_ocr(*args, **kwargs)
            finally:
                ocr_hist.observe(time.perf_counter() - t0)

        oc.process = _timed_ocr


# Config knobs exposed as gauges so the dashboard can show the current
# tuning values in stat panels (workers, threads, batch caps, etc.).
# Values come from env vars present at worker-start time and never change
# during the process lifetime.
_CONFIG_KNOBS: tuple[tuple[str, str, str], ...] = (
    (
        "glmocr_config_cpu_workers",
        "CPU_WORKERS",
        "Gunicorn worker processes in the CPU container.",
    ),
    (
        "glmocr_config_cpu_threads",
        "CPU_THREADS",
        "Gthread threads per gunicorn worker.",
    ),
    (
        "glmocr_config_ocr_max_workers",
        "OCR_MAX_WORKERS",
        "Per-request SGLang fan-out pool.",
    ),
    (
        "glmocr_config_ocr_conn_pool",
        "OCR_CONN_POOL",
        "HTTP connection pool size for SGLang calls.",
    ),
    (
        "glmocr_config_sgl_max_running",
        "SGL_MAX_RUNNING_REQUESTS",
        "SGLang concurrent-request batching cap.",
    ),
    (
        "glmocr_config_sgl_max_total_tokens",
        "SGL_MAX_TOTAL_TOKENS",
        "SGLang KV-cache total-token budget.",
    ),
    (
        "glmocr_config_sgl_max_prefill_tokens",
        "SGL_MAX_PREFILL_TOKENS",
        "SGLang prefill-token batch cap.",
    ),
    (
        "glmocr_config_sgl_chunked_size",
        "SGL_CHUNKED_PREFILL_SIZE",
        "SGLang chunked-prefill chunk size (tokens).",
    ),
)


def _install_config_gauges() -> None:
    """Publish each tuning knob as a static gauge. Safe to call in every
    gunicorn worker — Gauge with multiprocess_mode='max' collapses the
    per-worker values into one (the values are identical anyway)."""
    try:
        from prometheus_client import Gauge  # type: ignore
    except ImportError:
        return

    for metric_name, env_name, description in _CONFIG_KNOBS:
        raw = os.environ.get(env_name, "") or ""
        try:
            value = float(raw)
        except ValueError:
            # Env var missing or non-numeric — skip registration entirely so
            # the dashboard renders "No data" honestly instead of pretending
            # the server runs with zero threads/workers/tokens.
            continue
        try:
            g = Gauge(metric_name, description, multiprocess_mode="max")
            g.set(value)
        except ValueError:
            # Already registered (re-import).
            continue
