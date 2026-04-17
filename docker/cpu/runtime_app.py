"""
Runtime-integrity observability for the CPU container.

Exposes GET /runtime with three nested views so you can verify that the
knobs set via .env actually took effect inside the running process:

  env_claimed     - raw environment variables (what .env declared)
  config_loaded   - values read from /app/config.yaml (what glmocr sees)
  runtime_actual  - live measurements: gunicorn master, worker PIDs, threads,
                    RSS; glmocr Pipeline introspection if reachable
  sglang          - SGLang's own /get_server_info + /metrics (live batching)

This is deliberately side-effect-free: it only reads.
"""
from __future__ import annotations

import json as _json
import os
import socket
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
    "CPU_WORKERS", "CPU_THREADS", "GUNICORN_TIMEOUT", "GLMOCR_PORT",
    "OCR_MAX_WORKERS", "OCR_CONNECT_TIMEOUT", "OCR_REQUEST_TIMEOUT",
    "OCR_RETRY_MAX", "OCR_RETRY_BACKOFF_BASE", "OCR_RETRY_BACKOFF_MAX",
    "OCR_CONN_POOL", "OCR_MODEL_NAME",
    "LAYOUT_ENABLED", "LAYOUT_DEVICE", "LAYOUT_USE_POLYGON",
    "SGLANG_HOST", "SGLANG_PORT", "SGLANG_SCHEME",
    "SGL_MODEL_PATH", "SGL_SERVED_MODEL_NAME", "SGL_TP_SIZE", "SGL_DTYPE",
    "SGL_MAX_RUNNING_REQUESTS", "SGL_MAX_PREFILL_TOKENS",
    "SGL_MAX_TOTAL_TOKENS", "SGL_MEM_FRACTION_STATIC",
    "SGL_CHUNKED_PREFILL", "SGL_SCHEDULE_POLICY",
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
        "pipeline.maas.enabled": pipeline.get("maas", {}).get("enabled") if isinstance(pipeline, dict) else None,
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
                    workers.append({
                        "pid": child.pid,
                        "ppid": child.ppid(),
                        "threads": child.num_threads(),
                        "status": child.status(),
                        "rss_mb": round(child.memory_info().rss / 1_048_576, 1),
                        "started_at": round(child.create_time(), 2),
                        "name": child.name(),
                    })
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
            return {"status": resp.status, "body": _json.loads(resp.read().decode("utf-8"))}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "error": exc.reason, "body": None}
    except Exception as exc:
        return {"status": None, "error": repr(exc), "body": None}


def _http_get_text(url: str, timeout: float = 3.0) -> dict[str, Any]:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"status": resp.status, "text": resp.read().decode("utf-8", errors="replace")}
    except Exception as exc:
        return {"status": None, "error": repr(exc), "text": None}


def _filter_sglang_metrics(raw: str | None) -> dict[str, float | int]:
    """Pick out the batching-relevant lines from SGLang's Prometheus exposition."""
    if not raw:
        return {}
    interesting_prefixes = (
        "sglang:num_running_reqs",
        "sglang:num_queue_reqs",
        "sglang:num_used_tokens",
        "sglang:token_usage",
        "sglang:cache_hit_rate",
        "sglang:max_running_requests",
        "sglang:max_total_num_tokens",
        "sglang:gen_throughput",
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
    return jsonify({
        "generated_at": round(time.time(), 2),
        "cpu": {
            "env_claimed": _env_claimed(),
            "config_loaded": _loaded_config(),
            "runtime_actual": _runtime_actual(),
        },
        "sglang": _sglang_info(),
    })


@bp.route("/runtime/summary", methods=["GET"])
def runtime_summary() -> Any:
    """Terse, human-oriented integrity summary."""
    env = _env_claimed()
    cfg = _loaded_config()
    actual = _runtime_actual()
    sgl = _sglang_info()

    sgl_info_body = sgl["server_info"].get("body") or {}
    sgl_metrics = sgl.get("metrics_live") or {}

    return jsonify({
        "cpu_workers":          {"env": env.get("CPU_WORKERS"),
                                 "actual": actual.get("worker_count")},
        "cpu_threads_per_worker": {"env": env.get("CPU_THREADS"),
                                   "actual_per_worker": [w["threads"] for w in actual.get("workers", [])]},
        "ocr_max_workers":      {"env": env.get("OCR_MAX_WORKERS"),
                                 "config": cfg.get("pipeline.max_workers")},
        "sglang_max_running":   {"env": env.get("SGL_MAX_RUNNING_REQUESTS"),
                                 "runtime": sgl_info_body.get("max_running_requests"),
                                 "live_running": sgl_metrics.get("sglang:num_running_reqs"),
                                 "live_queued":  sgl_metrics.get("sglang:num_queue_reqs")},
        "sglang_batch_tokens":  {"env_prefill": env.get("SGL_MAX_PREFILL_TOKENS"),
                                 "env_total": env.get("SGL_MAX_TOTAL_TOKENS"),
                                 "runtime_prefill": sgl_info_body.get("max_prefill_tokens"),
                                 "runtime_total":   sgl_info_body.get("max_total_tokens")},
        "sglang_dtype":         {"env": env.get("SGL_DTYPE"),
                                 "runtime": sgl_info_body.get("dtype")},
        "sglang_tp_size":       {"env": env.get("SGL_TP_SIZE"),
                                 "runtime": sgl_info_body.get("tp_size")},
        "sglang_mem_fraction":  {"env": env.get("SGL_MEM_FRACTION_STATIC"),
                                 "runtime": sgl_info_body.get("mem_fraction_static")},
        "sglang_chunked":       {"env": env.get("SGL_CHUNKED_PREFILL"),
                                 "runtime": sgl_info_body.get("chunked_prefill_size")
                                             or sgl_info_body.get("enable_chunked_prefill")},
        "sglang_model":         {"env": env.get("SGL_MODEL_PATH"),
                                 "runtime": sgl_info_body.get("model_path")},
    })


def install(app) -> None:
    """
    Attach the runtime blueprint to the glmocr Flask app.

    Falls back to WSGI dispatch if the upstream app isn't a Flask instance,
    so the endpoint still works without relying on glmocr internals.
    """
    try:
        app.register_blueprint(bp)
        return app
    except AttributeError:
        from flask import Flask
        from werkzeug.middleware.dispatcher import DispatcherMiddleware  # type: ignore
        side = Flask("glmocr_runtime_side")
        side.register_blueprint(bp)
        return DispatcherMiddleware(app, {"/runtime": side})
