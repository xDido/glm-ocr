# 12 — CPU container spec (build from scratch)

**Purpose:** enough detail to rebuild `docker/cpu/` from scratch in the new prod repo, without copying any file from the dev repo.

**Ship strategy:** two-stage.
- **MVP (prompt 03):** a minimum-viable container running `glmocr[selfhosted,server]` behind gunicorn with the tuned config + ONNX layout backend. This ships the 1.76× ONNX win and all the HTTP/retry/pool tuning. Skips the custom numpy postproc + layout coalescer (complex Python code that's heavy to re-derive).
- **Full parity (future):** port the numpy postproc + layout coalescer from dev's `runtime_app.py` + `layout_postprocess.py`. These add another ~30% rps at c=12 but they're ~800 lines of Python and bit-parity-validated against upstream. Do this after MVP is live, via a separate PR using the dev repo files as reference (fetch via `git clone` of the dev repo in read-only mode).

The rest of this doc specs the MVP container.

---

## File inventory — what you'll create

```
docker/cpu/
├── Dockerfile
├── entrypoint.sh
├── wsgi.py
├── gunicorn_conf.py
├── config.yaml.template
└── config.layout-off.template   # optional, used only if LAYOUT_ENABLED=false
```

No `runtime_app.py`, no `layout_postprocess.py`, no `async_app.py`, no `export_layout_onnx.py` for MVP. Those are for the "full parity" phase. The ONNX layout win comes from glmocr's own `LAYOUT_BACKEND=onnx` support (it exports on first boot when `onnxruntime` is installed).

---

## `Dockerfile`

```dockerfile
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps:
#   poppler-utils         -> PDF rendering fallback for glmocr
#   libgl1, libglib2.0-0  -> opencv runtime libs (imported by glmocr)
#   gettext-base          -> envsubst (used by entrypoint to render config)
#   curl                  -> container healthcheck
#   g++                   -> occasionally needed for wheel builds
RUN apt-get update && apt-get install -y --no-install-recommends \
      poppler-utils libgl1 libglib2.0-0 gettext-base curl g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch CPU-only from PyTorch's dedicated index.
# The default torch wheel on PyPI pulls nvidia-cuda-*, nvidia-cudnn-*,
# nvidia-nccl-* (~5 GB). This container never uses CUDA; layout runs on
# CPU. Pre-installing +cpu satisfies glmocr's torch requirement so pip
# does not re-resolve it from PyPI in the next step.
RUN pip install \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch>=2.2,<3" \
      "torchvision>=0.17,<1"

# glmocr + runtime dependencies.
#   onnxruntime + onnx + onnxscript: powers LAYOUT_BACKEND=onnx (1.76× on CPU)
#   gunicorn + prometheus-flask-exporter: HTTP server + metrics
#   psutil: for process metrics
RUN pip install \
      "glmocr[selfhosted,server]" \
      "gunicorn>=21.2,<23" \
      "psutil>=5.9" \
      "pyyaml>=6" \
      "prometheus-flask-exporter>=0.23" \
      "prometheus-client>=0.19" \
      "onnxruntime>=1.18" \
      "onnx>=1.16" \
      "onnxscript>=0.1"

COPY wsgi.py            /app/wsgi.py
COPY gunicorn_conf.py   /app/gunicorn_conf.py
COPY config.yaml.template         /app/config.yaml.template
COPY config.layout-off.template   /app/config.layout-off.template
COPY entrypoint.sh      /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 5002

# Invoke via `bash` explicitly — doesn't depend on exec bit surviving
# checkout, or on the shebang being parseable with CRLF line endings.
ENTRYPOINT ["bash", "/app/entrypoint.sh"]
```

---

## `entrypoint.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${GLMOCR_PORT:=5002}"
: "${CPU_WORKERS:=4}"
: "${CPU_THREADS:=16}"
: "${GUNICORN_TIMEOUT:=480}"
: "${LAYOUT_ENABLED:=true}"

if [[ "${LAYOUT_ENABLED,,}" == "false" ]]; then
    TEMPLATE=/app/config.layout-off.template
    echo "[entrypoint] LAYOUT_ENABLED=false -> using layout-bypass template"
else
    TEMPLATE=/app/config.yaml.template
fi

envsubst < "${TEMPLATE}" > /app/config.yaml
echo "[entrypoint] rendered /app/config.yaml:"
sed 's/^/  | /' /app/config.yaml

export GLMOCR_CONFIG=/app/config.yaml

# Prometheus multi-worker aggregation dir. Must be wiped on start so stale
# dead-worker values from previous boots don't pollute the aggregation.
export PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-/tmp/prom_multiproc}"
export prometheus_multiproc_dir="${PROMETHEUS_MULTIPROC_DIR}"
rm -rf "${PROMETHEUS_MULTIPROC_DIR}"
mkdir -p "${PROMETHEUS_MULTIPROC_DIR}"

echo "[entrypoint] gunicorn workers=${CPU_WORKERS} threads=${CPU_THREADS} timeout=${GUNICORN_TIMEOUT}"

exec gunicorn \
    --config /app/gunicorn_conf.py \
    --bind "0.0.0.0:${GLMOCR_PORT}" \
    --workers "${CPU_WORKERS}" \
    --threads "${CPU_THREADS}" \
    --worker-class gthread \
    --timeout "${GUNICORN_TIMEOUT}" \
    --graceful-timeout 30 \
    --access-logfile - \
    --error-logfile - \
    wsgi:app
```

---

## `wsgi.py`

glmocr's `create_app()` builds a Flask app but doesn't call `pipeline.start()` — which is where the layout model is loaded. Without an explicit start, every OCR request trips "Layout detector not started". So we start the pipeline per worker at import time:

```python
"""Gunicorn entry point for glmocr.server.

Loads the config once per worker fork, builds the Flask app, and starts
the pipeline (so the layout detector's model is loaded eagerly). Registers
atexit hooks so the pipeline stops cleanly on worker shutdown.
"""
from __future__ import annotations

import atexit
import os

CONFIG_PATH = os.environ.get("GLMOCR_CONFIG", "/app/config.yaml")


def _load_config(path: str):
    # Try the common loader paths; fall back to YAML.
    try:
        from glmocr.config import load_config  # type: ignore
        return load_config(path)
    except Exception:
        pass
    try:
        from glmocr.config import Config  # type: ignore
        return Config.from_yaml(path)
    except Exception:
        pass
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build_app():
    from glmocr.server import create_app  # type: ignore
    return create_app(_load_config(CONFIG_PATH))


app = _build_app()

_pipeline = app.config.get("pipeline")
if _pipeline is not None:
    _pipeline.start()
    atexit.register(_pipeline.stop)
```

**Note — Prometheus multi-worker exporter.** If you want `/metrics` aggregated across all gunicorn workers, wire `GunicornPrometheusMetrics.for_app_factory()` here. glmocr may already register basic metrics; confirm by hitting `/metrics` after a smoke. If it doesn't, add:

```python
from prometheus_flask_exporter.multiprocess import GunicornPrometheusMetrics
metrics = GunicornPrometheusMetrics(app)
```

---

## `gunicorn_conf.py`

Minimal version for MVP. Covers Prometheus cleanup on worker exit (required for multi-worker counters) and a crash-dump hook:

```python
"""Gunicorn hooks for Prometheus multi-worker metrics + thread-dump diagnosis."""
from __future__ import annotations

import faulthandler
import os
import signal
import sys


def post_fork(server, worker):
    """Per-worker setup. Dumps thread stacks on SIGTERM so we can see what
    was running when gunicorn decided to kill a hung worker."""
    try:
        faulthandler.enable()
        faulthandler.register(signal.SIGTERM, all_threads=True, chain=True)
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except Exception:
        pass


def worker_abort(worker):
    """Called on SIGABRT. Gunicorn raises this when the worker is past --timeout."""
    sys.stderr.write(f"[gunicorn_conf] worker {worker.pid} SIGABRT; thread dump:\n")
    faulthandler.dump_traceback(all_threads=True)
    sys.stderr.flush()


def child_exit(server, worker):
    """Multi-worker Prometheus counter cleanup on exit."""
    try:
        from prometheus_client import multiprocess
        multiprocess.mark_process_dead(worker.pid)
    except ImportError:
        pass
```

---

## `config.yaml.template`

```yaml
server:
  host: 0.0.0.0
  port: ${GLMOCR_PORT}
  debug: false

logging:
  level: INFO

pipeline:
  maas:
    enabled: false

  ocr_api:
    api_scheme: ${SGLANG_SCHEME}
    api_host: ${SGLANG_HOST}
    api_port: ${SGLANG_PORT}
    api_path: /v1/chat/completions
    api_mode: openai
    model: ${OCR_MODEL_NAME}

    connect_timeout: ${OCR_CONNECT_TIMEOUT}
    request_timeout: ${OCR_REQUEST_TIMEOUT}

    retry_max_attempts: ${OCR_RETRY_MAX}
    retry_backoff_base_seconds: ${OCR_RETRY_BACKOFF_BASE}
    retry_backoff_max_seconds: ${OCR_RETRY_BACKOFF_MAX}
    retry_status_codes: [429, 500, 502, 503, 504]

    connection_pool_size: ${OCR_CONN_POOL}
    ssl_verify: false

  max_workers: ${OCR_MAX_WORKERS}

  layout:
    model_dir: ${LAYOUT_MODEL_DIR}       # e.g. PaddlePaddle/PP-DocLayoutV3_safetensors
    device: ${LAYOUT_DEVICE}             # "cpu" in prod
    use_polygon: ${LAYOUT_USE_POLYGON}   # "false" — polygon extraction not needed
```

## `config.layout-off.template`

Same as above but with `pipeline.layout:` omitted entirely (glmocr accepts this — pipeline runs with no layout detection and sends whole pages to SGLang). Optional; only needed if you set `LAYOUT_ENABLED=false` for A/B testing or when layout breaks.

---

## Routes (from glmocr.server)

`glmocr.server.create_app()` mounts:

| Route | Method | Purpose |
|---|---|---|
| `/glmocr/parse` | POST | Main OCR endpoint. Body: `{"images": [...base64 urls...]}`. Returns `{json_result, markdown_result}`. |
| `/health` | GET | Flask liveness. Returns `{"status": "ok"}`, 200. |
| `/metrics` | GET | Prometheus exposition (if `prometheus-flask-exporter` is wired). |

**Do not add custom routes** in MVP. Everything glmocr upstream provides is sufficient for prod.

---

## LAYOUT_BACKEND=onnx — the one non-trivial knob

glmocr's layout detector defaults to torch eager. Setting `LAYOUT_BACKEND=onnx` (picked up by glmocr's own pipeline, if supported) tells it to export the model to ONNX on first boot and run it via ONNX Runtime — 1.76× faster on CPU vs torch eager.

**If glmocr upstream doesn't honor `LAYOUT_BACKEND=onnx` at the config level,** the MVP ships torch-eager layout (still works, just slower). Porting the dev repo's `export_layout_onnx.py` + the `runtime_app.py` ORT shim is the full-parity upgrade path. Test with:

```bash
docker exec glmocr-cpu python -c "from glmocr.config import load_config; c = load_config('/app/config.yaml'); print(dir(c.pipeline.layout))"
```

Look for a `backend` attribute. If present, set `LAYOUT_BACKEND=onnx` via SSM and redeploy. If absent, the flag is a no-op and you're on torch eager until full-parity port.

---

## Healthcheck

```
HEALTHCHECK --interval=10s --timeout=5s --retries=3 --start-period=60s \
    CMD curl -fsS http://localhost:5002/health || exit 1
```

Put this in the Dockerfile OR the ECS task-def `healthCheck:` block — not both (ECS wins if both are set). Recommend ECS-only for prod (task-def changes are tracked; Dockerfile healthcheck is baked in).

---

## Build + smoke locally (optional, before first ECR push)

```bash
docker build -t glmocr-cpu:local docker/cpu/

# Smoke against a dummy SGLang — it'll fail OCR but start cleanly
docker run --rm -p 5002:5002 \
    -e CPU_WORKERS=2 -e CPU_THREADS=8 \
    -e GLMOCR_PORT=5002 \
    -e OCR_MAX_WORKERS=4 -e OCR_CONN_POOL=128 \
    -e OCR_CONNECT_TIMEOUT=5 -e OCR_REQUEST_TIMEOUT=30 \
    -e OCR_RETRY_MAX=1 -e OCR_RETRY_BACKOFF_BASE=0.5 -e OCR_RETRY_BACKOFF_MAX=4 \
    -e OCR_MODEL_NAME=glm-ocr \
    -e SGLANG_HOST=127.0.0.1 -e SGLANG_PORT=30000 -e SGLANG_SCHEME=http \
    -e LAYOUT_ENABLED=true -e LAYOUT_DEVICE=cpu \
    -e LAYOUT_MODEL_DIR=PaddlePaddle/PP-DocLayoutV3_safetensors \
    -e LAYOUT_USE_POLYGON=false \
    -e GUNICORN_TIMEOUT=180 \
    glmocr-cpu:local

# In another terminal
curl localhost:5002/health       # {"status":"ok"}
curl localhost:5002/metrics      # prometheus exposition
```

A real `/glmocr/parse` POST requires SGLang. Prod smoke happens in `prompts/08-first-smoke.md`.

---

## Full-parity phase (future work — NOT in MVP)

When you're ready to port the dev optimizations. Each item below corresponds to a specific dev-repo file; clone the dev repo as a read-only reference:

```bash
git clone <dev-repo-url> /tmp/glmocr-dev   # or scp/rsync from the dev box
```

| Dev file | What to port | Expected win |
|---|---|---|
| `docker/cpu/runtime_app.py` + `layout_postprocess.py` | Numpy post-proc path, drops torch from the request hot path | +6–17% rps, -14–31% p95 |
| `docker/cpu/runtime_app.py` (batcher section, lines ~703-788) | Cross-request layout coalescer (LAYOUT_BATCH_ENABLED) | ~5× rps at c=12 on a 2-worker setup |
| `docker/cpu/export_layout_onnx.py` | Explicit ONNX export (if upstream glmocr doesn't do it automatically) | 1.76× on layout forward |
| `docker/cpu/runtime_app.py` (`/runtime` endpoint) | Runtime introspection endpoint — useful for load tests | Observability only |

Each is an env-flag rollback (LAYOUT_POSTPROC=numpy|torch, LAYOUT_BATCH_ENABLED=true|false). Ship one at a time, matrix-test between each per `09-runbook.md`.

---

Next: [`13-loadtest-spec.md`](./13-loadtest-spec.md).
