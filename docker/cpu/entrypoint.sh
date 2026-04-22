#!/usr/bin/env bash
set -euo pipefail

: "${GLMOCR_PORT:=5002}"
: "${CPU_WORKERS:=2}"
: "${CPU_THREADS:=8}"
: "${GUNICORN_TIMEOUT:=180}"
: "${LAYOUT_ENABLED:=true}"
: "${LAYOUT_BACKEND:=torch}"

# Render the config. Two template variants let us toggle layout detection
# without carrying an awkward conditional inside the YAML.
if [[ "${LAYOUT_ENABLED,,}" == "false" ]]; then
    TEMPLATE=/app/config.layout-off.template
    echo "[entrypoint] LAYOUT_ENABLED=false -> using layout-bypass template"
else
    TEMPLATE=/app/config.yaml.template
    echo "[entrypoint] LAYOUT_ENABLED=true  -> using default template"
fi

envsubst < "${TEMPLATE}" > /app/config.yaml
echo "[entrypoint] rendered /app/config.yaml:"
sed 's/^/  | /' /app/config.yaml

export GLMOCR_CONFIG=/app/config.yaml

# Optional: export the layout model to ONNX on first boot. Idempotent — the
# script skips if the target already exists in the HF cache volume.
if [[ "${LAYOUT_ENABLED,,}" == "true" && "${LAYOUT_BACKEND,,}" == "onnx" ]]; then
    echo "[entrypoint] LAYOUT_BACKEND=onnx -> running export (idempotent)"
    python /app/export_layout_onnx.py
fi

# Multi-worker Prometheus metrics need a shared tmpfs-style dir. Wipe it on
# start so stale dead-worker values from previous boots don't pollute the
# aggregation.
export PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-/tmp/prom_multiproc}"
# prometheus_client historically honored `prometheus_multiproc_dir` (lower
# case, without the PROMETHEUS_ prefix); set both for compatibility.
export prometheus_multiproc_dir="${PROMETHEUS_MULTIPROC_DIR}"
rm -rf "${PROMETHEUS_MULTIPROC_DIR}"
mkdir -p "${PROMETHEUS_MULTIPROC_DIR}"

echo "[entrypoint] gunicorn workers=${CPU_WORKERS} threads=${CPU_THREADS} "\
"timeout=${GUNICORN_TIMEOUT}"

# Fix 4 — optional async sidecar. Starts uvicorn on ASYNC_PORT (default 5003)
# running the FastAPI app in docker/cpu/async_app.py. The gunicorn/Flask app
# on ${GLMOCR_PORT} stays exactly as-is. The matrix harness targets the
# async handler by pointing CPU_URL at :${ASYNC_PORT}. Leaving LAYOUT_ASYNC
# unset or "false" is a no-op — the image ships the async code either way.
: "${LAYOUT_ASYNC:=false}"
: "${ASYNC_PORT:=5003}"
: "${ASYNC_WORKERS:=4}"
if [[ "${LAYOUT_ASYNC,,}" == "true" ]]; then
    echo "[entrypoint] LAYOUT_ASYNC=true -> starting uvicorn sidecar on ${ASYNC_PORT} "\
"(workers=${ASYNC_WORKERS})"
    uvicorn async_app:app \
        --host 0.0.0.0 \
        --port "${ASYNC_PORT}" \
        --workers "${ASYNC_WORKERS}" \
        --log-level info \
        --no-access-log \
        &
    ASYNC_PID=$!
    # Forward SIGTERM to the sidecar on container stop.
    trap "kill -TERM ${ASYNC_PID} 2>/dev/null || true" TERM INT EXIT
fi

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
