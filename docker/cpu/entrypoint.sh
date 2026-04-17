#!/usr/bin/env bash
set -euo pipefail

: "${GLMOCR_PORT:=5002}"
: "${CPU_WORKERS:=2}"
: "${CPU_THREADS:=8}"
: "${GUNICORN_TIMEOUT:=180}"
: "${LAYOUT_ENABLED:=true}"

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

echo "[entrypoint] gunicorn workers=${CPU_WORKERS} threads=${CPU_THREADS} timeout=${GUNICORN_TIMEOUT}"
exec gunicorn \
    --bind "0.0.0.0:${GLMOCR_PORT}" \
    --workers "${CPU_WORKERS}" \
    --threads "${CPU_THREADS}" \
    --worker-class gthread \
    --timeout "${GUNICORN_TIMEOUT}" \
    --graceful-timeout 30 \
    --access-logfile - \
    --error-logfile - \
    wsgi:app
