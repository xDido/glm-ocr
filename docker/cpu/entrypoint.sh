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

# Optional pipeline.page_loader block — only appended when any of the four
# knobs is set. Smaller max_pixels cuts image-token count per region and
# directly shortens SGLang prefill (the dominant TTFT component at c≥16).
# Leaving all four unset preserves glmocr upstream defaults. See
# docs/OPTIMIZATIONS.md §TBD (max_pixels shrink).
if [[ -n "${PAGE_LOADER_MAX_PIXELS:-}${PAGE_LOADER_MIN_PIXELS:-}${PAGE_LOADER_T_PATCH_SIZE:-}${PAGE_LOADER_PATCH_EXPAND_FACTOR:-}" ]]; then
    {
        echo ""
        echo "  # Appended by entrypoint when PAGE_LOADER_* env vars are set."
        echo "  page_loader:"
        [[ -n "${PAGE_LOADER_MAX_PIXELS:-}" ]]            && echo "    max_pixels: ${PAGE_LOADER_MAX_PIXELS}"
        [[ -n "${PAGE_LOADER_MIN_PIXELS:-}" ]]            && echo "    min_pixels: ${PAGE_LOADER_MIN_PIXELS}"
        [[ -n "${PAGE_LOADER_T_PATCH_SIZE:-}" ]]          && echo "    t_patch_size: ${PAGE_LOADER_T_PATCH_SIZE}"
        [[ -n "${PAGE_LOADER_PATCH_EXPAND_FACTOR:-}" ]]   && echo "    patch_expand_factor: ${PAGE_LOADER_PATCH_EXPAND_FACTOR}"
    } >> /app/config.yaml
fi

echo "[entrypoint] rendered /app/config.yaml:"
sed 's/^/  | /' /app/config.yaml

export GLMOCR_CONFIG=/app/config.yaml

# Optional: export the layout model to ONNX on first boot. Idempotent — the
# script skips if the target already exists in the HF cache volume.
if [[ "${LAYOUT_ENABLED,,}" == "true" && "${LAYOUT_BACKEND,,}" == "onnx" ]]; then
    echo "[entrypoint] LAYOUT_BACKEND=onnx -> running export (idempotent)"
    python /app/export_layout_onnx.py
fi

# Paddle2ONNX variant: fetch alex-dinh/PP-DocLayoutV3-ONNX into the hf-cache
# if not already present. Idempotent — the check is a plain -f so mounting an
# already-populated cache skips the curl. Unlike the torch export, there's no
# source to re-derive from, so this is download-or-nothing.
if [[ "${LAYOUT_VARIANT,,}" == "paddle2onnx" ]]; then
    : "${HF_HOME:=/root/.cache/huggingface}"
    PADDLE_ONNX_PATH="${HF_HOME}/glmocr-layout-onnx/pp_doclayout_v3_paddle2onnx.onnx"
    if [[ ! -f "${PADDLE_ONNX_PATH}" ]]; then
        echo "[entrypoint] LAYOUT_VARIANT=paddle2onnx -> fetching model (~131 MB)"
        mkdir -p "$(dirname "${PADDLE_ONNX_PATH}")"
        curl -fsSL \
            -o "${PADDLE_ONNX_PATH}" \
            "https://huggingface.co/alex-dinh/PP-DocLayoutV3-ONNX/resolve/main/PP-DocLayoutV3.onnx"
        echo "[entrypoint] fetched $(du -h "${PADDLE_ONNX_PATH}" | cut -f1) → ${PADDLE_ONNX_PATH}"
    else
        echo "[entrypoint] paddle2onnx model already cached at ${PADDLE_ONNX_PATH}"
    fi
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
