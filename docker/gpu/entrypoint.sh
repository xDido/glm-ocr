#!/usr/bin/env bash
set -euo pipefail

: "${SGL_MODEL_PATH:=zai-org/GLM-OCR}"
: "${SGL_SERVED_MODEL_NAME:=glm-ocr}"
: "${SGL_TP_SIZE:=1}"
: "${SGL_DTYPE:=float16}"

ARGS=(
    --model-path          "${SGL_MODEL_PATH}"
    --served-model-name   "${SGL_SERVED_MODEL_NAME}"
    --host                0.0.0.0
    --port                30000
    --tp-size             "${SGL_TP_SIZE}"
    --dtype               "${SGL_DTYPE}"
    --trust-remote-code
    --enable-metrics
)

# Optional knobs: appended only when the env var is present.
[[ -n "${SGL_MAX_RUNNING_REQUESTS:-}" ]] && ARGS+=(--max-running-requests  "${SGL_MAX_RUNNING_REQUESTS}")
[[ -n "${SGL_MAX_PREFILL_TOKENS:-}"   ]] && ARGS+=(--max-prefill-tokens    "${SGL_MAX_PREFILL_TOKENS}")
[[ -n "${SGL_MAX_TOTAL_TOKENS:-}"     ]] && ARGS+=(--max-total-tokens      "${SGL_MAX_TOTAL_TOKENS}")
[[ -n "${SGL_MEM_FRACTION_STATIC:-}"  ]] && ARGS+=(--mem-fraction-static   "${SGL_MEM_FRACTION_STATIC}")
[[ -n "${SGL_SCHEDULE_POLICY:-}"      ]] && ARGS+=(--schedule-policy       "${SGL_SCHEDULE_POLICY}")

# Presence-only flag (no value).
CHUNKED="${SGL_CHUNKED_PREFILL:-false}"
if [[ "${CHUNKED,,}" == "true" || "${CHUNKED}" == "1" ]]; then
    ARGS+=(--chunked-prefill)
fi

echo "[entrypoint] launching SGLang with args:"
printf '  %s\n' "${ARGS[@]}"

exec python3 -m sglang.launch_server "${ARGS[@]}"
