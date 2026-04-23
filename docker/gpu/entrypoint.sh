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
[[ -n "${SGL_CONTEXT_LENGTH:-}"       ]] && ARGS+=(--context-length        "${SGL_CONTEXT_LENGTH}")
[[ -n "${SGL_SCHEDULE_POLICY:-}"           ]] && ARGS+=(--schedule-policy           "${SGL_SCHEDULE_POLICY}")
[[ -n "${SGL_CUDA_GRAPH_MAX_BS:-}"         ]] && ARGS+=(--cuda-graph-max-bs         "${SGL_CUDA_GRAPH_MAX_BS}")
[[ -n "${SGL_SCHEDULE_CONSERVATIVENESS:-}" ]] && ARGS+=(--schedule-conservativeness "${SGL_SCHEDULE_CONSERVATIVENESS}")

# Chunked prefill: the bare --chunked-prefill flag was renamed to
# --chunked-prefill-size N in modern SGLang. Enable when SGL_CHUNKED_PREFILL
# is truthy; size comes from SGL_CHUNKED_PREFILL_SIZE (default 8192).
CHUNKED="${SGL_CHUNKED_PREFILL:-false}"
if [[ "${CHUNKED,,}" == "true" || "${CHUNKED}" == "1" ]]; then
    : "${SGL_CHUNKED_PREFILL_SIZE:=8192}"
    ARGS+=(--chunked-prefill-size "${SGL_CHUNKED_PREFILL_SIZE}")
fi

# Speculative decoding. zai-org/GLM-OCR ships MTP/NEXTN heads baked into
# the weights; enabling this turns them on for ~2–4x decode throughput on
# typical OCR output with no change to the generated tokens. Defaults
# mirror the upstream README's recommended launch line. Leave
# SGL_SPECULATIVE unset/false to run plain autoregressive decoding.
SPEC="${SGL_SPECULATIVE:-false}"
if [[ "${SPEC,,}" == "true" || "${SPEC}" == "1" ]]; then
    : "${SGL_SPEC_ALGORITHM:=NEXTN}"
    : "${SGL_SPEC_NUM_STEPS:=3}"
    : "${SGL_SPEC_EAGLE_TOPK:=1}"
    : "${SGL_SPEC_NUM_DRAFT_TOKENS:=4}"
    export SGLANG_ENABLE_SPEC_V2=1
    ARGS+=(
        --speculative-algorithm        "${SGL_SPEC_ALGORITHM}"
        --speculative-num-steps        "${SGL_SPEC_NUM_STEPS}"
        --speculative-eagle-topk       "${SGL_SPEC_EAGLE_TOPK}"
        --speculative-num-draft-tokens "${SGL_SPEC_NUM_DRAFT_TOKENS}"
    )
fi

echo "[entrypoint] launching SGLang with args:"
printf '  %s\n' "${ARGS[@]}"

exec python3 -m sglang.launch_server "${ARGS[@]}"
