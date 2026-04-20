#!/usr/bin/env bash
# Register the CPU + GPU ECS task definitions against AWS. Override
# ECS_ENDPOINT only if pointing at a custom endpoint.
#
# Relies on envsubst to expand ${VAR} placeholders inside the task-def JSON.
# Pass all the same knobs from .env — this script sources .env if present.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -f "${ROOT}/.env" ]]; then
    # shellcheck disable=SC1091
    set -a; source "${ROOT}/.env"; set +a
fi

: "${AWS_REGION:=us-east-1}"
: "${AWS_ACCESS_KEY_ID:=test}"
: "${AWS_SECRET_ACCESS_KEY:=test}"
: "${ECS_ENDPOINT:=}"

: "${EXECUTION_ROLE_ARN:=arn:aws:iam::000000000000:role/ecsTaskExecutionRole}"
: "${TASK_ROLE_ARN:=arn:aws:iam::000000000000:role/ecsTaskRole}"
: "${CPU_IMAGE_URI:=glmocr-cpu:local}"
# GPU side runs the official SGLang image straight from Docker Hub — the ECS
# task-def supplies the launch command inline, so no custom image is needed.
: "${SGL_IMAGE_TAG:=latest}"
: "${GPU_IMAGE_URI:=lmsysorg/sglang:${SGL_IMAGE_TAG}}"
: "${HF_TOKEN_SECRET_ARN:=arn:aws:secretsmanager:${AWS_REGION}:000000000000:secret:hf-token}"

export AWS_REGION AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY \
       EXECUTION_ROLE_ARN TASK_ROLE_ARN CPU_IMAGE_URI GPU_IMAGE_URI \
       SGL_IMAGE_TAG \
       HF_TOKEN_SECRET_ARN \
       CPU_WORKERS CPU_THREADS GUNICORN_TIMEOUT OCR_MAX_WORKERS \
       OCR_CONNECT_TIMEOUT OCR_REQUEST_TIMEOUT \
       OCR_RETRY_MAX OCR_RETRY_BACKOFF_BASE OCR_RETRY_BACKOFF_MAX \
       OCR_CONN_POOL OCR_MODEL_NAME \
       LAYOUT_ENABLED LAYOUT_DEVICE LAYOUT_USE_POLYGON \
       SGL_MODEL_PATH SGL_SERVED_MODEL_NAME SGL_TP_SIZE SGL_DTYPE \
       SGL_MAX_RUNNING_REQUESTS SGL_MAX_PREFILL_TOKENS SGL_MAX_TOTAL_TOKENS \
       SGL_MEM_FRACTION_STATIC SGL_CHUNKED_PREFILL SGL_SCHEDULE_POLICY

AWS_ARGS=(--region "${AWS_REGION}")
if [[ -n "${ECS_ENDPOINT}" ]]; then
    AWS_ARGS+=(--endpoint-url "${ECS_ENDPOINT}")
    echo "[register-tasks] using endpoint: ${ECS_ENDPOINT}"
else
    echo "[register-tasks] using real AWS region: ${AWS_REGION}"
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

for fam in cpu gpu; do
    src="${ROOT}/aws/ecs-task-${fam}.json"
    out="${tmp_dir}/ecs-task-${fam}.json"
    envsubst < "${src}" > "${out}"

    # On Windows (Git Bash/MSYS), aws.exe is native and can't resolve MSYS
    # paths like /tmp/... — convert to a Windows-native path for the file://
    # URI. On Linux/macOS cygpath is absent and we use ${out} directly.
    if command -v cygpath >/dev/null 2>&1; then
        uri="file://$(cygpath -m "${out}")"
    else
        uri="file://${out}"
    fi

    echo "[register-tasks] registering ${fam} task definition..."
    aws "${AWS_ARGS[@]}" ecs register-task-definition \
        --cli-input-json "${uri}" \
        --query 'taskDefinition.taskDefinitionArn' --output text
done

echo
echo "[register-tasks] existing task definitions:"
aws "${AWS_ARGS[@]}" ecs list-task-definitions --output table || true
