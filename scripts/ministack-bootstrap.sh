#!/usr/bin/env bash
# Bootstrap ministack for ECS wiring tests:
#   1. Create an ECS cluster
#   2. Create a Cloud Map HTTP namespace
#   3. Register the CPU + GPU task definitions
#
# This validates the ECS API shape against the ministack emulator. It does
# NOT schedule real GPU workloads — that requires a real NVIDIA host.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -f "${ROOT}/.env" ]]; then
    # shellcheck disable=SC1091
    set -a; source "${ROOT}/.env"; set +a
fi

: "${AWS_REGION:=us-east-1}"
: "${AWS_ACCESS_KEY_ID:=test}"
: "${AWS_SECRET_ACCESS_KEY:=test}"
: "${ECS_ENDPOINT:=http://localhost:4566}"

AWS=(aws --region "${AWS_REGION}" --endpoint-url "${ECS_ENDPOINT}")

echo "[ministack] endpoint=${ECS_ENDPOINT}"

echo "[ministack] creating cluster 'glmocr'..."
"${AWS[@]}" ecs create-cluster --cluster-name glmocr \
    --query 'cluster.clusterArn' --output text || true

echo "[ministack] creating Cloud Map namespace 'glmocr.local'..."
"${AWS[@]}" servicediscovery create-http-namespace \
    --name glmocr.local \
    --query 'OperationId' --output text || true

echo "[ministack] registering task definitions..."
bash "${ROOT}/aws/register-tasks.sh"

echo
echo "[ministack] cluster list:"
"${AWS[@]}" ecs list-clusters --output table || true

echo
echo "[ministack] task definitions:"
"${AWS[@]}" ecs list-task-definitions --output table || true
