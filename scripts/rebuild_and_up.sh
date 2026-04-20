#!/usr/bin/env bash
# rebuild_and_up.sh — clean-rebuild the CPU image (slim + torch+cpu) and
# bring the whole stack back up healthy. Expected runtime:
#   - CPU rebuild: ~3–4 min first time, ~30 s once the slim base +
#     torch+cpu wheel are in the layer cache.
#   - SGLang first-boot: ~5 min (CUDA init + NEXTN setup with cached
#     weights).
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[rebuild] docker compose down..."
docker compose down --remove-orphans

echo "[rebuild] building cpu image using Dockerfile.slim..."
# --no-cache the first time the Dockerfile changes so the new slim
# layer actually pulls. On subsequent runs the cache is a win, so drop
# --no-cache unless FORCE_NO_CACHE is set.
if [[ "${FORCE_NO_CACHE:-0}" == "1" ]]; then
    docker compose build --no-cache cpu
else
    docker compose build cpu
fi

echo "[rebuild] docker compose up -d..."
docker compose up -d

echo "[rebuild] waiting for sglang + cpu healthy (up to 10 min)..."
DEADLINE=$(( $(date +%s) + 600 ))
for svc in sglang cpu; do
    container="glmocr-${svc}"
    while :; do
        status=$(docker inspect --format '{{.State.Health.Status}}' "${container}" 2>/dev/null || echo "missing")
        if [[ "${status}" == "healthy" ]]; then
            echo "[rebuild] ${svc} healthy"
            break
        fi
        if (( $(date +%s) > DEADLINE )); then
            echo "[rebuild] TIMEOUT waiting for ${svc} — last status: ${status}"
            docker compose logs --tail 40 "${svc}"
            exit 1
        fi
        sleep 5
    done
done

echo "[rebuild] stack is healthy — ready for sweep."
