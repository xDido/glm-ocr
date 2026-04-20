#!/usr/bin/env bash
# OmniDocBench concurrency sweep — asyncio driver.
#
# Runs the same bench at several concurrency levels against one shared
# OmniDocBench image pool and emits ONE markdown comparison report.
# Per-level bench JSONs live only in a temp dir.
#
# Env knobs:
#   SWEEP_CONCURRENCIES   space-separated list (default: "1 2 4 8 16")
#   SWEEP_TOTAL_PER_C     requests per level    (default: 30)
#   OMNIDOC_SAMPLE_POOL   pool size             (default: 64)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib/loadtest_common.sh"

init_run "asyncio-sweep"
log "starting run_id=omnidoc-${TS}-asyncio-sweep"

preflight_omnidoc

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT INT TERM

URLS_FILE="${TMP_DIR}/urls.txt"
build_omnidoc_pool "${URLS_FILE}"
log "pool_size=${POOL_SIZE}"

SWEEP_CONCURRENCIES="${SWEEP_CONCURRENCIES:-1 2 4 8 16}"
SWEEP_TOTAL_PER_C="${SWEEP_TOTAL_PER_C:-30}"
log "sweep concurrencies: ${SWEEP_CONCURRENCIES}"
log "requests per level:  ${SWEEP_TOTAL_PER_C}"

BENCH_JSONS=()
for c in ${SWEEP_CONCURRENCIES}; do
    out="${TMP_DIR}/bench-c${c}.json"
    BENCH_JSONS+=("${out}")
    log "--- running concurrency=${c} total=${SWEEP_TOTAL_PER_C} ---"
    annotate "asyncio-sweep" "asyncio-sweep c=${c} start"

    python loadtest/asyncio/bench.py \
        --host "${CPU_URL}" \
        --concurrency "${c}" \
        --total "${SWEEP_TOTAL_PER_C}" \
        --image-list-file "${URLS_FILE}" \
        --json-out "${out}" \
        --run-id "${TS}-c${c}" \
        --warmup 2 \
        || warn "c=${c} reported failures"

    annotate "asyncio-sweep" "asyncio-sweep c=${c} end"
done

REPORT="${RUN_PREFIX}.md"
python scripts/lib/render_report.py sweep \
    --bench "${BENCH_JSONS[@]}" \
    --out "${REPORT}" \
    --run-id "omnidoc-${TS}-asyncio-sweep" \
    --pool-size "${POOL_SIZE}" \
    --cpu-url "${CPU_URL}"

log "report -> ${REPORT}"
log "Grafana: ${GRAFANA_URL:-http://localhost:3000}${DASHBOARD_PATH}"
