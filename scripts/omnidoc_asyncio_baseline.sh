#!/usr/bin/env bash
# OmniDocBench baseline latency test — asyncio driver.
#
# Sends BASELINE_TOTAL requests serially (concurrency=1), pacing them
# BASELINE_INTERVAL seconds apart, so each request lands on an idle
# backend. Measures the "warm serial" latency a single user sees when
# the system isn't under load — the right metric for a single-user SLO.
#
# Defaults: 100 requests, 5s apart, wall time ~= 100 * max(5s, latency).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib/loadtest_common.sh"

: "${BASELINE_TOTAL:=100}"
: "${BASELINE_INTERVAL:=5}"

init_run "asyncio-baseline"
log "starting run_id=omnidoc-${TS}-asyncio-baseline"

preflight_omnidoc

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT INT TERM

URLS_FILE="${TMP_DIR}/urls.txt"
build_omnidoc_pool "${URLS_FILE}"
log "pool_size=${POOL_SIZE}"

BENCH_JSON="${TMP_DIR}/bench.json"
REPORT="${RUN_PREFIX}.md"

log "baseline: total=${BASELINE_TOTAL} interval=${BASELINE_INTERVAL}s concurrency=1"
annotate "asyncio-baseline" "baseline start"
python loadtest/asyncio/bench.py \
    --host "${CPU_URL}" \
    --concurrency 1 \
    --interval-seconds "${BASELINE_INTERVAL}" \
    --total "${BASELINE_TOTAL}" \
    --image-list-file "${URLS_FILE}" \
    --json-out "${BENCH_JSON}" \
    --pushgateway-url "${PUSHGATEWAY_URL}" \
    --run-id "${TS}-baseline" \
    --warmup 2 \
    || warn "baseline reported failures"
annotate "asyncio-baseline" "baseline end"

python scripts/lib/render_report.py simple \
    --bench "${BENCH_JSON}" \
    --out "${REPORT}" \
    --run-id "omnidoc-${TS}-asyncio-baseline" \
    --pool-size "${POOL_SIZE}" \
    --cpu-url "${CPU_URL}"

log "report -> ${REPORT}"
log "Grafana dashboard: ${GRAFANA_URL:-http://localhost:3000}${DASHBOARD_PATH}"
