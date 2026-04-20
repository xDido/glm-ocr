#!/usr/bin/env bash
# OmniDocBench load test — asyncio driver.
#
# Samples OMNIDOC_SAMPLE_POOL images from datasets/OmniDocBench, fires
# concurrency=16 asyncio requests for ~OMNIDOC_DURATION seconds, pushes
# the summary to Pushgateway (for live Grafana visibility), and writes
# ONE self-contained markdown report to loadtest/results/. Every other
# artifact (urls list, bench JSON) lives only in a temp dir.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib/loadtest_common.sh"

init_run "asyncio"
log "starting run_id=omnidoc-${TS}-asyncio"

preflight_omnidoc

# All intermediates live in a per-run temp dir that is wiped on exit.
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT INT TERM

URLS_FILE="${TMP_DIR}/urls.txt"
build_omnidoc_pool "${URLS_FILE}"
log "pool_size=${POOL_SIZE}"

# Size --total so ~1 rps/conn × concurrency × duration ≈ total.
total=$((OMNIDOC_DURATION * 4))
BENCH_JSON="${TMP_DIR}/bench.json"
REPORT="${RUN_PREFIX}.md"

log "asyncio: concurrency=16 duration~=${OMNIDOC_DURATION}s total=${total}"
annotate "asyncio" "asyncio start"
python loadtest/asyncio/bench.py \
    --host "${CPU_URL}" \
    --concurrency 16 \
    --total "${total}" \
    --image-list-file "${URLS_FILE}" \
    --json-out "${BENCH_JSON}" \
    --pushgateway-url "${PUSHGATEWAY_URL}" \
    --run-id "${TS}" \
    --warmup 2 \
    || warn "asyncio reported failures"
annotate "asyncio" "asyncio end"

python scripts/lib/render_report.py simple \
    --bench "${BENCH_JSON}" \
    --out "${REPORT}" \
    --run-id "omnidoc-${TS}-asyncio" \
    --pool-size "${POOL_SIZE}" \
    --cpu-url "${CPU_URL}"

log "report -> ${REPORT}"
log "Grafana dashboard: ${GRAFANA_URL:-http://localhost:3000}${DASHBOARD_PATH}"
