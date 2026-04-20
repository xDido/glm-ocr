#!/usr/bin/env bash
# c=6 asyncio run with live probe capture. Polls
# glmocr_in_flight_requests, sglang:num_running_reqs, and
# sglang:num_queue_reqs every 2s so the report can retrospectively
# answer "where is the bottleneck?" by comparing CPU in-flight to
# SGLang running/queued.
#
# Writes ONE markdown report; bench.json, probe.jsonl, and the url
# list all live in a temp dir and are wiped on exit.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib/loadtest_common.sh"

init_run "asyncio-probe-c6"
log "starting run_id=omnidoc-${TS}-asyncio-probe-c6"

preflight_omnidoc

TMP_DIR="$(mktemp -d)"

URLS_FILE="${TMP_DIR}/urls.txt"
build_omnidoc_pool "${URLS_FILE}"
log "pool_size=${POOL_SIZE}"

PROBE_OUT="${TMP_DIR}/probe.jsonl"
BENCH_OUT="${TMP_DIR}/bench.json"
REPORT="${RUN_PREFIX}.md"

log "starting probe -> ${PROBE_OUT}"
python scripts/runtime_probe_loop.py "${PROBE_OUT}" &
PROBE_PID=$!
log "probe PID=${PROBE_PID}"

cleanup() {
    if kill -0 "${PROBE_PID}" 2>/dev/null; then
        log "stopping probe (PID=${PROBE_PID})"
        kill "${PROBE_PID}" 2>/dev/null || true
        wait "${PROBE_PID}" 2>/dev/null || true
    fi
    rm -rf "${TMP_DIR}"
}
trap cleanup EXIT INT TERM

annotate "asyncio-probe-c6" "asyncio-probe-c6 start"
log "running bench: concurrency=6 total=480"
python loadtest/asyncio/bench.py \
    --host "${CPU_URL}" \
    --concurrency 6 \
    --total 480 \
    --image-list-file "${URLS_FILE}" \
    --json-out "${BENCH_OUT}" \
    --pushgateway-url "${PUSHGATEWAY_URL}" \
    --run-id "${TS}-c6" \
    --warmup 2 \
    || warn "bench reported failures"
annotate "asyncio-probe-c6" "asyncio-probe-c6 end"

# Stop the probe now so the report captures the full jsonl file.
if kill -0 "${PROBE_PID}" 2>/dev/null; then
    kill "${PROBE_PID}" 2>/dev/null || true
    wait "${PROBE_PID}" 2>/dev/null || true
fi

python scripts/lib/render_report.py probe \
    --bench "${BENCH_OUT}" \
    --probe "${PROBE_OUT}" \
    --out "${REPORT}" \
    --run-id "omnidoc-${TS}-asyncio-probe-c6" \
    --pool-size "${POOL_SIZE}" \
    --cpu-url "${CPU_URL}" \
    --prom-url "${PROMETHEUS_URL:-http://localhost:9090}"

log "report -> ${REPORT}"
log "Grafana: ${GRAFANA_URL:-http://localhost:3000}${DASHBOARD_PATH}"
