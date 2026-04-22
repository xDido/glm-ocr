#!/usr/bin/env bash
# OmniDocBench asyncio combined matrix — loaded sweep.
#
# Runs three bench trials in one pass, sharing a single pinned image
# pool (MATRIX_POOL_SEED) so per-cell latency deltas reflect concurrency
# only, not workload variance. Renders ONE combined markdown report via
# render_report.py's sweep mode.
#
# Trial matrix:
#   loaded, back-to-back : c=12, c=24, c=32
#
# Paced baselines (c=1 at intervals) were removed after measurement —
# they produced essentially the same p50/p95/p99 at i=1/5/10s within
# sampling noise, and c=1 latency is also recoverable from any loaded
# trial's min/low-percentile samples. Skipping them saves ~10 min.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib/loadtest_common.sh"

: "${MATRIX_TOTAL:=100}"
: "${MATRIX_POOL_SEED:=42}"
: "${MATRIX_POOL_SIZE:=128}"

# Make the pool deterministic so runs on different days compare cleanly.
# build_omnidoc_pool honors POOL_SEED + OMNIDOC_SAMPLE_POOL; this forwards
# MATRIX_POOL_SEED / MATRIX_POOL_SIZE so callers only need to think about
# these two knobs. Pool size is pinned to match the 2026-04-20 baseline
# (128 images) so future matrix deltas reflect config changes, not a
# shrunken pool.
export POOL_SEED="${MATRIX_POOL_SEED}"
export OMNIDOC_SAMPLE_POOL="${MATRIX_POOL_SIZE}"

init_run "asyncio-matrix"
log "starting run_id=omnidoc-${TS}-asyncio-matrix total=${MATRIX_TOTAL} pool_seed=${MATRIX_POOL_SEED} pool_size=${MATRIX_POOL_SIZE}"

preflight_omnidoc

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT INT TERM

URLS_FILE="${TMP_DIR}/urls.txt"
build_omnidoc_pool "${URLS_FILE}"
log "pool_size=${POOL_SIZE}"

BENCH_JSONS=()

run_trial() {
    local label="$1" conc="$2" interval="$3"
    local out="${TMP_DIR}/bench-${label}.json"
    log "trial label=${label} concurrency=${conc} interval=${interval}s"
    annotate "asyncio-matrix" "trial ${label} start"
    python loadtest/asyncio/bench.py \
        --host "${CPU_URL}" \
        --concurrency "${conc}" \
        --interval-seconds "${interval}" \
        --total "${MATRIX_TOTAL}" \
        --pool-seed "${MATRIX_POOL_SEED}" \
        --image-list-file "${URLS_FILE}" \
        --json-out "${out}" \
        --pushgateway-url "${PUSHGATEWAY_URL}" \
        --run-id "${TS}-${label}" \
        --warmup 2 \
        || warn "trial ${label} reported failures"
    annotate "asyncio-matrix" "trial ${label} end"
    BENCH_JSONS+=("${out}")
}

# Fail-fast: loaded trials first (~30-90 s each). If the stack is broken
# under load we find out inside 3 minutes rather than after 40 min of
# paced baselines.
#
# c=40 / c=64 extend the sweep past the 4-worker × 16-thread admission
# ceiling so we can see where throughput peaks and tails blow up.
#
# The 25 s gap between trials is load-bearing: augment_matrix_report.py
# segments Prometheus's glmocr_in_flight_requests timeline by >=20 s of
# idle, and uses those segments to retrofit per-trial phase decomposition.
# Without the gap, all five trials merge into one segment and the
# augmenter can't match them back to report rows.
run_trial "c12"    12 0
sleep 25
run_trial "c24"    24 0
sleep 25
run_trial "c32"    32 0
sleep 25
run_trial "c40"    40 0
sleep 25
run_trial "c64"    64 0

REPORT="${RUN_PREFIX}.md"
python scripts/lib/render_report.py sweep \
    --bench "${BENCH_JSONS[@]}" \
    --out "${REPORT}" \
    --run-id "omnidoc-${TS}-asyncio-matrix" \
    --pool-size "${POOL_SIZE}" \
    --cpu-url "${CPU_URL}"

# Retroactively append CPU + SGLang per-trial runtime signals + phase
# decomposition to the report by matching trial wall times against
# Prometheus's glmocr_in_flight_requests timeline. Requires the 25 s
# inter-trial gaps above so each trial is its own segment. Non-fatal
# if Prometheus is unreachable (e.g. observability stack not up).
python scripts/augment_matrix_report.py --report "${REPORT}" \
    || warn "augment_matrix_report.py failed; report has bench data only"

# Preserve the image pool next to the report so future runs can diff /
# re-feed it and get apples-to-apples numbers. Without this, a doc-
# complexity shift between runs looks like a perf regression.
URLS_COPY="${RUN_PREFIX}.urls.txt"
cp "${URLS_FILE}" "${URLS_COPY}"
log "pool urls -> ${URLS_COPY} (pool_seed=${MATRIX_POOL_SEED}, pool_size=${POOL_SIZE})"

log "report -> ${REPORT}"
log "Grafana dashboard: ${GRAFANA_URL:-http://localhost:3000}${DASHBOARD_PATH}"
