#!/usr/bin/env bash
# OmniDocBench load test — asyncio driver.
#
# Samples OMNIDOC_SAMPLE_POOL images from datasets/OmniDocBench, fires
# concurrency=16 asyncio requests for ~OMNIDOC_DURATION seconds, writes
# JSON summary, and pushes the summary to Pushgateway so Grafana can
# chart it alongside server-side metrics.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib/loadtest_common.sh"

init_run "asyncio"
log "starting run_id=omnidoc-${TS}-asyncio"

preflight_omnidoc

URLS_FILE="${RUN_PREFIX}.urls.txt"
build_omnidoc_pool "${URLS_FILE}"
log "pool_size=${POOL_SIZE} written to ${URLS_FILE}"

# Size --total so ~1 rps/conn × concurrency × duration ≈ total.
total=$((OMNIDOC_DURATION * 4))
out="${RUN_PREFIX}.json"

log "asyncio: concurrency=16 duration~=${OMNIDOC_DURATION}s total=${total}"
annotate "asyncio" "asyncio start"
python loadtest/asyncio/bench.py \
    --host "${CPU_URL}" \
    --concurrency 16 \
    --total "${total}" \
    --image-list-file "${URLS_FILE}" \
    --json-out "${out}" \
    --pushgateway-url "${PUSHGATEWAY_URL}" \
    --run-id "${TS}" \
    --warmup 2 \
    || warn "asyncio reported failures"
annotate "asyncio" "asyncio end"
log "results -> ${out}"

if [[ -f "${out}" ]]; then
    python - <<PY
import json, pathlib
p = pathlib.Path("${out}")
s = json.loads(p.read_text())
lat = s["latency_ms"]
print(f"  asyncio : ok={s['successes']:>4} fail={s['failures']:>3} "
      f"rps={s['throughput_rps']:6.1f}  "
      f"p50={lat['p50']:6.0f}ms p95={lat['p95']:6.0f}ms p99={lat['p99']:6.0f}ms")
PY
fi

log "Grafana dashboard: ${GRAFANA_URL:-http://localhost:3000}${DASHBOARD_PATH}"
