#!/usr/bin/env bash
# OmniDocBench load test — k6 driver.
#
# Samples OMNIDOC_SAMPLE_POOL images, runs k6 with constant-VU hold for
# OMNIDOC_DURATION seconds, remote-writes live metrics to Prometheus,
# exports the summary JSON under loadtest/results/omnidoc-<ts>-k6.json.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib/loadtest_common.sh"

init_run "k6"
log "starting run_id=omnidoc-${TS}-k6"

preflight_omnidoc

URLS_FILE="${RUN_PREFIX}.urls.txt"
build_omnidoc_pool "${URLS_FILE}"
log "pool_size=${POOL_SIZE} written to ${URLS_FILE}"

out="${RUN_PREFIX}.json"

log "k6: DURATION=${OMNIDOC_DURATION}s (remote-writing to ${PROMETHEUS_URL})"
annotate "k6" "k6 start"
IMAGES="${IMAGES_CSV}" DURATION="${OMNIDOC_DURATION}" \
K6_PROMETHEUS_RW_SERVER_URL="${PROMETHEUS_URL}/api/v1/write" \
K6_PROMETHEUS_RW_TREND_STATS="p(50),p(90),p(95),p(99),min,max,avg" \
    k6 run loadtest/k6/ocr_load.js \
        -e HOST="${CPU_URL}" \
        --out "experimental-prometheus-rw" \
        --tag "run_id=${TS}" \
        --summary-export="${out}" \
        || warn "k6 exited non-zero (may be threshold violation)"
annotate "k6" "k6 end"
log "results -> ${out}"

if [[ -f "${out}" ]]; then
    python - <<PY
import json, pathlib
p = pathlib.Path("${out}")
s = json.loads(p.read_text())
metrics = s.get("metrics", {})
reqs = metrics.get("http_reqs", {})
total_reqs = int((reqs.get("values") or {}).get("count") or 0)
fail_rate_m = metrics.get("http_req_failed", {}).get("values") or {}
fail_rate = float(fail_rate_m.get("rate") or 0.0)
dur = metrics.get("http_req_duration", {}).get("values") or {}
p50 = float(dur.get("med") or 0)
p95 = float(dur.get("p(95)") or 0)
p99 = float(dur.get("p(99)") or 0)
run_state = s.get("state", {}).get("testRunDurationMs", 0) / 1000.0
rps = total_reqs / run_state if run_state else 0
print(f"  k6      : ok={int(total_reqs*(1-fail_rate)):>4} "
      f"fail={int(total_reqs*fail_rate):>3} "
      f"rps={rps:6.1f}  "
      f"p50={p50:6.0f}ms p95={p95:6.0f}ms p99={p99:6.0f}ms")
PY
fi

log "Grafana dashboard: ${GRAFANA_URL:-http://localhost:3000}${DASHBOARD_PATH}"
