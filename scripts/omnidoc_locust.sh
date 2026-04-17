#!/usr/bin/env bash
# OmniDocBench load test — locust driver.
#
# Samples OMNIDOC_SAMPLE_POOL images, runs locust headless for
# OMNIDOC_DURATION seconds at 50 users / 5 ramp, emits CSV under
# loadtest/results/omnidoc-<ts>-locust_stats.csv. The web UI runs on
# LOCUST_WEB_PORT so locust_exporter can scrape it during the run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib/loadtest_common.sh"

init_run "locust"
log "starting run_id=omnidoc-${TS}-locust"

preflight_omnidoc

URLS_FILE="${RUN_PREFIX}.urls.txt"
build_omnidoc_pool "${URLS_FILE}"
log "pool_size=${POOL_SIZE} written to ${URLS_FILE}"

csv_prefix="${RUN_PREFIX}"

log "locust: -u 50 -r 5 -t ${OMNIDOC_DURATION}s (web UI on :${LOCUST_WEB_PORT} for exporter)"
annotate "locust" "locust start"
(
    cd loadtest/locust
    LOCUST_IMAGES="${IMAGES_CSV}" \
    locust -f locustfile.py --headless \
        -u 50 -r 5 -t "${OMNIDOC_DURATION}s" \
        --host "${CPU_URL}" \
        --web-port "${LOCUST_WEB_PORT}" \
        --csv "../../${csv_prefix}" \
        || true
)
annotate "locust" "locust end"
log "results -> ${csv_prefix}_stats.csv"

if [[ -f "${csv_prefix}_stats.csv" ]]; then
    python - <<PY
import csv, pathlib
p = pathlib.Path("${csv_prefix}_stats.csv")
with p.open() as fh:
    rows = list(csv.DictReader(fh))
agg = next((r for r in rows if r.get("Name") == "Aggregated"), rows[-1] if rows else None)
if agg:
    rc = int(agg.get("Request Count") or 0)
    fc = int(agg.get("Failure Count") or 0)
    med = float(agg.get("Median Response Time") or 0)
    p95 = float(agg.get("95%") or 0)
    p99 = float(agg.get("99%") or 0)
    rps = float(agg.get("Requests/s") or 0)
    print(f"  locust  : ok={rc-fc:>4} fail={fc:>3} rps={rps:6.1f}  "
          f"p50={med:6.0f}ms p95={p95:6.0f}ms p99={p99:6.0f}ms")
PY
fi

log "Grafana dashboard: ${GRAFANA_URL:-http://localhost:3000}${DASHBOARD_PATH}"
