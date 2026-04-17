#!/usr/bin/env bash
# Shared helpers for scripts/omnidoc_*.sh.
# Source this once at the top of each per-driver script; it sets env
# defaults, constants, and exports helper functions (log/warn/die,
# init_run, preflight_omnidoc, build_omnidoc_pool, annotate).

# ---------------------------------------------------------------------------
# Env defaults (only set if not already defined by caller / .env).
# ---------------------------------------------------------------------------
: "${CPU_URL:=http://localhost:5002}"
: "${GRAFANA_URL:=http://localhost:3000}"
: "${GRAFANA_USER:=admin}"
: "${GRAFANA_ADMIN_PASSWORD:=admin}"
: "${OMNIDOC_DURATION:=120}"
: "${OMNIDOC_SAMPLE_POOL:=64}"
: "${PROMETHEUS_URL:=http://localhost:9090}"
: "${PUSHGATEWAY_URL:=http://localhost:9091}"
: "${LOCUST_WEB_PORT:=8089}"

# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------
DATASET_DIR="datasets/OmniDocBench"
CONTAINER_DATASET_ROOT="/app/datasets/OmniDocBench"
RESULTS_DIR="loadtest/results"
DASHBOARD_PATH="/d/glmocr-load/glm-ocr-load-test"

# ---------------------------------------------------------------------------
# Logging. DRIVER_TAG is set by init_run.
# ---------------------------------------------------------------------------
log()  { printf '\033[1;34m[omnidoc-%s]\033[0m %s\n' "${DRIVER_TAG:-?}" "$*"; }
warn() { printf '\033[1;33m[omnidoc-%s]\033[0m %s\n' "${DRIVER_TAG:-?}" "$*" >&2; }
die()  { printf '\033[1;31m[omnidoc-%s]\033[0m %s\n' "${DRIVER_TAG:-?}" "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# init_run <driver>
#   Sets DRIVER_TAG, TS, RUN_PREFIX; creates RESULTS_DIR.
# ---------------------------------------------------------------------------
init_run() {
    DRIVER_TAG="$1"
    TS="$(date +%Y%m%d-%H%M%S)"
    RUN_PREFIX="${RESULTS_DIR}/omnidoc-${TS}-${DRIVER_TAG}"
    mkdir -p "${RESULTS_DIR}"
}

# ---------------------------------------------------------------------------
# preflight_omnidoc
#   Verifies dataset dir exists, CPU /health responds, Grafana reachable.
#   Silently disables Grafana annotations on failure (sets GRAFANA_URL="").
# ---------------------------------------------------------------------------
preflight_omnidoc() {
    [[ -d "${DATASET_DIR}" ]] \
      || die "dataset missing at ${DATASET_DIR} — run 'make datasets-omnidocbench'"

    # Shell redirect (>/dev/null) avoids the mingw curl `-o /dev/null`
    # write-error bug on Windows Git-Bash.
    if ! curl -fsS "${CPU_URL}/health" >/dev/null 2>&1; then
        die "CPU container not responding at ${CPU_URL}/health — run 'make up'"
    fi

    if ! curl -fsS "${GRAFANA_URL}/api/health" >/dev/null 2>&1; then
        warn "Grafana not reachable at ${GRAFANA_URL} — annotations will be skipped"
        GRAFANA_URL=""
    fi
}

# ---------------------------------------------------------------------------
# build_omnidoc_pool <urls_file>
#   Samples OMNIDOC_SAMPLE_POOL images under DATASET_DIR, writes them
#   (rewritten to CONTAINER_DATASET_ROOT) one-per-line to <urls_file>.
#   Sets POOL_SIZE (int) and IMAGES_CSV (comma-joined).
# ---------------------------------------------------------------------------
build_omnidoc_pool() {
    local urls_file="$1"

    # `shuf -n N` consumes stdin cleanly. `shuf | head -n N` would SIGPIPE
    # shuf once head closes stdin; with pipefail that kills the script
    # with exit 141.
    find "${DATASET_DIR}" -type f \
         \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
         2>/dev/null \
         | shuf -n "${OMNIDOC_SAMPLE_POOL}" \
         | sed -E "s#^${DATASET_DIR}#file://${CONTAINER_DATASET_ROOT}#" \
         > "${urls_file}"

    POOL_SIZE="$(wc -l < "${urls_file}" | tr -d ' ')"
    [[ "${POOL_SIZE}" -gt 0 ]] || die "no images found under ${DATASET_DIR}"

    IMAGES_CSV="$(paste -sd, "${urls_file}")"
}

# ---------------------------------------------------------------------------
# annotate <tag> <text>
#   Posts a Grafana annotation. No-op when GRAFANA_URL="".
# ---------------------------------------------------------------------------
annotate() {
    [[ -n "${GRAFANA_URL}" ]] || return 0
    local tag="$1" text="$2"
    local now_ms=$(($(date +%s) * 1000))
    curl -sS -u "${GRAFANA_USER}:${GRAFANA_ADMIN_PASSWORD}" \
        -H "Content-Type: application/json" \
        -d "{\"time\":${now_ms},\"tags\":[\"driver\",\"${tag}\"],\"text\":\"${text}\"}" \
        "${GRAFANA_URL}/api/annotations" >/dev/null 2>&1 || true
}
