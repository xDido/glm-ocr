#!/usr/bin/env bash
# Integrity probe: hits /runtime/summary on the CPU container and pretty-prints.
# Compares ENV values (what we claimed) against live process/SGLang state.
#
# Usage: scripts/runtime_probe.sh [cpu-base-url]
set -euo pipefail

CPU_URL="${1:-${CPU_URL:-http://localhost:5002}}"
MODE="${MODE:-summary}"    # summary | full

case "${MODE}" in
    summary) path="/runtime/summary" ;;
    full)    path="/runtime" ;;
    *) echo "MODE must be 'summary' or 'full'" >&2; exit 2 ;;
esac

echo "[probe] GET ${CPU_URL}${path}"

if command -v jq >/dev/null 2>&1; then
    curl -fsS "${CPU_URL}${path}" | jq .
else
    curl -fsS "${CPU_URL}${path}"
    echo
    echo "(install jq for pretty-printed output)"
fi
