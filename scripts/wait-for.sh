#!/usr/bin/env bash
# Poll an HTTP(S) URL until it returns any response, or time out.
#
# Usage: wait-for.sh <url> [timeout_seconds]
set -euo pipefail

URL="${1:?url required (e.g. http://localhost:5002/health)}"
TIMEOUT="${2:-600}"

echo "[wait-for] ${URL} (timeout ${TIMEOUT}s)"
start=$(date +%s)
while true; do
    if curl -fsS -o /dev/null --max-time 3 "${URL}" 2>/dev/null; then
        echo "[wait-for] ${URL} -> up"
        exit 0
    fi
    now=$(date +%s)
    if (( now - start > TIMEOUT )); then
        echo "[wait-for] ${URL} -> timed out after ${TIMEOUT}s" >&2
        exit 1
    fi
    sleep 3
done
