#!/usr/bin/env bash
# End-to-end smoke test: fires a single OCR request at the CPU container
# and fails loudly if the response is not 200 + non-empty JSON.
set -euo pipefail

CPU_URL="${1:-${CPU_URL:-http://localhost:5002}}"
IMAGES_JSON="${IMAGES_JSON:-{\"images\":[\"file:///app/samples/receipt.png\"]}}"

echo "[smoke] CPU   : ${CPU_URL}"
echo "[smoke] body  : ${IMAGES_JSON}"

echo "[smoke] waiting for cpu to respond..."
for i in $(seq 1 60); do
    if curl -fsS -o /dev/null "${CPU_URL}/" 2>/dev/null \
       || curl -fsS -o /dev/null "${CPU_URL}/health" 2>/dev/null; then
        break
    fi
    sleep 2
done

tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT

status=$(curl -s -o "${tmp}" -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -d "${IMAGES_JSON}" \
    "${CPU_URL}/glmocr/parse")

echo "[smoke] http status: ${status}"
echo "[smoke] response (first 500 chars):"
head -c 500 "${tmp}"; echo

if [[ "${status}" != "200" ]]; then
    echo "[smoke] FAILED: expected 200, got ${status}" >&2
    exit 1
fi

if ! grep -q '"markdown_result"\|"json_result"' "${tmp}"; then
    echo "[smoke] FAILED: response missing markdown_result/json_result" >&2
    exit 1
fi

echo "[smoke] OK"
