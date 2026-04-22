#!/usr/bin/env bash
# Speculative-decoding sweep on SGLang.
#
# Iterates SGL_SPEC_NUM_STEPS x SGL_SPEC_NUM_DRAFT_TOKENS, subject to
# draft >= steps + 1 (SGLang invariant). For each cell:
#   1. Edit .env to new values.
#   2. docker compose up -d --force-recreate sglang (model stays cached;
#      only the scheduler + speculative config re-inits).
#   3. Wait for /health to return 200.
#   4. Warm the CPU workers.
#   5. Run MATRIX_TOTAL=200 asyncio matrix.
#   6. Record <cell-label> -> <report-path> in /tmp/spec_sweep_map.tsv.
#
# Baseline (current) is (steps=3, draft=4). Current OpenVINO matrices at
# that config are already committed as reference.
set -u  # don't -e — a single trial's matrix failure shouldn't abort the whole sweep
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MAP_FILE="/tmp/spec_sweep_map.tsv"
: > "${MAP_FILE}"
printf 'label\treport\trps12\tp99_12\trps24\tp99_24\trps32\tp99_32\tfails\n' >> "${MAP_FILE}"

SAMPLE_URL='{"images":["file:///app/datasets/OmniDocBench/images/PPT_1001115_eng_page_003.png"]}'

run_cell() {
    local ns=$1 ndt=$2
    local label="spec_ns${ns}_ndt${ndt}"
    echo "================================================================"
    echo "[sweep] cell ${label}  (SGL_SPEC_NUM_STEPS=${ns} SGL_SPEC_NUM_DRAFT_TOKENS=${ndt})"
    echo "================================================================"

    # 1. Patch .env
    sed -i "s/^SGL_SPEC_NUM_STEPS=.*/SGL_SPEC_NUM_STEPS=${ns}/" .env
    sed -i "s/^SGL_SPEC_NUM_DRAFT_TOKENS=.*/SGL_SPEC_NUM_DRAFT_TOKENS=${ndt}/" .env

    # 2. Restart sglang with new env
    docker compose up -d --force-recreate sglang >/dev/null 2>&1
    echo "[sweep] sglang restart requested; waiting for /health..."

    # 3. Wait for healthy (max ~4 min)
    local waited=0
    while ! curl -fsS http://localhost:30000/health >/dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [ $waited -gt 240 ]; then
            echo "[sweep] sglang failed to become healthy in 240s; skipping cell ${label}"
            printf '%s\tSKIPPED\t-\t-\t-\t-\t-\t-\t-\n' "${label}" >> "${MAP_FILE}"
            return 0
        fi
    done
    echo "[sweep] sglang healthy after ${waited}s"

    # 4. Warm CPU workers (20 concurrent hits)
    for i in $(seq 1 20); do
        curl -s -o /dev/null -X POST -H "Content-Type: application/json" \
             -d "${SAMPLE_URL}" \
             http://localhost:5002/glmocr/parse &
    done
    wait

    # 5. Matrix run
    MATRIX_TOTAL=200 bash scripts/omnidoc_asyncio_matrix.sh 2>&1 \
        | tee "/tmp/sweep_${label}.log" \
        | tail -5

    # 6. Find the matrix report just produced (newest)
    local report
    report=$(ls -t loadtest/results/omnidoc-*-asyncio-matrix.md 2>/dev/null | head -1)

    # Parse RPS + p99 at c=12/24/32 + total failures
    local rps12 p99_12 rps24 p99_24 rps32 p99_32 fails
    rps12=$(awk -F'|' '/^\| 12 \|/ {gsub(/[ ,]/,"",$8); print $8; exit}' "${report}")
    p99_12=$(awk -F'|' '/^\| 12 \|/ {gsub(/[ ,]/,"",$13); print $13; exit}' "${report}")
    rps24=$(awk -F'|' '/^\| 24 \|/ {gsub(/[ ,]/,"",$8); print $8; exit}' "${report}")
    p99_24=$(awk -F'|' '/^\| 24 \|/ {gsub(/[ ,]/,"",$13); print $13; exit}' "${report}")
    rps32=$(awk -F'|' '/^\| 32 \|/ {gsub(/[ ,]/,"",$8); print $8; exit}' "${report}")
    p99_32=$(awk -F'|' '/^\| 32 \|/ {gsub(/[ ,]/,"",$13); print $13; exit}' "${report}")
    fails=$(awk -F'|' '/^\| (12|24|32|40|64) \|/ {gsub(/[ ,]/,"",$5); sum+=$5} END {print sum+0}' "${report}")

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${label}" "${report}" "${rps12}" "${p99_12}" \
        "${rps24}" "${p99_24}" "${rps32}" "${p99_32}" "${fails}" \
        >> "${MAP_FILE}"

    echo "[sweep] ${label}  rps12=${rps12}  p99_12=${p99_12}  rps24=${rps24}  p99_24=${p99_24}  rps32=${rps32}  p99_32=${p99_32}  fails=${fails}"
}

# Valid pairs: draft >= steps + 1 (SGLang's NEXTN invariant).
run_cell 3 4   # baseline (current)
run_cell 3 5
run_cell 3 6
run_cell 4 5
run_cell 4 6
run_cell 4 7
run_cell 5 6
run_cell 5 7
run_cell 5 8

echo
echo "================================================================"
echo "[sweep] DONE — summary in ${MAP_FILE}"
echo "================================================================"
cat "${MAP_FILE}"
