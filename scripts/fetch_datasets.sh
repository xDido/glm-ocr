#!/usr/bin/env bash
# Pull OCR benchmark datasets into ./datasets.
#
# Usage:
#   scripts/fetch_datasets.sh funsd           # small, fast (~25 MB)
#   scripts/fetch_datasets.sh omnidocbench    # the benchmark used in the
#                                              GLM-OCR paper (~1.5 GB)
#   scripts/fetch_datasets.sh docvqa          # mid-size Q&A on docs
#   scripts/fetch_datasets.sh all
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/datasets"
mkdir -p "${DEST}"

fetch_funsd() {
    local out="${DEST}/funsd"
    if [[ -d "${out}" ]]; then echo "[funsd] already present at ${out}"; return; fi
    mkdir -p "${out}"
    echo "[funsd] downloading FUNSD (small forms dataset)..."
    # Official host. If it 404s, grab the HF mirror: nielsr/funsd
    if curl -fsSL -o "${out}/dataset.zip" \
        "https://guillaumejaume.github.io/FUNSD/dataset.zip"; then
        (cd "${out}" && python -m zipfile -e dataset.zip . && rm dataset.zip)
    else
        echo "[funsd] primary host failed, falling back to HF mirror..."
        require_hf
        huggingface-cli download nielsr/funsd \
            --repo-type dataset --local-dir "${out}"
    fi
    echo "[funsd] -> ${out}"
}

fetch_omnidocbench() {
    local out="${DEST}/OmniDocBench"
    if [[ -d "${out}" && -n "$(ls -A "${out}" 2>/dev/null)" ]]; then
        echo "[omnidocbench] already present at ${out}"; return
    fi
    require_hf
    echo "[omnidocbench] pulling opendatalab/OmniDocBench (~1.5 GB)..."
    huggingface-cli download opendatalab/OmniDocBench \
        --repo-type dataset --local-dir "${out}"
    echo "[omnidocbench] -> ${out}"
    echo "  this is the benchmark GLM-OCR reports on; apples-to-apples"
}

fetch_docvqa() {
    local out="${DEST}/docvqa"
    if [[ -d "${out}" ]]; then echo "[docvqa] already present at ${out}"; return; fi
    require_hf
    echo "[docvqa] pulling nielsr/docvqa_1200_examples (compact subset)..."
    huggingface-cli download nielsr/docvqa_1200_examples \
        --repo-type dataset --local-dir "${out}"
    echo "[docvqa] -> ${out}"
}

require_hf() {
    if ! command -v huggingface-cli >/dev/null 2>&1; then
        echo "huggingface-cli not found. Install with:" >&2
        echo "  pip install 'huggingface_hub[cli]'" >&2
        exit 1
    fi
}

case "${1:-funsd}" in
    funsd)         fetch_funsd ;;
    omnidocbench)  fetch_omnidocbench ;;
    docvqa)        fetch_docvqa ;;
    all)           fetch_funsd; fetch_docvqa; fetch_omnidocbench ;;
    *) echo "unknown dataset: ${1}"; exit 2 ;;
esac

echo
echo "[datasets] current contents of ${DEST}:"
ls -la "${DEST}"
