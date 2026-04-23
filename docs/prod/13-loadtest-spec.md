# 13 — Load-test harness + scripts spec (build from scratch)

**Purpose:** enough detail to rebuild `scripts/` and `loadtest/` from scratch. Covers the matrix harness, the bench driver, the smoke test, and the weights baker.

**Ship strategy (same as CPU container):**
- **MVP (prompt 08):** bench driver + matrix script + smoke curl. Writes Markdown reports per run. No phase decomposition, no Prometheus annotation pushing, no sweep reports. Just rps/p50/p95/p99 per concurrency cell — enough to gate a deploy.
- **Full parity (future):** port `augment_matrix_report.py` (retroactive phase decomposition from Prometheus) + `lib/render_report.py sweep` (unified multi-config reports) from the dev repo when you start doing per-change regression gates.

---

## File inventory — what you'll create

```
loadtest/
└── asyncio/
    └── bench.py                 # async HTTP load driver

scripts/
├── smoke.sh                     # one-request curl smoke test
├── bake-weights.sh              # HF download → S3 tar+upload
├── omnidoc_asyncio_matrix.sh    # multi-c sweep orchestrator
├── smoke_test.png               # tiny sample image (commit a small PNG)
└── lib/
    └── loadtest_common.sh       # shared helpers (run_id, log, etc.)
```

No `augment_matrix_report.py`, no `render_report.py`, no OmniDocBench integration for MVP. The matrix uses a small test-image pool you supply via `IMAGE_URLS_FILE` — typically 5–20 images is enough to get stable numbers.

---

## `loadtest/asyncio/bench.py`

~200 lines of Python. Uses `aiohttp` + asyncio. Key contract:

```
Usage:
  python bench.py --host <url> --concurrency <c> --total <n> \
      --image-list-file <path> --json-out <path> \
      [--warmup <n>] [--interval-seconds <s>] [--run-id <str>]

Emits:
  - JSON summary at --json-out with: {
      ok, fail, wall_seconds, rps, latency_ms: {p50, p90, p95, p99, mean, min, max},
      error_samples: [...first 5 failure reasons...]
    }
  - stdout: same summary pretty-printed.
```

### Minimal implementation

```python
"""Async HTTP load driver for glmocr.

Reads a list of image URLs (or base64 data URLs) from --image-list-file,
POSTs to <host>/glmocr/parse at the requested concurrency, collects per-
request latencies, and writes a JSON summary.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from statistics import mean

import aiohttp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True, help="e.g. http://localhost:5002")
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--total", type=int, required=True)
    p.add_argument("--image-list-file", required=True)
    p.add_argument("--json-out", required=True)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--interval-seconds", type=float, default=0.0,
                   help="seconds between request starts (0 = back-to-back)")
    p.add_argument("--pool-seed", type=int, default=42)
    p.add_argument("--run-id", default="local")
    return p.parse_args()


async def one_request(session, url, body):
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=body,
                                timeout=aiohttp.ClientTimeout(total=120)) as r:
            await r.read()
            ok = 200 <= r.status < 300
            err = None if ok else f"status={r.status}"
    except Exception as e:
        ok, err = False, f"{type(e).__name__}('{e}')"
    return (time.perf_counter() - t0) * 1000, ok, err


def _percentile(sorted_values, pct):
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100)
    lo, hi = int(k), min(int(k) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


async def run(args):
    images = [line.strip() for line in Path(args.image_list_file).read_text().splitlines() if line.strip()]
    rng = random.Random(args.pool_seed)
    url = f"{args.host.rstrip('/')}/glmocr/parse"

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=args.concurrency * 2),
    ) as session:

        # Warmup (not counted)
        for _ in range(args.warmup):
            img = rng.choice(images)
            await one_request(session, url, {"images": [img]})

        # Main run — semaphore-gated concurrency
        sem = asyncio.Semaphore(args.concurrency)
        latencies = []
        errors = []

        async def task(i):
            img = rng.choice(images)
            if args.interval_seconds > 0:
                await asyncio.sleep(i * args.interval_seconds)
            async with sem:
                lat, ok, err = await one_request(session, url, {"images": [img]})
                latencies.append((lat, ok))
                if not ok and err and len(errors) < 5:
                    errors.append(err)

        t0 = time.perf_counter()
        await asyncio.gather(*(task(i) for i in range(args.total)))
        wall = time.perf_counter() - t0

    ok_lats = sorted(l for l, ok in latencies if ok)
    n_ok = len(ok_lats)
    n_fail = args.total - n_ok

    summary = {
        "host": args.host,
        "concurrency": args.concurrency,
        "total": args.total,
        "ok": n_ok,
        "fail": n_fail,
        "wall_seconds": round(wall, 3),
        "rps": round(n_ok / wall, 3) if wall else 0.0,
        "latency_ms": {
            "mean": round(mean(ok_lats), 1) if ok_lats else 0,
            "min":  round(ok_lats[0], 1) if ok_lats else 0,
            "p50":  round(_percentile(ok_lats, 50), 1),
            "p90":  round(_percentile(ok_lats, 90), 1),
            "p95":  round(_percentile(ok_lats, 95), 1),
            "p99":  round(_percentile(ok_lats, 99), 1),
            "max":  round(ok_lats[-1], 1) if ok_lats else 0,
        },
        "error_samples": errors,
        "run_id": args.run_id,
    }
    Path(args.json_out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
```

**The images file** is a text file, one image URL per line. Each line can be:
- A public HTTP URL to a PDF/PNG/JPEG
- A `file:///app/...` URL (inside the container) pointing at a mounted image
- A `data:image/png;base64,...` inline URL (convenient for CI — avoids mounting volumes)

For CI, generate a small base64 file at harness start:

```bash
{ for f in scripts/test_images/*.png; do
    printf 'data:image/png;base64,%s\n' "$(base64 -w0 "$f")"
  done
} > "${URLS_FILE}"
```

---

## `scripts/omnidoc_asyncio_matrix.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${CPU_URL:?CPU_URL required, e.g. http://localhost:5002}"
: "${MATRIX_TOTAL:=100}"

TS="$(date +%Y%m%d-%H%M%S)"
RUN_ID="matrix-${TS}"
OUT_DIR="loadtest/results"
mkdir -p "${OUT_DIR}"

# Image pool — build from scripts/test_images/*.png as base64 data URLs.
URLS_FILE="$(mktemp)"
trap 'rm -f "${URLS_FILE}"' EXIT INT TERM
IMAGES_DIR="${IMAGES_DIR:-scripts/test_images}"
if [[ ! -d "${IMAGES_DIR}" ]]; then
    echo "no ${IMAGES_DIR}/ — add a few small PNGs to seed the pool" >&2
    exit 1
fi
for f in "${IMAGES_DIR}"/*.png "${IMAGES_DIR}"/*.jpg; do
    [[ -f "$f" ]] || continue
    printf 'data:image/png;base64,%s\n' "$(base64 -w0 "$f")"
done > "${URLS_FILE}"
POOL_SIZE=$(wc -l < "${URLS_FILE}")
echo "[matrix] pool_size=${POOL_SIZE}"

REPORT="${OUT_DIR}/${RUN_ID}.md"
{
    echo "# Matrix report — ${RUN_ID}"
    echo ""
    echo "- host: ${CPU_URL}"
    echo "- total per cell: ${MATRIX_TOTAL}"
    echo "- pool size: ${POOL_SIZE}"
    echo ""
    echo "| c | ok | fail | wall_s | rps | p50 | p90 | p95 | p99 | mean | max |"
    echo "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
} > "${REPORT}"

run_cell() {
    local c="$1"
    local json="${OUT_DIR}/${RUN_ID}-c${c}.json"
    echo "[matrix] running c=${c}"
    python loadtest/asyncio/bench.py \
        --host "${CPU_URL}" \
        --concurrency "${c}" \
        --total "${MATRIX_TOTAL}" \
        --image-list-file "${URLS_FILE}" \
        --json-out "${json}" \
        --warmup 2 \
        --run-id "${RUN_ID}-c${c}" \
        || echo "[matrix] c=${c} had failures"

    python -c "
import json, sys
d = json.load(open('${json}'))
l = d['latency_ms']
print(f\"| {d['concurrency']} | {d['ok']} | {d['fail']} | {d['wall_seconds']} | {d['rps']} | {l['p50']} | {l['p90']} | {l['p95']} | {l['p99']} | {l['mean']} | {l['max']} |\")
" >> "${REPORT}"

    # Inter-trial gap — matches dev's 25s gap used by the augmenter. Even
    # without the augmenter, it makes Grafana / CloudWatch charts clearly
    # segmented per cell.
    sleep 25
}

for c in 12 24 32 40 64; do
    run_cell "$c"
done

echo "[matrix] report -> ${REPORT}"
```

---

## `scripts/smoke.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

ALB="${1:?alb url required, e.g. http://glmocr-alb-....elb.amazonaws.com:5002}"
IMG="${2:-scripts/test_images/sample.png}"

if [[ ! -f "$IMG" ]]; then
    echo "image $IMG not found" >&2
    exit 1
fi

# Health
echo "[smoke] GET /health"
curl -fsS "${ALB}/health" ; echo

# OCR parse
echo "[smoke] POST /glmocr/parse"
B64="$(base64 -w0 "$IMG")"
BODY="$(python3 -c "
import json, sys
b = sys.argv[1]
print(json.dumps({'images': [f'data:image/png;base64,{b}']}))
" "$B64")"

RESP="$(curl -fsS -X POST "${ALB}/glmocr/parse" \
    -H 'content-type: application/json' \
    -d "$BODY")"

echo "$RESP" | python3 -c "
import json, sys
d = json.load(sys.stdin)
md = d.get('markdown_result') or ''
jr = d.get('json_result')
print(f'markdown_len={len(md)}')
print(f'json_type={type(jr).__name__}')
assert len(md) > 0 or jr is not None, 'empty response — smoke failed'
print('[smoke] OK')
"
```

---

## `scripts/bake-weights.sh`

```bash
#!/usr/bin/env bash
# Download GLM-OCR from HuggingFace Hub, tar it, upload to S3 for SageMaker.
# Run on your laptop (requires HF_TOKEN + AWS creds).
set -euo pipefail

MODEL_ID="${MODEL_ID:-zai-org/GLM-OCR}"
BUCKET="${WEIGHTS_BUCKET:?set WEIGHTS_BUCKET, e.g. glmocr-prod-weights-<acct>}"
KEY="${KEY:-glm-ocr/$(date +%Y%m%d)/model.tar.gz}"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

export HF_HOME="${WORK}/hf"
pip install --upgrade huggingface_hub > /dev/null

echo "[bake] downloading ${MODEL_ID}"
huggingface-cli download "${MODEL_ID}" \
    --local-dir "${WORK}/model" \
    --token "${HF_TOKEN:-}"

# Tar from INSIDE the model dir so the archive root is the model contents
# (SageMaker extracts to /opt/ml/model; tar with a nested folder breaks this).
echo "[bake] tarring"
tar -C "${WORK}/model" -czvf "${WORK}/model.tar.gz" .

echo "[bake] uploading to s3://${BUCKET}/${KEY}"
aws s3 cp "${WORK}/model.tar.gz" "s3://${BUCKET}/${KEY}" \
    --sse AES256 \
    --storage-class STANDARD

echo "[bake] done: s3://${BUCKET}/${KEY}"
echo ""
echo "Next: update the SSM param"
echo "  aws ssm put-parameter --overwrite \\"
echo "    --name /glmocr/prod/sagemaker/model_data_key \\"
echo "    --value '${KEY}'"
```

---

## `scripts/lib/loadtest_common.sh` (minimal)

```bash
#!/usr/bin/env bash
# Shared helpers for loadtest scripts. Keep minimal; the dev repo's version
# has OmniDocBench integration + Prometheus annotation pushes — both future
# work.

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { log "WARN: $*" >&2; }
die() { warn "FATAL: $*"; exit 1; }
```

---

## What's intentionally missing vs dev

| Dev feature | Status in MVP | Port later? |
|---|---|---|
| OmniDocBench pool (1000+ real PDFs) | Out — use small PNG pool | Optional; add if you need per-doc-class regression testing |
| `augment_matrix_report.py` (Prometheus phase decomposition) | Out | Yes — worth porting once metrics flow is stable |
| `render_report.py sweep` (multi-config unified reports) | Out | Only needed if you do multi-knob sweeps in prod |
| Pushgateway annotations | Out | Replace with direct Grafana Cloud `/api/annotations` calls in the matrix script |
| k6 / locust drivers | Out — asyncio bench is enough | No; asyncio is the only one we actually use |
| Per-trial Grafana annotations | Out | Easy addition; ~10 lines of curl in the matrix script |

The Markdown report produced by the MVP matrix script is simpler than the dev version but has the same essentials: rps, failure count, latency percentiles. Enough to tell a CI job "pass" vs "fail."

---

## CI integration

In `prompts/08-first-smoke.md` the workflow runs `bash scripts/omnidoc_asyncio_matrix.sh` after deploy. Add a pass-gate:

```bash
# In cdk-deploy.yml, after the matrix step:
python3 -c "
import glob, json, sys
fails = 0
for f in glob.glob('loadtest/results/matrix-*-c12.json') + glob.glob('loadtest/results/matrix-*-c24.json'):
    d = json.load(open(f))
    if d['fail'] > 0:
        print(f'{f}: {d[\"fail\"]} failures')
        fails += 1
sys.exit(1 if fails else 0)
"
```

Gates on "no failures at c=12 and c=24" — the dev tuned baseline. Sub-baseline rps is NOT a CI gate; single-run noise (±25%) would flap it. rps regressions surface on the dashboard.

---

Next: back to [`prompts/01-bootstrap-cdk.md`](./prompts/01-bootstrap-cdk.md) if starting implementation, or [`09-runbook.md`](./09-runbook.md) if ops.
