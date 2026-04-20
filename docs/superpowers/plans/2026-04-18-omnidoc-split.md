# OmniDoc Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `scripts/loadtest_omnidoc.sh` with three standalone per-driver scripts (`omnidoc_asyncio.sh`, `omnidoc_locust.sh`, `omnidoc_k6.sh`) sharing a bash helper, and purge the `loadtest/samples/` fixture pool from the codebase.

**Architecture:** Each standalone script sources `scripts/lib/loadtest_common.sh` for preflight, image-pool sampling, Grafana annotations, and logging. Drivers (bench.py / locustfile.py / ocr_load.js) lose their hardcoded `file:///app/samples/*` fallbacks and now error out if run without an image pool. The smoke-test fixture (previously bundled in `loadtest/samples/`) relocates to `scripts/smoke_test.png` and is bind-mounted as a single file.

**Tech Stack:** bash, docker-compose, GNU make, aiohttp/asyncio (Python), locust (Python), k6 (JS).

**Prerequisites for running this plan:** OmniDocBench must be downloaded (`make datasets-omnidocbench`) on the target host before Tasks 3-5 can be verified. `make up` must be runnable (GPU + CPU containers come up).

---

## File Map

### Create

- `scripts/lib/loadtest_common.sh` — shared bash helpers (env defaults, logging, preflight, pool builder, Grafana annotate, `init_run`).
- `scripts/omnidoc_asyncio.sh` — standalone asyncio runner.
- `scripts/omnidoc_locust.sh` — standalone locust runner.
- `scripts/omnidoc_k6.sh` — standalone k6 runner.
- `scripts/smoke_test.png` — relocated smoke-test fixture (from `loadtest/samples/receipt.png`).

### Delete

- `scripts/loadtest_omnidoc.sh`
- `loadtest/samples/` (entire directory).

### Modify

- `docker-compose.yml` — swap the samples-dir bind mount for a single-file smoke-fixture bind mount.
- `scripts/smoke_test.sh` — update default body URI.
- `loadtest/asyncio/bench.py` — remove fallback pool, fail clearly when no pool provided; update docstring example.
- `loadtest/k6/ocr_load.js` — remove hardcoded `IMAGES` fallback.
- `loadtest/locust/locustfile.py` — remove hardcoded `LOCUST_IMAGES` fallback; update docstring.
- `Makefile` — delete `load-*` and `omnidoc-load` targets; add `omnidoc-asyncio`/`omnidoc-locust`/`omnidoc-k6`; update help + `.PHONY`.
- `.env.example` — delete `LOCUST_IMAGES` and `DRIVERS` lines.
- `.gitignore` — delete the three `loadtest/samples/*` lines.
- `README.md` — rewrite "Load tests" section; update "Files" bullet.

---

## Task 1: Relocate the smoke-test fixture and swap bind mount

Front-loaded because it de-risks the known Windows single-file-bind-mount concern flagged in the spec.

**Files:**
- Move: `loadtest/samples/receipt.png` → `scripts/smoke_test.png`
- Modify: `docker-compose.yml` (the CPU service mount list)
- Modify: `scripts/smoke_test.sh` (the `IMAGES_JSON` default on line 7)

- [ ] **Step 1: Move the fixture with `git mv`**

```bash
git mv loadtest/samples/receipt.png scripts/smoke_test.png
```

- [ ] **Step 2: Swap the CPU-container bind mount in `docker-compose.yml`**

Find the line (currently `docker-compose.yml:93`):

```yaml
      - ./loadtest/samples:/app/samples:ro
```

Replace with:

```yaml
      - ./scripts/smoke_test.png:/app/smoke_test.png:ro
```

- [ ] **Step 3: Update `scripts/smoke_test.sh` default body**

Find line 7:

```bash
IMAGES_JSON="${IMAGES_JSON:-{\"images\":[\"file:///app/samples/receipt.png\"]}}"
```

Replace with:

```bash
IMAGES_JSON="${IMAGES_JSON:-{\"images\":[\"file:///app/smoke_test.png\"]}}"
```

- [ ] **Step 4: Restart CPU container so the new mount is picked up**

```bash
docker compose up -d --force-recreate cpu
```

Expected: `cpu` container recreates cleanly (no mount errors).

- [ ] **Step 5: Verify smoke still passes**

```bash
make smoke
```

Expected output contains `[smoke] OK` and `"markdown_result"` in the response body.

If this step fails with a mount-type error on Windows (the risk flagged in the spec), fall back to a fixtures directory: `mkdir scripts/fixtures && git mv scripts/smoke_test.png scripts/fixtures/smoke_test.png`, set the compose mount to `./scripts/fixtures:/app/fixtures:ro`, and update `smoke_test.sh` to use `file:///app/fixtures/smoke_test.png`. Re-run `make smoke`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "relocate smoke fixture out of loadtest/samples; single-file bind mount"
```

---

## Task 2: Create shared helper `scripts/lib/loadtest_common.sh`

**Files:**
- Create: `scripts/lib/loadtest_common.sh`

- [ ] **Step 1: Create the directory**

```bash
mkdir -p scripts/lib
```

- [ ] **Step 2: Write `scripts/lib/loadtest_common.sh`**

Full content:

```bash
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
```

- [ ] **Step 3: Verify bash parses the helper**

```bash
bash -n scripts/lib/loadtest_common.sh && echo "parse OK"
```

Expected: `parse OK`.

- [ ] **Step 4: Verify `init_run` sets the expected vars**

```bash
bash -c 'source scripts/lib/loadtest_common.sh && init_run test && echo "TAG=${DRIVER_TAG} TS=${TS} PREFIX=${RUN_PREFIX}"'
```

Expected output resembles:
```
TAG=test TS=20260418-HHMMSS PREFIX=loadtest/results/omnidoc-20260418-HHMMSS-test
```

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/loadtest_common.sh
git commit -m "add shared loadtest helper for per-driver omnidoc scripts"
```

---

## Task 3: Create `scripts/omnidoc_asyncio.sh`

**Files:**
- Create: `scripts/omnidoc_asyncio.sh`

- [ ] **Step 1: Write the script**

Full content:

```bash
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
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x scripts/omnidoc_asyncio.sh
```

- [ ] **Step 3: Parse-check**

```bash
bash -n scripts/omnidoc_asyncio.sh && echo "parse OK"
```

Expected: `parse OK`.

- [ ] **Step 4: Smoke-run the script with a tiny pool + short duration**

Preconditions: `make up` is running; `datasets/OmniDocBench/` exists.

```bash
OMNIDOC_SAMPLE_POOL=4 OMNIDOC_DURATION=20 scripts/omnidoc_asyncio.sh
```

Expected: script completes (~25s including warmup). Log lines include `[omnidoc-asyncio]` prefix. A results file `loadtest/results/omnidoc-<ts>-asyncio.json` exists and parses as JSON. Final summary line starts with `  asyncio :`.

- [ ] **Step 5: Commit**

```bash
git add scripts/omnidoc_asyncio.sh
git commit -m "add standalone scripts/omnidoc_asyncio.sh"
```

---

## Task 4: Create `scripts/omnidoc_locust.sh`

**Files:**
- Create: `scripts/omnidoc_locust.sh`

- [ ] **Step 1: Write the script**

Full content:

```bash
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
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x scripts/omnidoc_locust.sh
```

- [ ] **Step 3: Parse-check**

```bash
bash -n scripts/omnidoc_locust.sh && echo "parse OK"
```

Expected: `parse OK`.

- [ ] **Step 4: Smoke-run with a tiny pool + short duration**

Preconditions: `make up` is running; `datasets/OmniDocBench/` exists; `locust` is on PATH.

```bash
OMNIDOC_SAMPLE_POOL=4 OMNIDOC_DURATION=20 scripts/omnidoc_locust.sh
```

Expected: script completes. A CSV `loadtest/results/omnidoc-<ts>-locust_stats.csv` exists with a header row and at least one data row. Final summary line starts with `  locust  :`.

- [ ] **Step 5: Commit**

```bash
git add scripts/omnidoc_locust.sh
git commit -m "add standalone scripts/omnidoc_locust.sh"
```

---

## Task 5: Create `scripts/omnidoc_k6.sh`

**Files:**
- Create: `scripts/omnidoc_k6.sh`

- [ ] **Step 1: Write the script**

Full content:

```bash
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
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x scripts/omnidoc_k6.sh
```

- [ ] **Step 3: Parse-check**

```bash
bash -n scripts/omnidoc_k6.sh && echo "parse OK"
```

Expected: `parse OK`.

- [ ] **Step 4: Smoke-run with a tiny pool + short duration**

Preconditions: `make up` is running; `datasets/OmniDocBench/` exists; `k6` is on PATH.

```bash
OMNIDOC_SAMPLE_POOL=4 OMNIDOC_DURATION=20 scripts/omnidoc_k6.sh
```

Expected: script completes (~40s including the 10s ramp-up + 10s ramp-down inside the k6 scenario). A JSON `loadtest/results/omnidoc-<ts>-k6.json` exists and parses. Final summary line starts with `  k6      :`.

- [ ] **Step 5: Commit**

```bash
git add scripts/omnidoc_k6.sh
git commit -m "add standalone scripts/omnidoc_k6.sh"
```

---

## Task 6: Remove hardcoded sample-URL fallbacks from the three drivers

**Files:**
- Modify: `loadtest/asyncio/bench.py` (lines 8-10 docstring; lines 91-99 `_resolve_image_pool`)
- Modify: `loadtest/k6/ocr_load.js` (lines 13-18 `HOST`/`IMAGES`)
- Modify: `loadtest/locust/locustfile.py` (line 8 docstring; lines 18-25 `_images()`)

- [ ] **Step 1: Update `loadtest/asyncio/bench.py` docstring example**

Find lines 8-10:

```python
Examples:
    python bench.py --host http://localhost:5002 --concurrency 16 --total 128
    python bench.py --host http://localhost:5002 --image-url file:///app/samples/receipt.png
```

Replace with:

```python
Examples:
    scripts/omnidoc_asyncio.sh        # full OmniDocBench run
    python bench.py --host http://localhost:5002 --concurrency 16 --total 128 \
        --image-url file:///app/datasets/OmniDocBench/images/<name>.jpg
```

- [ ] **Step 2: Replace the fallback in `_resolve_image_pool`**

Find lines 91-99:

```python
def _resolve_image_pool(args: argparse.Namespace) -> list[str]:
    pool: list[str] = list(args.image_url or [])
    if args.image_list_file:
        with args.image_list_file.open("r", encoding="utf-8") as fh:
            pool.extend(line.strip() for line in fh if line.strip()
                        and not line.lstrip().startswith("#"))
    if not pool:
        pool = ["file:///app/samples/receipt.png"]
    return pool
```

Replace with:

```python
def _resolve_image_pool(args: argparse.Namespace) -> list[str]:
    pool: list[str] = list(args.image_url or [])
    if args.image_list_file:
        with args.image_list_file.open("r", encoding="utf-8") as fh:
            pool.extend(line.strip() for line in fh if line.strip()
                        and not line.lstrip().startswith("#"))
    if not pool:
        sys.stderr.write(
            "bench.py: no image pool — pass --image-url or --image-list-file, "
            "or run via scripts/omnidoc_asyncio.sh\n"
        )
        sys.exit(2)
    return pool
```

- [ ] **Step 3: Verify the asyncio driver now exits cleanly when called bare**

```bash
python loadtest/asyncio/bench.py --host http://localhost:5002 --total 1
echo "exit=$?"
```

Expected: `bench.py: no image pool — ...` on stderr; `exit=2`.

- [ ] **Step 4: Update `loadtest/k6/ocr_load.js`**

Find lines 13-18:

```javascript
const HOST = __ENV.HOST || 'http://localhost:5002';
const IMAGES = (__ENV.IMAGES ||
  'file:///app/samples/receipt.png,' +
  'file:///app/samples/table.png,' +
  'file:///app/samples/invoice.pdf'
).split(',').map(s => s.trim()).filter(Boolean);
```

Replace with:

```javascript
const HOST = __ENV.HOST || 'http://localhost:5002';

if (!__ENV.IMAGES) {
  throw new Error(
    'ocr_load.js: IMAGES env var is required. Run via scripts/omnidoc_k6.sh, ' +
    'or set IMAGES=<comma-separated-urls> yourself.'
  );
}
const IMAGES = __ENV.IMAGES.split(',').map(s => s.trim()).filter(Boolean);
```

- [ ] **Step 5: Verify the k6 script errors cleanly when called without IMAGES**

```bash
k6 run loadtest/k6/ocr_load.js -e HOST=http://localhost:5002 || true
```

Expected: k6 prints an error mentioning `IMAGES env var is required`.

- [ ] **Step 6: Update `loadtest/locust/locustfile.py` docstring**

Find line 8:

```python
    LOCUST_IMAGES  - comma-separated image URLs (defaults to in-container samples)
```

Replace with:

```python
    LOCUST_IMAGES  - comma-separated image URLs (required; set by scripts/omnidoc_locust.sh)
```

- [ ] **Step 7: Replace the fallback in `_images()`**

Find lines 18-25:

```python
def _images() -> list[str]:
    raw = os.environ.get(
        "LOCUST_IMAGES",
        "file:///app/samples/receipt.png,"
        "file:///app/samples/table.png,"
        "file:///app/samples/invoice.pdf",
    )
    return [u.strip() for u in raw.split(",") if u.strip()]
```

Replace with:

```python
def _images() -> list[str]:
    raw = os.environ.get("LOCUST_IMAGES")
    if not raw:
        raise RuntimeError(
            "locustfile.py: LOCUST_IMAGES env var is required. Run via "
            "scripts/omnidoc_locust.sh, or set LOCUST_IMAGES=<csv> yourself."
        )
    return [u.strip() for u in raw.split(",") if u.strip()]
```

- [ ] **Step 8: Verify the locust entry point errors when called without LOCUST_IMAGES**

```bash
cd loadtest/locust && python -c "import locustfile" ; cd -
```

Expected: traceback ends with `RuntimeError: locustfile.py: LOCUST_IMAGES env var is required. ...`.

- [ ] **Step 9: Re-verify all three standalone scripts still succeed end-to-end**

```bash
OMNIDOC_SAMPLE_POOL=4 OMNIDOC_DURATION=20 scripts/omnidoc_asyncio.sh
OMNIDOC_SAMPLE_POOL=4 OMNIDOC_DURATION=20 scripts/omnidoc_locust.sh
OMNIDOC_SAMPLE_POOL=4 OMNIDOC_DURATION=20 scripts/omnidoc_k6.sh
```

Expected: each completes with its per-driver summary line.

- [ ] **Step 10: Commit**

```bash
git add loadtest/asyncio/bench.py loadtest/k6/ocr_load.js loadtest/locust/locustfile.py
git commit -m "drop hardcoded sample-URL fallbacks from asyncio/k6/locust drivers"
```

---

## Task 7: Delete the orchestrator and the samples directory

Done after Task 6 so we don't break in-flight development.

**Files:**
- Delete: `scripts/loadtest_omnidoc.sh`
- Delete: `loadtest/samples/` (the whole directory, including `README.md` and any remaining files)

- [ ] **Step 1: Remove the orchestrator**

```bash
git rm scripts/loadtest_omnidoc.sh
```

- [ ] **Step 2: Remove the samples directory**

```bash
git rm -r loadtest/samples
```

- [ ] **Step 3: Verify nothing in-tree still references either path**

```bash
grep -rEn 'loadtest_omnidoc\.sh|loadtest/samples' \
     --exclude-dir=.git --exclude-dir=node_modules \
     --exclude-dir=.claude --exclude-dir=docs \
     . && echo "STRAY REF FOUND" || echo "clean"
```

Expected: `clean`. (The `--exclude-dir=docs` is because `docs/superpowers/specs/` and `docs/superpowers/plans/` legitimately reference the path as history.)

- [ ] **Step 4: Commit**

```bash
git commit -m "remove loadtest_omnidoc.sh orchestrator and loadtest/samples fixture pool"
```

---

## Task 8: Rewrite Makefile targets

**Files:**
- Modify: `Makefile` (`.PHONY` list; help block; target recipes)

- [ ] **Step 1: Update the `.PHONY` line**

Find:

```makefile
.PHONY: help up down build logs ps smoke runtime runtime-full \
        load-asyncio load-locust load-k6 \
        omnidoc-load obs-open \
        datasets datasets-funsd datasets-omnidocbench \
        register-tasks \
        clean
```

Replace with:

```makefile
.PHONY: help up down build logs ps smoke runtime runtime-full \
        omnidoc-asyncio omnidoc-locust omnidoc-k6 \
        obs-open \
        datasets datasets-funsd datasets-omnidocbench \
        register-tasks \
        clean
```

- [ ] **Step 2: Update the help block**

Find:

```makefile
	@echo "  load-asyncio      - run aiohttp bench script"
	@echo "  load-locust       - run locust headless for 2 minutes"
	@echo "  load-k6           - run k6 scenario"
	@echo "  omnidoc-load      - sequential all-driver load test vs OmniDocBench"
	@echo "  obs-open          - print Prometheus + Grafana URLs"
```

Replace with:

```makefile
	@echo "  omnidoc-asyncio   - OmniDocBench asyncio load test (sampled pool)"
	@echo "  omnidoc-locust    - OmniDocBench locust load test (sampled pool)"
	@echo "  omnidoc-k6        - OmniDocBench k6 load test (sampled pool)"
	@echo "  obs-open          - print Prometheus + Grafana URLs"
```

- [ ] **Step 3: Remove the old target recipes**

Find and delete the blocks:

```makefile
load-asyncio:
	python loadtest/asyncio/bench.py --host $(CPU_URL) --concurrency 16 --total 128

load-locust:
	cd loadtest/locust && \
	  locust -f locustfile.py --headless -u 50 -r 5 -t 2m \
	    --host $(CPU_URL) --csv ../results/locust

load-k6:
	k6 run loadtest/k6/ocr_load.js \
	  -e HOST=$(CPU_URL) \
	  --summary-export=loadtest/results/k6.json
```

and:

```makefile
omnidoc-load:
	CPU_URL=$(CPU_URL) "C:/Program Files/Git/bin/bash.exe" scripts/loadtest_omnidoc.sh
```

- [ ] **Step 4: Add the three new target recipes**

Insert (anywhere after `obs-open:` and before `register-tasks:`):

```makefile
omnidoc-asyncio:
	CPU_URL=$(CPU_URL) "$(SHELL)" scripts/omnidoc_asyncio.sh

omnidoc-locust:
	CPU_URL=$(CPU_URL) "$(SHELL)" scripts/omnidoc_locust.sh

omnidoc-k6:
	CPU_URL=$(CPU_URL) "$(SHELL)" scripts/omnidoc_k6.sh
```

- [ ] **Step 5: Sanity-check the Makefile parses and the new targets are visible**

```bash
make -n omnidoc-asyncio && make -n omnidoc-locust && make -n omnidoc-k6
```

Expected: each prints the corresponding `CPU_URL=... "$(SHELL)" scripts/omnidoc_*.sh` command without errors.

```bash
make help | grep -E 'load-|omnidoc-'
```

Expected output lists only `omnidoc-asyncio`, `omnidoc-locust`, `omnidoc-k6` — no `load-*`, no `omnidoc-load`.

- [ ] **Step 6: Verify a Makefile-invoked run still succeeds end-to-end**

```bash
OMNIDOC_SAMPLE_POOL=4 OMNIDOC_DURATION=20 make omnidoc-asyncio
```

Expected: the same successful run as in Task 3 Step 4, this time via `make`.

- [ ] **Step 7: Commit**

```bash
git add Makefile
git commit -m "swap load-* and omnidoc-load targets for omnidoc-{asyncio,locust,k6}"
```

---

## Task 9: Clean up `.env.example` and `.gitignore`

**Files:**
- Modify: `.env.example`
- Modify: `.gitignore`

- [ ] **Step 1: Remove `LOCUST_IMAGES` from `.env.example`**

Find (around line 162):

```
# Standalone Locust image pool. Used when you invoke `make load-locust`
# directly (not via `omnidoc-load`). The orchestrator overrides this at
# run time with the OmniDocBench sample.
LOCUST_IMAGES=file:///app/samples/receipt.png,file:///app/samples/table.png,file:///app/samples/invoice.pdf
```

Delete the whole block (4 lines — the comment paragraph and the `LOCUST_IMAGES=` line).

- [ ] **Step 2: Remove `DRIVERS` from `.env.example`**

Find (around line 156):

```
# Subset of drivers to run inside `make omnidoc-load`. Space-separated.
# Useful when you want to iterate on a single driver without paying the
# full ~6 min sequential run. Valid names: asyncio locust k6
DRIVERS=asyncio locust k6
```

Delete the whole block (4 lines).

- [ ] **Step 3: Remove the samples entries from `.gitignore`**

Find (lines 8-10):

```
loadtest/samples/*
!loadtest/samples/.gitkeep
!loadtest/samples/README.md
```

Delete all three lines.

- [ ] **Step 4: Verify grep finds no more stale references**

```bash
grep -En 'LOCUST_IMAGES|loadtest/samples|DRIVERS=' .env.example .gitignore || echo "clean"
```

Expected: `clean`.

- [ ] **Step 5: Commit**

```bash
git add .env.example .gitignore
git commit -m "drop LOCUST_IMAGES, DRIVERS, and loadtest/samples entries"
```

---

## Task 10: Rewrite the README "Load tests" section

**Files:**
- Modify: `README.md` (the "Load tests" section around lines 130-140 and the Files bullet around line 175)

- [ ] **Step 1: Replace the "Load tests" section**

Find:

````markdown
## Load tests

```bash
make load-asyncio     # python loadtest/asyncio/bench.py --concurrency 16 --total 128
make load-locust      # locust headless, 50 users, 5/s ramp, 2 minutes
make load-k6          # k6 ramping VUs with p95 threshold
```

Results land in `loadtest/results/`. The asyncio script prints p50/p95/p99
directly. Sweep `OCR_MAX_WORKERS × SGL_MAX_RUNNING_REQUESTS` to build the
sizing table for ECS.
````

Replace with:

````markdown
## Load tests

All load tests sample from OmniDocBench. Run `make datasets-omnidocbench`
once before the first run.

```bash
make omnidoc-asyncio      # aiohttp bench; pushes summary to Pushgateway
make omnidoc-locust       # locust headless, 50 users / 5-ramp, scraped by locust_exporter
make omnidoc-k6           # k6; remote-writes metrics to Prometheus
```

Each script samples `OMNIDOC_SAMPLE_POOL` images (default 64) and runs
for `OMNIDOC_DURATION` seconds (default 120). Results land under
`loadtest/results/omnidoc-<timestamp>-<driver>.*`. Grafana annotations
mark each driver's start and end on the "GLM-OCR Load Test" dashboard.

Sweep `OCR_MAX_WORKERS × SGL_MAX_RUNNING_REQUESTS` to build the sizing
table for ECS.
````

- [ ] **Step 2: Update the `loadtest/` bullet under "Files"**

Find:

```markdown
- `loadtest/` — Locust, k6, asyncio bench; samples; results
```

Replace with:

```markdown
- `loadtest/` — Locust, k6, asyncio bench; results
```

- [ ] **Step 3: Verify no stray samples / orchestrator references remain in README**

```bash
grep -En 'samples|loadtest_omnidoc|load-asyncio|load-locust|load-k6|omnidoc-load' README.md || echo "clean"
```

Expected: `clean`.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "document the three omnidoc-* load-test targets in README"
```

---

## Spec coverage check

| Spec section | Covered by |
|---|---|
| Files deleted — orchestrator | Task 7 Step 1 |
| Files deleted — loadtest/samples/ | Task 7 Step 2 |
| Files deleted — Makefile targets `load-*` + `omnidoc-load` | Task 8 Steps 1-3 |
| Files deleted — `.gitignore` entries | Task 9 Step 3 |
| Files deleted — `.env.example` entries (DRIVERS, LOCUST_IMAGES) | Task 9 Steps 1-2 |
| Files created — `scripts/lib/loadtest_common.sh` | Task 2 |
| Files created — `scripts/omnidoc_asyncio.sh` | Task 3 |
| Files created — `scripts/omnidoc_locust.sh` | Task 4 |
| Files created — `scripts/omnidoc_k6.sh` | Task 5 |
| Files created — `scripts/smoke_test.png` | Task 1 Step 1 |
| Files modified — `docker-compose.yml` bind mount | Task 1 Step 2 |
| Files modified — `scripts/smoke_test.sh` path | Task 1 Step 3 |
| Files modified — `bench.py` fallback + docstring | Task 6 Steps 1-3 |
| Files modified — `ocr_load.js` fallback | Task 6 Steps 4-5 |
| Files modified — `locustfile.py` fallback + docstring | Task 6 Steps 6-8 |
| Files modified — `Makefile` targets | Task 8 |
| Files modified — `README.md` | Task 10 |
| Shared helper contents (env defaults, constants, log helpers, init_run, preflight_omnidoc, build_omnidoc_pool, annotate) | Task 2 Step 2 |
| Standalone script structure | Tasks 3-5 |
| Behavior parity — result-file paths | Tasks 3-5 (explicit `${RUN_PREFIX}` strings match old layout) |
| Behavior parity — outer "omnidoc run" annotations removed | Tasks 3-5 (only per-driver annotate calls) |
| Smoke-test bind-mount risk fallback | Task 1 Step 5 (documented fallback path) |
