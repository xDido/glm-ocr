# Split omnidoc load-test orchestrator into per-driver scripts

**Date:** 2026-04-18
**Status:** approved (pre-implementation)

## Goal

Replace the single `scripts/loadtest_omnidoc.sh` orchestrator with three
standalone per-driver scripts (`omnidoc_asyncio.sh`, `omnidoc_locust.sh`,
`omnidoc_k6.sh`) that share common bash helpers. Drop the "quick mode"
load-test Makefile targets and the `loadtest/samples/` fixture pool
entirely. OmniDocBench is the only image source for load tests going
forward. A single smoke-test image stays, relocated out of `loadtest/`.

## Motivation

- The orchestrator bundles four responsibilities (preflight, sample pool,
  annotations, summary) around three unrelated driver invocations. Running
  just one driver means editing `DRIVERS=` or commenting out `run_*` calls.
- Two fixture sources (`loadtest/samples/*` for quick mode, `OmniDocBench`
  for "real" runs) doubles the knob surface. The user has decided quick
  mode is not worth keeping.
- Per-driver scripts are easier to tweak and to invoke ad-hoc without
  loading the full orchestrator mental model.

## Non-goals

- No changes to driver semantics (asyncio/locust/k6 logic, ramps, or
  threshold config).
- No change to results directory layout (`loadtest/results/omnidoc-*`).
- No change to Prometheus/Pushgateway/Grafana wiring beyond what falls out
  of the split.
- No restoration of the "run all three on the same sample" property once
  the orchestrator is gone. Each standalone script samples its own pool.

## Files deleted

- `scripts/loadtest_omnidoc.sh` — the orchestrator.
- `loadtest/samples/` — whole directory, including `README.md`,
  `receipt.png`, and any `.gitkeep`.
- Makefile targets: `load-asyncio`, `load-locust`, `load-k6`, `omnidoc-load`,
  and their help lines + `.PHONY` entries.
- `.gitignore` entries covering `loadtest/samples/`.
- `.env.example` entries: `DRIVERS`, `LOCUST_IMAGES`.

## Files created

- `scripts/lib/loadtest_common.sh` — shared helper, `source`'d by each of
  the three standalone scripts.
- `scripts/omnidoc_asyncio.sh`
- `scripts/omnidoc_locust.sh`
- `scripts/omnidoc_k6.sh`
- `scripts/smoke_test.png` — single smoke-test fixture (the current
  `loadtest/samples/receipt.png` relocated).

## Files modified

- **`docker-compose.yml`** — CPU service: replace the bind mount
  `./loadtest/samples:/app/samples:ro` with the single-file mount
  `./scripts/smoke_test.png:/app/smoke_test.png:ro`.
- **`scripts/smoke_test.sh`** — default body changes from
  `file:///app/samples/receipt.png` to `file:///app/smoke_test.png`.
- **`loadtest/asyncio/bench.py`** — `_resolve_image_pool()` no longer
  falls back to `file:///app/samples/receipt.png` when the pool is empty;
  it raises `SystemExit` with a message pointing to
  `scripts/omnidoc_asyncio.sh` or the `--image-url` / `--image-list-file`
  flags. Docstring example updated to reference an OmniDocBench path.
- **`loadtest/k6/ocr_load.js`** — `IMAGES` no longer has a hardcoded
  fallback; when `__ENV.IMAGES` is unset the script `throw`s at module
  load with a clear message.
- **`loadtest/locust/locustfile.py`** — `_images()` raises
  `RuntimeError` when `LOCUST_IMAGES` is unset (no fallback list).
- **`Makefile`** — add three targets: `omnidoc-asyncio`, `omnidoc-locust`,
  `omnidoc-k6`, each invoking the matching `scripts/omnidoc_*.sh` under
  `"$(SHELL)"`. Update `.PHONY` list and the `help` block accordingly.
  Remove the four deleted targets.
- **`README.md`** — rewrite the "Load tests" section (lines ~130-140)
  around the three new `make omnidoc-*` targets. Remove any reference to
  `loadtest/samples/`, `LOCUST_IMAGES`, or the orchestrator.

## Shared helper: `scripts/lib/loadtest_common.sh`

### Exports (env defaults, `:=`)

- `CPU_URL` (default `http://localhost:5002`)
- `GRAFANA_URL` (default `http://localhost:3000`)
- `GRAFANA_USER` (default `admin`)
- `GRAFANA_ADMIN_PASSWORD` (default `admin`)
- `OMNIDOC_DURATION` (default `120`)
- `OMNIDOC_SAMPLE_POOL` (default `64`)
- `PROMETHEUS_URL` (default `http://localhost:9090`)
- `PUSHGATEWAY_URL` (default `http://localhost:9091`)
- `LOCUST_WEB_PORT` (default `8089`)

### Constants

- `DATASET_DIR="datasets/OmniDocBench"`
- `CONTAINER_DATASET_ROOT="/app/datasets/OmniDocBench"`
- `RESULTS_DIR="loadtest/results"`
- `DASHBOARD_PATH="/d/glmocr-load/glm-ocr-load-test"`

### Functions

- **`log "<msg>"`** — blue `[omnidoc-${DRIVER_TAG}]` prefix to stdout.
- **`warn "<msg>"`** — yellow prefix to stderr.
- **`die "<msg>"`** — red prefix to stderr, `exit 1`.
- **`preflight_omnidoc`** — verifies `DATASET_DIR` exists (else `die`),
  `curl -fsS "${CPU_URL}/health"` succeeds (else `die` with "run `make
  up`"), and `curl -fsS "${GRAFANA_URL}/api/health"` — on failure warns
  and sets `GRAFANA_URL=""`.
- **`build_omnidoc_pool <urls_file>`** — runs `find … | shuf -n
  "${OMNIDOC_SAMPLE_POOL}" | sed …` into the given path, exports
  `POOL_SIZE` (int) and `IMAGES_CSV` (comma-joined), `die`'s if pool is
  empty.
- **`annotate <tag> <text>`** — posts Grafana annotation, silent no-op
  when `GRAFANA_URL=""`. Same payload shape as today.
- **`init_run <driver>`** — sets `DRIVER_TAG=<driver>`, picks
  `TS="$(date +%Y%m%d-%H%M%S)"`, sets `RUN_PREFIX="${RESULTS_DIR}/omnidoc-${TS}-${driver}"`,
  and `mkdir -p "${RESULTS_DIR}"`.

## Standalone script structure

Each of the three scripts follows this template (body differs only in
the driver invocation and the summary block):

```bash
#!/usr/bin/env bash
# OmniDocBench load test — <driver>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/scripts/lib/loadtest_common.sh"

init_run "<driver>"
log "starting run_id=omnidoc-${TS}-<driver>"

preflight_omnidoc

URLS_FILE="${RUN_PREFIX}.urls.txt"
build_omnidoc_pool "${URLS_FILE}"
log "pool_size=${POOL_SIZE} written to ${URLS_FILE}"

annotate "<driver>" "<driver> start"
# — driver-specific invocation (lifted verbatim from run_<driver> in the
#   old orchestrator, wiring ${URLS_FILE} / ${IMAGES_CSV} / ${TS} /
#   ${RUN_PREFIX}) —
annotate "<driver>" "<driver> end"

# — per-driver summary (the matching inline Python heredoc from the old
#   orchestrator, reading ${RUN_PREFIX}-*.{json,csv}) —

log "Grafana dashboard: ${GRAFANA_URL:-http://localhost:3000}${DASHBOARD_PATH}"
```

### Driver-specific bodies

All three bodies are lifted verbatim from the matching `run_*` function
in the old orchestrator. Minor touch-ups only:

- **asyncio** — output goes to `${RUN_PREFIX}.json` (no trailing `-asyncio`
  since `RUN_PREFIX` already encodes the driver). Python summary heredoc
  reads the same path.
- **locust** — CSV prefix is `${RUN_PREFIX}`. The summary heredoc reads
  `${RUN_PREFIX}_stats.csv`.
- **k6** — summary-export path is `${RUN_PREFIX}.json`.

## Makefile targets

```makefile
omnidoc-asyncio:
	"$(SHELL)" scripts/omnidoc_asyncio.sh

omnidoc-locust:
	"$(SHELL)" scripts/omnidoc_locust.sh

omnidoc-k6:
	"$(SHELL)" scripts/omnidoc_k6.sh
```

Help block gains:

```
omnidoc-asyncio   - OmniDocBench asyncio load test (sampled pool)
omnidoc-locust    - OmniDocBench locust load test (sampled pool)
omnidoc-k6        - OmniDocBench k6 load test (sampled pool)
```

`.PHONY` gains these three, loses the four deleted targets.

## `.env.example` changes

Delete:

```
LOCUST_IMAGES=file:///app/samples/receipt.png,file:///app/samples/table.png,file:///app/samples/invoice.pdf
DRIVERS=asyncio locust k6
```

The surrounding AWS / OmniDoc sections are otherwise unchanged. Retain
the `OMNIDOC_DURATION` / `OMNIDOC_SAMPLE_POOL` knobs — they are consumed
by the new shared helper.

## Driver fallback removal (behavior change)

| File | Before | After |
|------|--------|-------|
| `bench.py:98` | defaults pool to `["file:///app/samples/receipt.png"]` | raises `SystemExit(2)` with message pointing at `scripts/omnidoc_asyncio.sh` |
| `ocr_load.js:14-18` | defaults `IMAGES` to three `file:///app/samples/*` URLs | `throw new Error(...)` at module load when `__ENV.IMAGES` is unset |
| `locustfile.py:19-24` | defaults `LOCUST_IMAGES` to three URLs | `raise RuntimeError(...)` in `_images()` when unset |

Consequence: the three driver entry points become non-runnable without
an upstream source of images. That is intentional — the only supported
source is now OmniDocBench via the standalone scripts (or the user
explicitly passing the pool via flag/env).

## Smoke test

`scripts/smoke_test.sh` continues to work via a single-file bind mount.
The fixture image moves from `loadtest/samples/receipt.png` (via `git
mv`) to `scripts/smoke_test.png`. `docker-compose.yml` replaces the
directory bind mount with a single-file bind mount at
`/app/smoke_test.png`. The samples directory disappears entirely.

## Behavior parity vs today

For a single driver: `make omnidoc-asyncio` produces the same results
file paths as the old `DRIVERS=asyncio scripts/loadtest_omnidoc.sh`
(e.g. `loadtest/results/omnidoc-<ts>-asyncio.json`,
`loadtest/results/omnidoc-<ts>-locust_stats.csv`), with one behavior
difference: the outer `"omnidoc run <ts> start/end"` Grafana
annotations are gone. Only the per-driver `<driver> start/end`
annotations remain.

For running all three against the same sample: no longer supported.
Callers who need this property must invoke the drivers directly with a
shared `--image-list-file` / `IMAGES=` themselves.

## Open risks

1. **Smoke test regressions from the bind-mount change.** Single-file
   bind mounts on Windows Docker Desktop are known to behave differently
   than directory bind mounts (occasional mount-replaced-with-empty-dir
   bugs). Verify `make smoke` still passes on the target host after the
   compose change; if it flakes, fall back to a tiny `scripts/fixtures/`
   dir mount.
2. **`_resolve_image_pool` callers.** `bench.py` is called by the new
   `omnidoc_asyncio.sh`, by `make load-asyncio` (going away), and by any
   user muscle memory on the command line. Removing the fallback is a
   breaking change for that last group — document the flag in the error
   message.
3. **Locust CSV path.** The old orchestrator wrote
   `${RUN_PREFIX}-locust_stats.csv` (prefix `omnidoc-<ts>-locust`). The
   new script writes `${RUN_PREFIX}_stats.csv` (prefix
   `omnidoc-<ts>-locust`). Paths are identical by construction, but
   worth a grep over the summary heredocs to confirm.

## Out of scope

- Restoring quick-mode load tests against a local fixture pool.
- Adding new drivers.
- Changing OmniDocBench preprocessing (PDF handling, label filtering,
  etc.).
- Parallel driver execution.
