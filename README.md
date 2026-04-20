# GLM-OCR Load-Test Harness

Two-container local rig for measuring timing, throughput, and request
concurrency of [zai-org/GLM-OCR](https://github.com/zai-org/GLM-OCR) before
deploying to AWS ECS.

- **CPU container** — upstream `glmocr.server` (Flask) behind gunicorn. Does
  image/PDF preprocessing and region-level fan-out. Flask HTTP-calls SGLang
  for every region — no model weights live here.
- **GPU container** — `lmsysorg/sglang` serving `zai-org/GLM-OCR` on an
  OpenAI-compatible `/v1/chat/completions` endpoint.
- **Load tests** — Locust (UI + CSV), k6 (thresholds), asyncio (fast sweep).

```
client ──HTTP──▶ cpu:5002 (gunicorn/Flask/glmocr.server)
                    │
                    └─HTTP /v1/chat/completions──▶ sglang:30000 (SGLang + GLM-OCR)
```

## Quickstart

```bash
cp .env.example .env
# edit .env — especially CPU_WORKERS, CPU_THREADS, SGL_MAX_RUNNING_REQUESTS

docker pull lmsysorg/sglang:latest   # one-off; compose also auto-pulls on `make up`
make up                   # builds CPU image + starts CPU + GPU; first GPU boot
                          # downloads ~1.8GB of weights into ./hf-cache/
                          # (git-ignored, persists locally)
make logs                 # tail until you see SGLang "server is fired up"
make smoke                # end-to-end OCR on a sample image
make runtime              # integrity check: env vs live process + SGLang state
make omnidoc-asyncio      # fast concurrency sweep
```

The GPU service runs `lmsysorg/sglang:${SGL_IMAGE_TAG}` (default `latest`)
straight from Docker Hub — no local GPU image build. The launch args are
assembled by `docker/gpu/entrypoint.sh`, which is bind-mounted into the
container at runtime. To pin a specific release, set `SGL_IMAGE_TAG` in
`.env` (e.g. `v0.5.10.post1-cu130-runtime`).

Tear down: `make down` (keeps HF cache) or `make clean` (drops everything).

## Knob reference

All knobs live in `.env`. The mapping:

| ENV var                    | Surface           | What it does                                                |
|----------------------------|-------------------|-------------------------------------------------------------|
| `CPU_WORKERS`              | gunicorn          | Process-level fan-out                                       |
| `CPU_THREADS`              | gunicorn          | Per-worker thread count                                     |
| `GUNICORN_TIMEOUT`         | gunicorn          | Hard kill for stuck workers (s)                             |
| `OCR_MAX_WORKERS`          | `pipeline.max_workers` | Parallel SGLang calls per document (region fan-out)    |
| `OCR_REQUEST_TIMEOUT`      | `pipeline.ocr_api.request_timeout` | SGLang request timeout (s)                 |
| `OCR_RETRY_MAX`            | `pipeline.ocr_api.retry_max_attempts` | Retry count for 429/5xx                 |
| `OCR_CONN_POOL`            | `pipeline.ocr_api.connection_pool_size` | Must be ≥ `CPU_THREADS × OCR_MAX_WORKERS` |
| `LAYOUT_ENABLED`           | template branch   | false → bypass PP-DocLayoutV3, send full pages              |
| `SGL_MAX_RUNNING_REQUESTS` | `--max-running-requests` | SGLang batch cap (how many requests batch on the GPU) |
| `SGL_MAX_PREFILL_TOKENS`   | `--max-prefill-tokens`   | Tokens per prefill step                              |
| `SGL_MAX_TOTAL_TOKENS`     | `--max-total-tokens`     | Total KV-cache slots across in-flight requests       |
| `SGL_MEM_FRACTION_STATIC`  | `--mem-fraction-static`  | Fraction of GPU mem reserved for KV cache            |
| `SGL_CHUNKED_PREFILL`      | `--chunked-prefill-size` | Interleave prefill with decode (size from `SGL_CHUNKED_PREFILL_SIZE`) |
| `SGL_SCHEDULE_POLICY`      | `--schedule-policy`      | `lpm` (prefix-cache aware) or `fcfs`                 |

See `.env.example` for the full list.

### Concurrency math

Total concurrent `/v1/chat/completions` SGLang sees is approximately
`CPU_WORKERS × CPU_THREADS × OCR_MAX_WORKERS`. Tune `SGL_MAX_RUNNING_REQUESTS`
against this target. `OCR_CONN_POOL` must comfortably exceed
`CPU_THREADS × OCR_MAX_WORKERS` or you'll see pool-exhaustion 503s that
look like SGLang failures.

## Runtime-integrity observability

The CPU container exposes `GET /runtime/summary` (terse) and `GET /runtime`
(full) that report what the process **actually** looks like — not what
`.env` claimed. Use this to verify knobs took effect:

```bash
make runtime           # summary view (integrity-focused)
make runtime-full      # full: env vs config vs live process + Prometheus
```

`summary` shows, per knob, `env → actual`:

```jsonc
{
  "cpu_workers":            { "env": "2",  "actual": 2 },
  "cpu_threads_per_worker": { "env": "8",  "actual_per_worker": [8, 8] },
  "ocr_max_workers":        { "env": "8",  "config": 8 },
  "sglang_max_running":     { "env": "16", "runtime": 16,
                              "live_running": 4, "live_queued": 0 },
  "sglang_batch_tokens":    { "env_prefill": "16384", "runtime_prefill": 16384, ... },
  "sglang_dtype":           { "env": "float16", "runtime": "float16" },
  "sglang_model":           { "env": "zai-org/GLM-OCR", "runtime": "zai-org/GLM-OCR" }
}
```

Sources:
- `workers` / `threads` come from `psutil` reading the gunicorn master's child PIDs.
- `sglang_*.runtime` comes from SGLang's `GET /get_server_info`.
- `live_running` / `live_queued` come from SGLang's Prometheus `/metrics`
  (enabled via `--enable-metrics` in the GPU entrypoint).
- `ocr_max_workers.config` is the value actually in the rendered
  `/app/config.yaml` — catches envsubst mishaps.

If `env` and `actual`/`runtime` ever disagree, something isn't wired
through — fail the config before trusting load-test numbers.

## Datasets

Test dataset fetcher + docs in `datasets/README.md`. Quick reference:

| Dataset | Size | Use |
|---|---|---|
| **OmniDocBench** | ~1.5 GB | The benchmark the GLM-OCR paper reports on. Use for apples-to-apples quality comparison. |
| **FUNSD** | ~25 MB | Small, noisy scanned forms. Fast smoke test. |
| **DocVQA (1200)** | ~150 MB | Doc-level Q&A; good for VLM eval. |

```bash
make datasets                    # FUNSD (fast)
make datasets-omnidocbench       # the full benchmark
```

Datasets land in `./datasets/` (git-ignored). The model weights land in
`./hf-cache/` (also git-ignored) — everything stays inside the project.

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

### Staged sweep

`scripts/tune_params.py` has three staged modes for producing an
evidence-based sizing report. All use `N=200` per cell, a seeded image
pool for reproducibility, and a 10% fail-rate abort that cuts off bad
cells mid-bench (configurable):

```bash
# Stage A — 1D scans across every SGLang knob (incl. speculative
# decoding sub-knobs + SGL_MEM_FRACTION_STATIC) and OCR_CONN_POOL.
python scripts/tune_params.py --stage a --dry-run     # preview matrix
python scripts/tune_params.py --stage a               # progress log shows [N/M pct%]

# Stage B — 2D fine-tune on the top-2 axes from Stage A.
python scripts/tune_params.py --stage b --axes OCR_MAX_WORKERS,SGL_MAX_RUNNING_REQUESTS

# Stage C — c-curve verification at a fixed config.
python scripts/tune_params.py --stage c \
    --set OCR_MAX_WORKERS=4 --set SGL_MAX_RUNNING_REQUESTS=24

# Flags:
#   --max-fail-rate 0.10         default; 0 disables early-abort
#   --min-sample-for-abort 40    observations before abort can trigger
```

The CPU container is built from `docker/cpu/Dockerfile.slim`
(python-slim + torch+cpu from PyTorch's CPU wheel index). Saves ~5 GB
over the CUDA-inclusive default. Rebuild cleanly with
`bash scripts/rebuild_and_up.sh`.

The sweep includes SGLang speculative decoding (NEXTN heads shipped in
the GLM-OCR weights); see `.env` for `SGL_SPECULATIVE` + sub-knobs and
`docker/gpu/entrypoint.sh` for how they wire into the launch args.

Raw per-trial JSON lands under `loadtest/results/raw/<run-id>/`; render
inline-PNG markdown reports with:

```bash
python scripts/lib/render_report.py stage \
    --trials loadtest/results/raw/<run-id>/_trials.json \
    --out    loadtest/results/<run-id>.md \
    --run-id <run-id>
```

## Windows + NVIDIA Docker notes

The repo targets Windows 10 + Docker Desktop. Before `make up`:

1. Install the latest NVIDIA driver for Windows.
2. Enable WSL2 backend in Docker Desktop settings.
3. Verify GPU passthrough: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`.
4. The first `make up` downloads ~1.8 GB of GLM-OCR weights into the
   `hf-cache` volume — budget 3-10 minutes depending on your link.

## Files

- `docker/cpu/` — CPU container (Dockerfile, WSGI shim, config template, entrypoint)
- `docker/gpu/` — SGLang arg-assembly entrypoint (bind-mounted into the
  upstream `lmsysorg/sglang` image; the sibling `Dockerfile` is kept only for
  environments that can't bind-mount)
- `aws/` — ECS task definitions + registration script
- `loadtest/` — Locust, k6, asyncio bench; results
- `scripts/` — smoke test, wait-for, dataset fetcher
