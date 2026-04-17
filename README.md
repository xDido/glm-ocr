# GLM-OCR Load-Test Harness

Two-container local rig for measuring timing, throughput, and request
concurrency of [zai-org/GLM-OCR](https://github.com/zai-org/GLM-OCR) before
deploying to AWS ECS.

- **CPU container** — upstream `glmocr.server` (Flask) behind gunicorn. Does
  image/PDF preprocessing and region-level fan-out. Flask HTTP-calls SGLang
  for every region — no model weights live here.
- **GPU container** — `lmsysorg/sglang` serving `zai-org/GLM-OCR` on an
  OpenAI-compatible `/v1/chat/completions` endpoint.
- **ministack overlay** — local AWS emulator. Register the shipped ECS task
  definitions against it to validate wiring before real AWS.
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
make load-asyncio         # fast concurrency sweep
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
| `SGL_MAX_RUNNING_REQUESTS` | `--max-running-requests` | SGLang batch cap                                     |
| `SGL_MAX_PREFILL_TOKENS`   | `--max-prefill-tokens`   | SGLang prefill budget                                |
| `SGL_MEM_FRACTION_STATIC`  | `--mem-fraction-static`  | Fraction of GPU mem reserved for KV cache            |
| `SGL_CHUNKED_PREFILL`      | `--chunked-prefill`      | Interleave prefill with decode                       |

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

```bash
make load-asyncio     # python loadtest/asyncio/bench.py --concurrency 16 --total 128
make load-locust      # locust headless, 50 users, 5/s ramp, 2 minutes
make load-k6          # k6 ramping VUs with p95 threshold
```

Results land in `loadtest/results/`. The asyncio script prints p50/p95/p99
directly. Sweep `OCR_MAX_WORKERS × SGL_MAX_RUNNING_REQUESTS` to build the
sizing table for ECS.

## ministack / ECS wiring test

```bash
make ministack-up     # adds ministack sidecar on :4566
make register-tasks   # registers ecs-task-{cpu,gpu}.json + Cloud Map services
aws --endpoint-url http://localhost:4566 --region us-east-1 \
    ecs list-task-definitions
```

**Caveat**: ministack accepts the GPU `resourceRequirements` in the task def
but has no way to schedule onto a real NVIDIA host. Expect
`DESIRED=1 / RUNNING=0` for the GPU service. ministack here validates API
shape and service-discovery wiring; **workload** tests go through plain
`docker compose` above.

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
- `loadtest/` — Locust, k6, asyncio bench; samples; results
- `scripts/` — smoke test, wait-for, ministack bootstrap
