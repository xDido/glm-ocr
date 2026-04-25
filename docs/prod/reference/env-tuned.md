# reference/env-tuned.md

**Provenance:** copied verbatim from the dev repo's `docs/OPTIMIZATIONS.md` §"TL;DR — the final `.env` block" (file timestamp 2026-04-23). These values are the measured-best from ~3 weeks of load-testing on g4dn.2xlarge-class dev hardware.

**In prod, the .env file itself does not exist.** Instead:

- `SGL_*` knobs → SSM Parameter Store `/glmocr/prod/sgl/*` → SageMaker endpoint `Environment`
- `OCR_*`, `LAYOUT_*`, `CPU_*`, `OMP_*`, `MKL_*`, `GLMOCR_*` knobs → SSM Parameter Store `/glmocr/prod/cpu/*` → Fargate task-def `environment`
- `HF_TOKEN` → Secrets Manager `glmocr/prod/hf_token` → Fargate task-def `secrets`
- `GRAFANA_CLOUD_*` → Secrets Manager `glmocr/prod/grafana_cloud` (JSON) → Alloy sidecar env
- `SGLANG_HOST`/`SGLANG_PORT`/`SGLANG_SCHEME` → **hardcoded in CDK** as `127.0.0.1:30000` (task-local sigv4-proxy). Do NOT expose as SSM — accidental change would break SigV4 signing.

See `docs/prod/07-secrets-and-config.md` for the full translation.

---

## The tuned block

```ini
# --- layout inference backend ---
LAYOUT_ENABLED=true
LAYOUT_DEVICE=cpu              # CUDA rejected on 8 GB cards
LAYOUT_BACKEND=onnx            # ORT instead of torch eager — 1.76x on forward
LAYOUT_POSTPROC=numpy          # numpy post-proc — +6–17% rps, -14–31% p95
LAYOUT_GRAPH=raw               # fused graph rolled back — regressed at c=24/32
LAYOUT_USE_POLYGON=false       # polygon mode not needed for block-level OCR
LAYOUT_COMPILE=false           # torch.compile regresses +19% mean
LAYOUT_ONNX_THREADS=2          # saturates 8-core cgroup: 4 workers x 2 ORT = 8
                               # grow with cores: at 12c use 3 (+34% rps c=12)
LAYOUT_BATCH_ENABLED=true      # cross-request layout coalescer
LAYOUT_BATCH_MAX=8             # >8 causes c=64 ServerDisconnectedError
LAYOUT_BATCH_WINDOW_MS=20

# --- CPU container sizing ---
CPU_WORKERS=4                  # gunicorn processes
CPU_THREADS=16                 # gthread per worker — 32 caused c=64 oversubscription
OMP_NUM_THREADS=1              # prevents MKL/OMP oversubscribing cgroup
MKL_NUM_THREADS=1

# --- HTTP fan-out + pool ---
OCR_MAX_WORKERS=32             # intra-request parallel region calls
OCR_CONN_POOL=2048             # must be >= CPU_THREADS * OCR_MAX_WORKERS
OCR_CONNECT_TIMEOUT=10
OCR_REQUEST_TIMEOUT=60
OCR_RETRY_MAX=2                # 1 -> 2 eliminated c=64 ServerDisconnectedError at n=200
OCR_RETRY_BACKOFF_BASE=0.5
OCR_RETRY_BACKOFF_MAX=8
GUNICORN_TIMEOUT=480           # must exceed OCR_REQUEST_TIMEOUT

# --- CPU <-> SGLang (local via sigv4-proxy sidecar in prod) ---
SGLANG_HOST=127.0.0.1          # hardcoded in CDK; do NOT put in SSM
SGLANG_PORT=30000
SGLANG_SCHEME=http
OCR_MODEL_NAME=glm-ocr

# --- SGLang server-side (SageMaker endpoint Environment) ---
SGL_MODEL_PATH=/opt/ml/model   # SM mounts weights here; not zai-org/GLM-OCR
SGL_SERVED_MODEL_NAME=glm-ocr
SGL_TP_SIZE=1
SGL_DTYPE=float16
SGL_MAX_RUNNING_REQUESTS=64
SGL_MAX_PREFILL_TOKENS=8192
SGL_MAX_TOTAL_TOKENS=200000
SGL_MEM_FRACTION_STATIC=0.95
SGL_CONTEXT_LENGTH=24576       # prompt+gen must be <= this (strict)
SGL_CHUNKED_PREFILL=true
SGL_CHUNKED_PREFILL_SIZE=8192
SGL_SCHEDULE_POLICY=lpm        # beats fcfs by 16-28% — don't change
SGL_SPECULATIVE=true
SGL_SPEC_ALGORITHM=NEXTN       # SGLang aliases NEXTN -> EAGLE internally
SGL_SPEC_NUM_STEPS=3
SGL_SPEC_EAGLE_TOPK=1
SGL_SPEC_NUM_DRAFT_TOKENS=4

# --- SGLang cookbook optimizations (optional, try after smoke) ---
# SGLANG_USE_CUDA_IPC_TRANSPORT=1   # 16 GB g4dn may or may not fit; flip + test
```

---

## Process model

**Gunicorn `--worker-class gthread --workers $CPU_WORKERS --threads $CPU_THREADS`.** Async Flask isn't the right model — blocking C extensions under layout make async cooperatively starved. Gthread gives you OS-level preemption per request.

**Math-thread caps:** without `OMP_NUM_THREADS=1` + `MKL_NUM_THREADS=1`, each of the 4 gunicorn workers spawns a per-kernel OpenMP pool sized to the host core count. On an 8-vCPU cgroup that's 4×8 = 32 math threads fighting 8 cores under `CPU_THREADS=16` = 64 Python threads — immediate catastrophic oversubscription. Pin both to 1 and let `LAYOUT_ONNX_THREADS` own the intra-op parallelism for the only kernel that benefits.

---

## Prod-specific deviations from the dev block

| Knob | Dev value | Prod value | Reason |
|---|---|---|---|
| `SGL_MODEL_PATH` | `zai-org/GLM-OCR` | `/opt/ml/model` | SageMaker mounts the S3-staged weights at that path; ignores HF ID |
| `SGLANG_HOST` | `sglang` (compose DNS) | `127.0.0.1` | sigv4-proxy sidecar — mandatory |
| `LAYOUT_ONNX_THREADS` | `2` (for 8 vCPU) | `2` initially; bump to 3 if Fargate task is 12 vCPU | Measured +34% rps at 12c |
| `HF_HOME` | `./hf-cache` bind mount | default `~/.cache/huggingface` | No HF-cache persistence needed in prod (weights come from S3) |

All other values **carry over unchanged**. Do not re-derive. If you think a value is wrong, run two matrix runs (minimum per `feedback_matrix_noise.md`) and file a diff.

---

## Known "don't touch" tripwires

| Knob | Why |
|---|---|
| `SGL_SCHEDULE_POLICY=lpm` | Tested against fcfs; fcfs is -16 to -28% rps and trips capacity cliffs at c=32/40 |
| `LAYOUT_BATCH_MAX=8` | >8 caused c=64 disconnects in dev measurements |
| `LAYOUT_POSTPROC=numpy` (not torch) | Torch path is slower; numpy path has bit-parity validated |
| `LAYOUT_GRAPH=raw` (not fused) | Fused regressed at c=24/32 in dev; kept the flag for future |
| ~~`LAYOUT_ASYNC`~~ (removed 2026-04-25) | Async sidecar was tested and removed; c=40 HealthWatchdog + c=64 ServerDisconnectedError. Don't re-add without new hypothesis |
| `SGL_CONTEXT_LENGTH >= 8432` | Hard floor: glmocr sends `max_tokens=8192`; SGLang strictly enforces `prompt + max_tokens <= context_length` |
| `SGLANG_HOST=127.0.0.1` | sigv4-proxy sidecar assumes loopback; public SageMaker URL would bypass signing + 403 |

---

## What was **rejected** (don't retry without new hypothesis)

See `memory-seed.md` in this folder for the full list and reasoning. Headliners:

- `LAYOUT_INPUT_SIZE=640x640` — only parity-validated on OmniDocBench, unsafe for passport/ID
- Static int8 quantization — DETR head too fragile for PTQ
- DocLayout-YOLO detector swap — 3× slower on Ryzen 5600X CPU (public benchmarks were GPU)
- FastAPI async sidecar — net-worse throughput + new failure modes
- OpenVINO Execution Provider — apparent 3× win was 15% silent empty responses
- `torch.compile` on layout model — +19% mean regression
- Layout on CUDA — steals VRAM from SGLang's KV cache on small GPUs
