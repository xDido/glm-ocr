# 09 — Production runbook

**Purpose:** on-call guide. What to do when an alarm fires, how to tune, and what each knob means in prod. Skim on first deploy; keep bookmarked for incidents.

---

## On-call quickref

**If anything is on fire, start here:**

```bash
# 1. Is the SageMaker endpoint healthy?
aws sagemaker describe-endpoint --endpoint-name glmocr-sglang --query 'EndpointStatus'

# 2. Are Fargate tasks healthy?
aws ecs describe-services --cluster glmocr-prod --services glmocr-cpu \
    --query 'services[0].{running:runningCount,desired:desiredCount,deployments:deployments[*].{status:status,rolloutState:rolloutState}}'

# 3. Recent CPU container logs
aws logs tail /glmocr/prod/cpu --since 10m --follow

# 4. Recent SageMaker container logs
aws logs tail /aws/sagemaker/Endpoints/glmocr-sglang --since 10m --follow
```

Dashboard: <https://dido.grafana.net/d/glmocr-load>. Alarms list: AWS Console → CloudWatch → Alarms → filter `glmocr-*`.

---

## The tuned knob set — what each one does

Source of truth: `reference/env-tuned.md`. This is the operator's-view summary. Everything below comes from dev measurements under `docs/OPTIMIZATIONS.md` + the 12 project memories in `reference/memory-seed.md`.

### CPU-side tunables (Fargate task env — SSM `/glmocr/prod/cpu/*`)

| Knob | Default | Effect when raised | Effect when lowered |
|---|---|---|---|
| `CPU_WORKERS=4` | — | Admits more parallel OS-process work; only helps if task has extra vCPUs unused. Diminishing past `vCPUs / 2`. | Less admission capacity → queue builds in gunicorn accept backlog. |
| `CPU_THREADS=16` | — | More per-worker gthread slots → more concurrent HTTP requests admitted. At 32 saw regression (oversubscription). | Tasks queue at ALB. |
| `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` | — | **Do not raise.** Lets C-kernels spawn extra OpenMP threads → CPU oversubscription (see kernel memory). | Can't go lower. |
| `OCR_MAX_WORKERS=32` | — | Per-request fan-out into SageMaker. Higher = faster when SGLang has headroom, but creates c=40+ queue explosions if endpoint is small. | Lower = less GPU pressure, slower per-request. |
| `OCR_REQUEST_TIMEOUT=60` | — | Slow requests still succeed. Watch for total wall time. | Faster fail on stuck endpoints; but HealthWatchdog may trip at under 30. |
| `OCR_RETRY_MAX=2` | — | Masks transient 503s. | At 0, transient SageMaker blips surface as user-facing 5xx. |
| `OCR_CONN_POOL=2048` | — | Must be ≥ `CPU_THREADS × OCR_MAX_WORKERS`. Raising never hurts. | Pool exhaustion surfaces as 503s mislabeled as SGLang failures. |
| `LAYOUT_BATCH_ENABLED=true`, `LAYOUT_BATCH_MAX=8`, `LAYOUT_BATCH_WINDOW_MS=20` | — | Cross-request layout coalescer. Measured 5× rps uplift on 2-worker. At BATCH_MAX=12 caused c=64 disconnects — **do not exceed 8.** | Layout runs unbatched; 2-5× rps regression. |
| `LAYOUT_ONNX_THREADS=2` | — | Intra-op parallelism. Target `CPU_WORKERS × LAYOUT_ONNX_THREADS ≤ vCPU count`. On 12-vCPU hosts, =3 is the measured optimum. | Layout forward slower. |

### SGLang-side tunables (SageMaker endpoint env — SSM `/glmocr/prod/sgl/*`)

| Knob | Default | Notes |
|---|---|---|
| `SGL_MAX_RUNNING_REQUESTS=64` | — | In-GPU batch size cap. Measured peaks 9–16 in dev regardless; raising this is a no-op unless CPU feeds faster. |
| `SGL_CONTEXT_LENGTH=24576` | — | Per-request KV budget. `prompt_tokens + max_tokens ≤ context_length` strictly (HTTP 400 on violation). Observed prompts ~240 tokens; floor is 8432. |
| `SGL_MEM_FRACTION_STATIC=0.95` | — | VRAM fraction pre-allocated. Lowering frees memory for optimizations like CUDA IPC. |
| `SGL_SCHEDULE_POLICY=lpm` | — | Longest-prefix-match. Beats fcfs by 16–28% rps (`project_sgl_schedule_policy` memory). **Don't change.** |
| `SGL_SPECULATIVE=true`, `SGL_SPEC_ALGORITHM=NEXTN` | — | SGLang aliases NEXTN → EAGLE. Already effectively MTP. See `project_cuda_ipc_mtp` memory. |
| `SGLANG_USE_CUDA_IPC_TRANSPORT=<unset>` | — | Flip to `1` after first-ship smoke. Watch CloudWatch logs for `out of memory` at the `storage._share_cuda_` call. |

---

## Alarm runbook

### 🚨 `glmocr-sm-5xx-errors` — SageMaker 5XX errors > 5/min

**Immediate:**
1. `aws logs tail /aws/sagemaker/Endpoints/glmocr-sglang --since 10m`
2. Look for: `CUDA out of memory` → flip `SGLANG_USE_CUDA_IPC_TRANSPORT` off (SSM) and redeploy.
3. Look for: `CUDA error` (any) → endpoint instance is unhealthy; SageMaker will self-heal. If persistent, restart: `aws sagemaker update-endpoint ... --retain-all-variant-properties false`.
4. Look for: rate-limit / throttle errors → autoscaling may be lagging; check `SageMakerInvocationsPerInstance`.

**If nothing obvious:** roll back to the previous endpoint-config (keep last 3 revisions). `aws sagemaker update-endpoint --endpoint-name glmocr-sglang --endpoint-config-name <previous>`.

### 🚨 `glmocr-fargate-memory-high` — Task memory > 85% for 5 min

**Immediate:** typically a slow memory leak from the ORT session or from the layout batcher retaining references. Task will OOM and restart on its own; watch the dashboard.

**If chronic:**
1. Force rolling deploy to clear: `aws ecs update-service --cluster glmocr-prod --service glmocr-cpu --force-new-deployment`.
2. If it returns within 24h, raise task memory from 16 GB → 24 GB (CDK context, redeploy).
3. Heap-profile: exec into a task (`aws ecs execute-command`) and run `py-spy dump --pid <gunicorn-pid>` to see what's retained. Expected culprits: ORT session, tokenizer cache. File an issue.

### 🚨 `glmocr-alb-5xx` — ALB HTTPCode_Target_5XX > 10 in 5 min

**Immediate:**
1. `aws logs tail /glmocr/prod/cpu --since 10m --filter-pattern '"5xx" OR "500" OR "503"'`
2. Most common: `HealthWatchdog: OCR service at <...>:30000 is no longer available` → the sigv4-proxy or SageMaker upstream is down. Check sigv4-proxy logs (`/glmocr/prod/sigv4-proxy`).
3. If proxy is fine: check SageMaker endpoint status; this alarm often precedes the SM-5xx alarm.

### 🚨 `glmocr-alloy-remote-write-lag` — remote_write_pending_samples > 100k

**Immediate:** Grafana Cloud is rate-limiting us. Check the cardinality allowlist in `docker/alloy/config.alloy` — did someone add a new histogram?

**Fix:** either raise the allowlist to admit it, or add the series name to the drop list.

### 🚨 `glmocr-sm-latency-high` — ModelLatency p95 > 30 s

**Most likely cause:** SGLang is processing a pathologically long prompt. Unusual in OCR context. Check `SGL_MAX_RUNNING_REQUESTS` and `SGL_CONTEXT_LENGTH` (SSM) — ensure they haven't been lowered.

**If truly pathological:** a single bad client document is pinning a slot. Look at the invocations dashboard; identify the originating caller; coordinate with upstream.

---

## Common operations

### Deploy a new code change

```bash
# It's a PR flow — see 08-ci-cd.md
# Merge the PR to main → CI deploys → manual approval → ~10 min rollout
```

### Flip a tuning knob

```bash
aws ssm put-parameter \
    --name /glmocr/prod/cpu/OCR_MAX_WORKERS \
    --value 40 \
    --overwrite
aws ecs update-service --cluster glmocr-prod --service glmocr-cpu --force-new-deployment

# Verify:
aws ecs describe-services --cluster glmocr-prod --services glmocr-cpu --query 'services[0].deployments'
# Wait ~60 s; Grafana panel "CPU container env" should show the new value.
```

### Rotate HF_TOKEN

```bash
# 1. Generate new token in HuggingFace Hub UI
# 2. Update Secrets Manager
aws secretsmanager put-secret-value \
    --secret-id glmocr/prod/hf_token \
    --secret-string "<new-token>"
# 3. Roll the task
aws ecs update-service --cluster glmocr-prod --service glmocr-cpu --force-new-deployment
```

### Bake and ship new weights

```bash
# On the MacBook
export HF_TOKEN=<token>
export WEIGHTS_BUCKET=glmocr-prod-weights-<acct>
KEY="glm-ocr/$(date +%Y%m%d)/model.tar.gz" bash scripts/bake-weights.sh
aws ssm put-parameter \
    --name /glmocr/prod/sagemaker/model_data_key \
    --value "$KEY" \
    --overwrite

# Then: update CDK to pick up the new key + redeploy SagemakerStack (causes
# an endpoint update; blue-green minimizes downtime). 8-15 min.
cd <repo> && cdk deploy --context stage=prod glmocr-sagemaker-prod
```

### Run a matrix test against prod

```bash
export CPU_URL=http://<alb-url>
MATRIX_TOTAL=100 bash scripts/omnidoc_asyncio_matrix.sh

# Report lands in loadtest/results/omnidoc-<ts>-asyncio-matrix.md
# Compare against reference/prod-baseline.json
```

### Debug a live task

```bash
# Enable ECS exec if not already
aws ecs update-service --cluster glmocr-prod --service glmocr-cpu --enable-execute-command

# Find a task ID
TASK=$(aws ecs list-tasks --cluster glmocr-prod --service-name glmocr-cpu --query 'taskArns[0]' --output text)

# Shell into the cpu container
aws ecs execute-command --cluster glmocr-prod --task $TASK --container cpu --interactive --command /bin/bash
```

---

## Cost envelope (single-AZ, 1 task, 1 SM instance, baseline load)

| Component | Size | Monthly (us-east-1, on-demand) |
|---|---|---|
| Fargate task | 8 vCPU / 16 GB, 1 task × 730 h | ~$350 |
| SageMaker endpoint | ml.g4dn.2xlarge × 730 h | ~$550 |
| NAT Gateway | 1 × 730 h + data | ~$35 |
| S3 (weights + reports) | ~5 GB | ~$0.50 |
| Secrets Manager | 3 secrets | ~$1.20 |
| SSM Parameter Store | Standard tier | $0 |
| CloudWatch Logs | ~5 GB ingested | ~$2.50 |
| Grafana Cloud | free-tier | $0 |
| **Total baseline** | | **~$940/mo** |

Scale-out: each additional Fargate task ≈ $350/mo; each additional SageMaker instance ≈ $550/mo.

---

## Future work (not in first ship)

| Item | Trigger | Effort |
|---|---|---|
| CUDA IPC Transport on g4dn.2xlarge | After smoke of endpoint; set `SGLANG_USE_CUDA_IPC_TRANSPORT=1` → redeploy → watch OOM | 1 hour |
| Streaming completions | Downstream client demands token-by-token | ~1 day (Go fallback proxy + `serve.py` update) |
| Alloy-in-SageMaker-container | SageMaker CW metrics feel too coarse | ~half day |
| Reserved Instances / Savings Plans | After 3 months of stable load | Budget |
| Xeon-AVX-512-VNNI re-evaluation of DocLayout-YOLO | If a different Fargate/EC2 CPU class is on the table | Separate probe session |
| Multi-AZ SageMaker endpoint | HA requirement emerges | CDK change |
| Request-level tracing (OpenTelemetry) | Debugging cross-service latency gets painful | ~2 days |

---

Next: [`10-claude-code-handoff.md`](./10-claude-code-handoff.md).
