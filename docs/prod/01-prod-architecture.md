# 01 — Production architecture

**Purpose:** one-page picture of the target AWS system and the why behind each choice. Read before any other numbered doc.

---

## System diagram

```
                            ┌─────────────────────────┐
   public caller ──HTTPS──► │  internal ALB (5002)    │  (VPC-internal; ALB
                            │  target: Fargate task   │   fronts optional —
                            └───────────┬─────────────┘   call task IP
                                        │                 directly if single-AZ)
                                        │
                ┌───────────────────────▼────────────────────────┐
                │          ECS Fargate Service — glmocr-cpu      │
                │          (min=1, max=4, CPU target 60%)        │
                │                                                 │
                │  ┌──────────────────┐  ┌────────────────────┐  │
                │  │ cpu container    │  │ sigv4-proxy        │  │
                │  │ (Dockerfile.slim)│  │ sidecar            │  │
                │  │                  │  │ (aws-sigv4-proxy)  │  │
                │  │ calls            │◄─┤ listens            │  │
                │  │ 127.0.0.1:30000 ─┼─►│ 127.0.0.1:30000    │  │
                │  └──────────────────┘  │ signs with SigV4,  │  │
                │  ┌──────────────────┐  │ forwards to SM     │  │
                │  │ alloy sidecar    │  │ InvokeEndpoint     │  │
                │  │ scrapes          │  └─────────┬──────────┘  │
                │  │ localhost:5002/  │            │             │
                │  │ metrics, remote_ │            │             │
                │  │ writes to        │            │             │
                │  │ Grafana Cloud    │            │             │
                │  └──────────────────┘            │             │
                └──────────────────────────────────┼─────────────┘
                                                   │
                                          (VPC endpoint:
                                           sagemaker-runtime)
                                                   │
                                                   ▼
                         ┌──────────────────────────────────────┐
                         │ SageMaker real-time endpoint          │
                         │ glmocr-sglang                         │
                         │  instance: ml.g4dn.2xlarge (T4 16 GB) │
                         │  image: ECR glmocr-sglang (BYOC)      │
                         │  ModelDataUrl: s3://…/glm-ocr.tar.gz  │
                         │  exposed contract: /ping /invocations │
                         │  internally: SGLang on :30000         │
                         └──────────────────────────────────────┘

   S3             Secrets Manager     CloudWatch        Grafana Cloud Mimir
   ├── weights    ├── hf_token        ├── SM metrics    └── all app metrics
   └── reports    └── grafana_cloud   └── ECS logs          (via Alloy sidecar)
```

---

## Component rationale

### CPU container → AWS Fargate
- No GPU needed (layout runs on CPU, ONNX Runtime).
- Fargate gives 8-vCPU / 16-GB or 16-vCPU / 32-GB sizes matching `g4dn.2xlarge` CPU tier — direct dev-to-prod sizing parity.
- Task-def update deploys new env knobs in ~60 s, matching the iteration speed we had locally with `docker compose up -d --force-recreate`.
- Prometheus `/metrics` endpoint of the CPU container is preserved — scraped by the Alloy sidecar in the same task (localhost).

### GPU container → SageMaker real-time endpoint
- Explicit user direction.
- `ml.g4dn.2xlarge` (NVIDIA T4, 16 GB VRAM): same GPU class as the dev-box target the `.env` was tuned against. Direct parity between the tuned knobs (`SGL_MEM_FRACTION_STATIC=0.95`, `SGL_CONTEXT_LENGTH=24576`, etc.) and the prod GPU — no re-tuning needed for the first cut.
- Managed autoscaling, blue-green deploys, and model registry — the reasons not to roll our own ECS-hosted SGLang.
- BYOC container (our own image around `lmsysorg/sglang:<pinned>`) because SageMaker's built-in images don't include SGLang.
- **CUDA IPC Transport** (cookbook optimization, blocked on dev's 8 GB 3060 Ti) is still uncertain at 16 GB — the `MmItemMemoryPool` allocation needs multi-GB contiguous free VRAM on top of the model + KV cache. Plumbing is already in the container; flip `SGLANG_USE_CUDA_IPC_TRANSPORT=1` in SageMaker endpoint config and smoke-test. If OOM, revert. See the CUDA IPC + MTP memory entry in `reference/memory-seed.md`.

### SigV4-proxy sidecar
- `glmocr.ocr_client` is a pip dependency whose HTTP client we don't control. Forking it to add SigV4 signing would couple the prod repo to a glmocr fork.
- The sidecar (`public.ecr.aws/aws-observability/aws-sigv4-proxy:latest`) listens on `127.0.0.1:30000` inside the task, signs every request with SigV4, and forwards to the SageMaker endpoint via the `sagemaker-runtime` VPC endpoint.
- From glmocr's point of view, nothing changed: it still POSTs to `sglang:30000/v1/chat/completions`.
- One-container, zero custom Go. Falls back to a ~30-line Go service only if the upstream image can't handle streaming quirks. See `05-sigv4-proxy-sidecar.md`.

### Alloy sidecar
- Fargate tasks have ephemeral IPs; a central Prometheus can't reliably scrape them.
- Alloy already does the remote-write egress to Grafana Cloud in dev. Running it as a sidecar keeps the dev dashboards (`glmocr-load.json`) working unchanged.
- The sidecar scrapes `localhost:5002/metrics` (CPU container) — zero network cost, no service-discovery churn.
- SageMaker endpoint's own `/metrics` is not exposed to Alloy; we rely on **CloudWatch** for SageMaker-side visibility (see `06-observability-prod.md`). Cross-visualized in Grafana via the CloudWatch datasource.

### Secrets / config split
- **Secrets Manager** for anything rotatable or sensitive: `HF_TOKEN`, `GRAFANA_CLOUD_PROM_TOKEN`, `GRAFANA_SA_TOKEN`.
- **SSM Parameter Store** (plain, non-secret) for tuned knobs: all `SGL_*`, `OCR_*`, `LAYOUT_*`. One param per knob under `/glmocr/prod/*`. Enables a single-knob hot-patch via the AWS console + ECS force-new-deployment.
- No `.env` file in the prod repo — the source of truth is Parameter Store + Secrets Manager, synced into Fargate via the task-def `secrets:` and `environment:` fields.

### Model weights → S3
- In dev, SGLang downloads `zai-org/GLM-OCR` from HuggingFace Hub on first boot (~1.8 GB) into a bind-mounted cache.
- In prod, we tar the weights and stage them in S3; SageMaker pulls via `ModelDataUrl` on endpoint creation and mounts them at `/opt/ml/model/` inside the container. Cold-start time: ~30–60 s (S3 → instance local disk, single decompress).
- Avoids relying on HuggingFace availability for endpoint boot.

---

## Data flow — one OCR request end-to-end

1. Client POSTs a JSON payload (`{"images": [...base64...]}`) to the internal ALB.
2. ALB routes to a Fargate task's `cpu` container on port 5002.
3. Gunicorn gthread worker processes the request. `LAYOUT_BATCH_ENABLED=true` coalesces with in-flight siblings (`LAYOUT_BATCH_WINDOW_MS=20`).
4. ONNX Runtime runs PP-DocLayoutV3 on the CPU (Conv-heavy, ~1,100 ms on g4dn-class). Returns N region crops.
5. `OCR_MAX_WORKERS=32` threadpool fans out N HTTP POSTs to `127.0.0.1:30000/v1/chat/completions`.
6. The **sigv4-proxy sidecar** accepts each POST on port 30000, SigV4-signs it, and calls `sagemaker-runtime:InvokeEndpoint` on the `glmocr-sglang` endpoint.
7. SageMaker delivers the POST body to the endpoint container's `POST /invocations`. Inside the container, `serve.py` forwards to the local SGLang at `localhost:30000/v1/chat/completions`.
8. SGLang runs GLM-OCR (with NEXTN speculative decoding), returns the OpenAI-shaped JSON response.
9. Response flows back through the same path in reverse. CPU container assembles the per-region outputs into a final markdown + JSON response.
10. Alloy sidecar, on a 5-second cadence, scrapes the CPU container's `/metrics` and remote-writes to Grafana Cloud Mimir.

---

## IAM trust boundaries

| Principal | What it can do | Why |
|---|---|---|
| Fargate task role | `sagemaker:InvokeEndpoint` on `glmocr-sglang`; `secretsmanager:GetSecretValue` on `glmocr/prod/*`; `ssm:GetParameters*` on `/glmocr/prod/*`; `logs:PutLogEvents` | Runtime calls |
| Fargate execution role | `ecr:GetDownloadUrlForLayer/BatchGetImage`; `logs:CreateLogStream`; `secretsmanager:GetSecretValue` (for image-pull secrets if private ECR) | Task bootstrap |
| SageMaker endpoint role | `s3:GetObject` on the weights bucket; `logs:PutLogEvents` | Endpoint boot + logs |
| CI role (GitHub Actions → AWS via OIDC) | `ecr:PutImage`; `cloudformation:*` on stacks named `glmocr-*`; `iam:PassRole` on the three roles above | Deploys |

No IAM user with a static access key. All access is via role assumption.

---

## Failure modes and blast radius

| Failure | Who notices | Mitigation |
|---|---|---|
| SageMaker endpoint in `Failed` state | Fargate `/glmocr/parse` returns 500 with glmocr's "OCR service unavailable" | CloudWatch alarm on `InvocationErrors` → page. Fix by redeploying the endpoint from a known-good model version in the registry. |
| Fargate task OOMs | ECS events + healthcheck fail | Task killed, service launches a new one. Alarm on `MemoryUtilization > 85%`. If chronic, raise task memory in CDK and redeploy. |
| HuggingFace rate-limit on boot | SageMaker endpoint never reaches `InService` | Not applicable: weights come from S3, not HF, at prod boot. |
| Grafana Cloud over-cardinality | Alloy `/metrics` shows queue growing; free-tier throttles | Allowlist in `config.alloy` limits series. If we add a new histogram in code, it needs an allowlist entry first. Alert on Alloy's own `remote_write_pending_samples`. |
| VPC endpoint for sagemaker-runtime down | All OCR requests return 500 | Multi-AZ VPC endpoint (CDK defaults). Alert on endpoint health. |
| HF_TOKEN rotated without SSM update | SageMaker boot fails (only if we fall back to HF download) | Not applicable in steady state. Only matters when re-baking the weights tarball — a manual job, runbook in `09-runbook.md`. |

---

Next: [`02-cdk-go-structure.md`](./02-cdk-go-structure.md).
