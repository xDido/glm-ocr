# 04 — Fargate task definition for the CPU container

**Purpose:** concrete ECS Fargate task spec for the CPU layer. Three containers in one task: `cpu`, `sigv4-proxy`, `alloy`.

---

## Why three containers, one task

They are **tightly coupled**: the CPU container calls `127.0.0.1:30000` which only reaches the sigv4-proxy if they share a network namespace (task-local loopback). Alloy must also scrape `127.0.0.1:5002/metrics`, same reason. Splitting into separate tasks would require a service mesh or an internal ALB for the proxy — extra parts for no gain.

The three containers share task CPU/memory but the cost is measured; see Sizing below.

---

## Task-def shape

```yaml
# Illustrative — real declaration is in internal/stacks/fargate.go
family: glmocr-cpu
cpu: "8192"              # 8 vCPU
memory: "16384"          # 16 GB (matches g4dn.2xlarge CPU side)
networkMode: awsvpc
requiresCompatibilities: [FARGATE]
runtimePlatform:
  operatingSystemFamily: LINUX
  cpuArchitecture: X86_64
executionRoleArn: <execution-role>
taskRoleArn:      <task-role>
containerDefinitions:
  - name: cpu
    image: <ecr>/glmocr-cpu:<commit-sha>
    essential: true
    cpu: 7168            # 7 vCPU — leave 0.5 each for the two sidecars
    memory: 14336        # 14 GB
    portMappings: [{containerPort: 5002, protocol: tcp}]
    environment:         # from SSM params (see 07-secrets-and-config.md)
      # Gunicorn / CPU sizing
      CPU_WORKERS:   "4"
      CPU_THREADS:   "16"
      GUNICORN_TIMEOUT: "480"
      GLMOCR_PORT: "5002"
      OMP_NUM_THREADS: "1"
      MKL_NUM_THREADS: "1"
      # OCR fan-out
      OCR_MAX_WORKERS: "32"
      OCR_CONNECT_TIMEOUT: "10"
      OCR_REQUEST_TIMEOUT: "60"
      OCR_RETRY_MAX: "2"
      OCR_RETRY_BACKOFF_BASE: "0.5"
      OCR_RETRY_BACKOFF_MAX: "8"
      OCR_CONN_POOL: "2048"
      OCR_MODEL_NAME: "glm-ocr"
      # SGLang target — stays local because of sigv4-proxy
      SGLANG_HOST: "127.0.0.1"
      SGLANG_PORT: "30000"
      SGLANG_SCHEME: "http"
      # Layout pipeline
      LAYOUT_ENABLED: "true"
      LAYOUT_DEVICE: "cpu"
      LAYOUT_USE_POLYGON: "false"
      LAYOUT_BACKEND: "onnx"
      LAYOUT_ONNX_THREADS: "2"
      LAYOUT_POSTPROC: "numpy"
      LAYOUT_GRAPH: "raw"
      LAYOUT_COMPILE: "false"
      LAYOUT_BATCH_ENABLED: "true"
      LAYOUT_BATCH_MAX: "8"
      LAYOUT_BATCH_WINDOW_MS: "20"
      # Mirror of the SGL caps for dashboard gauges (read-only in CPU)
      SGL_MAX_RUNNING_REQUESTS: "64"
      SGL_MAX_TOTAL_TOKENS: "200000"
      SGL_MAX_PREFILL_TOKENS: "8192"
      SGL_CHUNKED_PREFILL_SIZE: "8192"
      GLMOCR_PIPELINE_METRICS: "true"
    secrets:
      - name: HF_TOKEN
        valueFrom: <secrets-manager-arn>:glmocr/prod/hf_token:SecretString
    healthCheck:
      command: ["CMD-SHELL", "curl -fsS http://localhost:5002/health || exit 1"]
      interval: 10
      timeout: 5
      retries: 3
      startPeriod: 60
    dependsOn:
      - containerName: sigv4-proxy
        condition: HEALTHY
    logConfiguration:
      logDriver: awslogs
      options:
        awslogs-group: /glmocr/prod/cpu
        awslogs-region: <region>
        awslogs-stream-prefix: cpu

  - name: sigv4-proxy
    image: public.ecr.aws/aws-observability/aws-sigv4-proxy:latest
    essential: true
    cpu: 512             # 0.5 vCPU
    memory: 1024         # 1 GB
    portMappings: [{containerPort: 30000, protocol: tcp}]
    # See 05-sigv4-proxy-sidecar.md for the full command + env
    command:
      - "-v"
      - "--name=sagemaker"
      - "--region=<region>"
      - "--host=runtime.sagemaker.<region>.amazonaws.com"
      - "--port=:30000"
      - "--unsigned-payload"
    healthCheck:
      command: ["CMD-SHELL", "wget -q -O- localhost:30000/healthz || exit 1"]
      interval: 15
      timeout: 5
      retries: 3
      startPeriod: 30
    logConfiguration:
      logDriver: awslogs
      options:
        awslogs-group: /glmocr/prod/sigv4-proxy
        awslogs-region: <region>
        awslogs-stream-prefix: sigv4

  - name: alloy
    image: grafana/alloy:latest   # pin a tag in real CDK; see 06-observability-prod.md
    essential: false              # a scraping miss shouldn't kill inference
    cpu: 512
    memory: 1024
    portMappings: [{containerPort: 12345, protocol: tcp}]
    environment:
      ALLOY_CONFIG: "/etc/alloy/config.alloy"
    secrets:
      - name: GRAFANA_CLOUD_PROM_URL
        valueFrom: <arn>:glmocr/prod/grafana_cloud:SecretString:url::
      - name: GRAFANA_CLOUD_PROM_USER
        valueFrom: <arn>:glmocr/prod/grafana_cloud:SecretString:user::
      - name: GRAFANA_CLOUD_PROM_TOKEN
        valueFrom: <arn>:glmocr/prod/grafana_cloud:SecretString:token::
    command:
      - "run"
      - "--server.http.listen-addr=0.0.0.0:12345"
      - "--stability.level=generally-available"
      - "/etc/alloy/config.alloy"
    # mountPoints: config.alloy is baked in via a tiny sidecar-image build
    # (see 06-observability-prod.md)
    logConfiguration:
      logDriver: awslogs
      options:
        awslogs-group: /glmocr/prod/alloy
        awslogs-region: <region>
        awslogs-stream-prefix: alloy
```

---

## Env var translation table (dev `.env` → prod task-def)

Every knob in `.env` maps to one of: task-def `environment:` entry, task-def `secrets:` entry, or the SageMaker endpoint's `Environment` (from `03-sagemaker-sglang-byoc.md`).

| `.env` key | Destination | Notes |
|---|---|---|
| `CPU_WORKERS` … `MKL_NUM_THREADS` | task `cpu.environment` | From SSM `/glmocr/prod/cpu/*` |
| `OCR_*`, `LAYOUT_*`, `GLMOCR_*` | task `cpu.environment` | Same pattern |
| `SGLANG_HOST` / `SGLANG_PORT` / `SGLANG_SCHEME` | task `cpu.environment` | **Hardcoded to `127.0.0.1:30000`** — do NOT read from SSM. Leaves no path to accidentally point at the real SageMaker URL (would break SigV4). |
| `HF_TOKEN` | task `cpu.secrets` | Used by glmocr for HF model-artifact lookups (layout ONNX export etc.) |
| `SGL_*` | SageMaker endpoint `Environment` (see `03-sagemaker-sglang-byoc.md`) | Not on the task |
| `SGL_MAX_RUNNING_REQUESTS`, `SGL_MAX_TOTAL_TOKENS`, `SGL_MAX_PREFILL_TOKENS`, `SGL_CHUNKED_PREFILL_SIZE` | task `cpu.environment` **too** | Mirror only, for the `glmocr_config_sgl_*` dashboard gauges. Kept in sync manually with the SageMaker side. |
| `GRAFANA_CLOUD_PROM_URL/USER/TOKEN` | task `alloy.secrets` | Single composite secret with three JSON fields |
| `GRAFANA_CLOUD_PROM_URL_READ`, `GRAFANA_URL`, `GRAFANA_SA_TOKEN` | NOT in task | Only used by local dashboards + Grafana datasource (set in Grafana Cloud UI, not here) |

Canonical source: `reference/env-tuned.md` — each value there is the prod-ready default.

---

## IAM

### Task role (runtime)

```go
taskRole := iam.NewRole(stack, "glmocr-cpu-task-role", ...)
sm.GrantInvoke(taskRole)                              // sagemaker:InvokeEndpoint on glmocr-sglang
hfSecret.GrantRead(taskRole)                          // Secrets Manager read
grafanaSecret.GrantRead(taskRole)
ssm.Policy...GrantRead(taskRole)                      // GetParameters on /glmocr/prod/*
logs.GrantPutLogEvents(taskRole)
```

### Execution role (bootstrap)

Standard AWS managed policy `AmazonECSTaskExecutionRolePolicy`, plus `secretsmanager:GetSecretValue` for the secret ARNs listed in the task-def's `secrets:` (because ECS does the secret injection at container start, not the application).

---

## Sizing

Dev hardware was a Ryzen 5 5600X (12-thread Zen 3, AVX2). `g4dn.2xlarge` CPU side is Xeon Platinum 8259CL (8 vCPU, Cascade Lake). Both have AVX2; the Xeon adds AVX-512 VNNI but nothing in the current pipeline uses it (see `project_doclayout_yolo_probe` memory).

**Recommendation:** start at **8 vCPU / 16 GB** to match the g4dn CPU sizing the `.env` was tuned for. If load tests on prod hardware show room, bump to 16 vCPU (2× `CPU_WORKERS`, keep `LAYOUT_ONNX_THREADS=2` — scaling this is a separate experiment, see `09-runbook.md`).

Container-level split within the task:
- `cpu`: 7 vCPU / 14 GB — matches the `CPU_WORKERS=4 × LAYOUT_ONNX_THREADS=2 = 8 math threads` math.
- `sigv4-proxy`: 0.5 vCPU / 1 GB — it's a dumb forwarder.
- `alloy`: 0.5 vCPU / 1 GB — 5-second scrape cadence over one target is cheap.

---

## Autoscaling

```go
scalable := service.AutoScaleTaskCount(&awsapplicationautoscaling.EnableScalingProps{
    MinCapacity: jsii.Number(1),
    MaxCapacity: jsii.Number(4),
})
scalable.ScaleOnCpuUtilization(jsii.String("cpu-target"), &awsecs.CpuUtilizationScalingProps{
    TargetUtilizationPercent: jsii.Number(60),
    ScaleInCooldown:  awscdk.Duration_Minutes(jsii.Number(5)),
    ScaleOutCooldown: awscdk.Duration_Minutes(jsii.Number(1)),
})
// Also scale on ALB RequestCountPerTarget if CPU stays flat under burst
scalable.ScaleOnRequestCount(jsii.String("rps-target"), &awsecs.RequestCountScalingProps{
    TargetValue:           jsii.Number(200), // ~baseline p50 throughput × headroom
    AlbTargetGroup:        tg,
    ScaleInCooldown:       awscdk.Duration_Minutes(jsii.Number(5)),
    ScaleOutCooldown:      awscdk.Duration_Minutes(jsii.Number(1)),
})
```

**Why CPU target 60% (not 80%).** Layout residence time degrades non-linearly above ~70% CPU (measured in dev: `c=40` in `loadtest/results/omnidoc-20260423-*` regresses sharply). 60% keeps latency predictable.

---

## ALB

- **Internal** (`scheme: internal`) — no public listener. Frontend/API Gateway sits in front of this for auth.
- Target group: IP target type (Fargate awsvpc mode). Health check path `/health`, 5s interval, 2/10 healthy/unhealthy thresholds.
- HTTP on 5002; TLS terminates upstream (at the public edge) because ALB → Fargate is intra-VPC.

---

## Healthcheck strategy

Three layers:

1. **ALB health check** — `GET /health` on the `cpu` container. Controls traffic routing.
2. **ECS task healthcheck** (`cpu` container above) — same endpoint, controls task replacement.
3. **`cpu` container `dependsOn: sigv4-proxy HEALTHY`** — CPU won't start accepting until the proxy is up. Otherwise early OCR calls 404 because there's no one listening on 30000.

No healthcheck pings SageMaker directly — that would couple CPU liveness to endpoint availability. If SageMaker is down, the Fargate task is still healthy (Flask alive), requests fail with a clear upstream error, and CloudWatch alarms page on `SageMakerInvocationErrors`, not on ECS task health.

---

## What's different vs. dev

| Concern | Dev | Prod |
|---|---|---|
| `SGLANG_HOST` | `sglang` (compose DNS) | `127.0.0.1` (sidecar proxy) |
| Port 30000 binding | sglang container | sigv4-proxy sidecar |
| HF_HOME bind mount | `./hf-cache` | None — layout ONNX is exported fresh per task boot (cost: ~10 s one-time) |
| OmniDocBench dataset | `./datasets:/app/datasets:ro` | None in prod — loadtest runs from a build-box with the dataset, hitting the prod ALB |
| Prometheus scrape source | central Prometheus in compose | Alloy sidecar only |
| Observability spread | 6 compose services | Alloy sidecar + CloudWatch + Grafana Cloud |

---

Next: [`05-sigv4-proxy-sidecar.md`](./05-sigv4-proxy-sidecar.md).
