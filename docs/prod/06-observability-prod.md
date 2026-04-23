# 06 — Observability in production

**Purpose:** port the dev observability stack (Prometheus + Alloy + local Grafana) to prod without losing fidelity on the existing dashboards.

---

## What changes, what stays

| Component | Dev | Prod |
|---|---|---|
| Prometheus (local, 6h retention) | in docker-compose | **removed** — we rely on Grafana Cloud Mimir end-to-end |
| Alloy (central, scrapes compose services) | in docker-compose | **moves** — runs as a sidecar in each Fargate task |
| Grafana (local) | in docker-compose | **removed** — use Grafana Cloud UI at `dido.grafana.net` |
| Grafana Cloud Mimir (remote_write target) | optional, already wired | **primary** metrics store |
| CloudWatch | not used | **added** for SageMaker endpoint, ALB, ECS events |
| `glmocr-load.json` dashboard | bind-mounted into local Grafana | imported into Grafana Cloud (same JSON) |
| Cardinality allowlist | in `docker/alloy/config.alloy` | copied verbatim into the prod Alloy sidecar image |

**The key insight:** the dev stack already does remote_write to Grafana Cloud. Prod is *subtractive* — we strip out the local Prometheus/Grafana and push more aggressively on Grafana Cloud + CloudWatch. The dashboards don't move; their datasource stays Grafana Cloud Mimir.

---

## Why Alloy as a sidecar (not ECS SD)

Fargate tasks have short-lived private IPs. Options for scraping:

1. **Central Prometheus with ECS service discovery.** Requires a central Prometheus instance, plus a service-discovery adapter (PromECS). Extra infra. Hairy IAM.
2. **Push-based with pushgateway.** Works for short-lived jobs but long-running services can't reliably push gauges.
3. **Alloy as a sidecar, scraping `localhost:5002/metrics`, remote-writing to Grafana Cloud.** Zero service discovery. Each task ships its own metrics. Labels include `task_id` so per-task breakdowns still work.

Option 3 wins on simplicity. The compose-era `docker/alloy/config.alloy` is already 90% what we need — just change the scrape target from `cpu:5002` to `127.0.0.1:5002`.

---

## Alloy sidecar image

Build a tiny image that bakes the config (so we don't depend on a mounted volume, which Fargate makes awkward):

```dockerfile
# docker/alloy/Dockerfile
FROM grafana/alloy:v1.9.0
COPY config.alloy /etc/alloy/config.alloy
```

Push to the same ECR as the CPU image (or a separate `glmocr-alloy` repo).

### `config.alloy` (adapted from dev)

```alloy
// Scrape the CPU container on task-local loopback.
prometheus.scrape "cpu_app" {
  targets = [{
    "__address__" = "127.0.0.1:5002",
    "job"         = "glmocr_cpu",
  }]
  scrape_interval = "5s"
  metrics_path    = "/metrics"
  forward_to      = [prometheus.relabel.cardinality_cap.receiver]
}

// Remote-write to Grafana Cloud Mimir.
prometheus.remote_write "grafana_cloud" {
  endpoint {
    url = env("GRAFANA_CLOUD_PROM_URL")

    basic_auth {
      username = env("GRAFANA_CLOUD_PROM_USER")
      password = env("GRAFANA_CLOUD_PROM_TOKEN")
    }
  }

  // Add task-level labels so we can slice per-task on dashboards.
  external_labels = {
    env        = "prod",
    service    = "glmocr-cpu",
    region     = env("AWS_REGION"),
    task_id    = env("ECS_TASK_ID"),       // injected via task-def
  }
}

// Cardinality allowlist — drop anything not explicitly listed.
// Ported verbatim from dev. Guards against blowing the 10k free-tier cap.
prometheus.relabel "cardinality_cap" {
  forward_to = [prometheus.remote_write.grafana_cloud.receiver]

  rule {
    source_labels = ["__name__"]
    regex         = "glmocr_.*|process_.*|python_.*|http_.*_seconds|http_.*_total|layout_.*|ocr_.*"
    action        = "keep"
  }
}
```

`ECS_TASK_ID` comes from the ECS metadata endpoint. Inject via a small startup wrapper or let Alloy read `AWS_ECS_CONTAINER_METADATA_URI_V4` (Fargate auto-sets it).

---

## SageMaker metrics → CloudWatch

SageMaker real-time endpoints emit CloudWatch metrics **automatically**:

- `Invocations` — count
- `InvocationsPerInstance` — per-instance load
- `ModelLatency` — request-side latency (ms)
- `OverheadLatency` — SageMaker routing overhead (ms)
- `CPUUtilization`, `GPUUtilization`, `MemoryUtilization`, `GPUMemoryUtilization` — instance health
- `Invocation4XXErrors`, `Invocation5XXErrors` — HTTP error counts

These replace the per-phase histograms the dev repo got from scraping SGLang's `/metrics`. If you want SGLang's own histograms (queue wait, TTFT, inter-token latency), add an Alloy sidecar **inside the SageMaker container** that scrapes `localhost:30000/metrics` and remote-writes to Grafana Cloud. This is a second-phase add — not in the first deploy — because SageMaker containers are more awkward to modify on a hot path.

### Cross-visualize in Grafana Cloud

Add a CloudWatch datasource to Grafana Cloud (Settings → Data sources → CloudWatch). Authentication via an IAM role that Grafana Cloud assumes — the CDK `ObservabilityStack` creates this role with `cloudwatch:GetMetricData`, trust-policy scoped to Grafana Cloud's external ID.

Once added, build a new Grafana dashboard (or add panels to `glmocr-load.json`) that combines:

- Fargate `/metrics` histograms (from Alloy → Mimir) for the CPU phase
- CloudWatch SageMaker metrics for the GPU phase

---

## Porting `glmocr-load.json`

```bash
# On the MacBook, after CDK deploy is green:
aws s3 cp docker/grafana/dashboards/glmocr-load.json - | \
    curl -X POST https://dido.grafana.net/api/dashboards/db \
    -H "Authorization: Bearer $GRAFANA_SA_TOKEN" \
    -H "Content-Type: application/json" \
    --data-binary @-
```

Update the datasource UID inside the JSON first (`grafanacloud-<name>-prom` instead of the local `prometheus`). One-time find-replace.

---

## What we lose vs. dev

- **Sweep dashboard (`glmocr-sweep-progress.json`)** — was for local multi-config sweeps. Not relevant in prod; we don't sweep in prod.
- **Grafana annotations from the matrix script** — the compose harness wrote annotations to local Grafana's SQLite. Prod matrix runs (the smoke step in CI) push to Grafana Cloud via the same `/api/annotations` endpoint. Auth via the SA token already in `.env`.
- **Alloy `/debug` UI** — Fargate ports aren't exposed publicly. Exec into the task with `ecs exec` (enable in the service) when you need to debug Alloy.

---

## CloudWatch alarms (first wave)

Minimum viable page-worthy alarms (set up in `ObservabilityStack`):

| Alarm | Threshold | Why |
|---|---|---|
| SM endpoint `Invocation5XXErrors` | > 5 in 1 min | Backend is failing |
| SM endpoint `ModelLatency p95` | > 30 s | Near the CPU-side `OCR_REQUEST_TIMEOUT=60` |
| Fargate task `CPUUtilization` | > 85% for 10 min | Saturating; autoscale should be catching this |
| Fargate task `MemoryUtilization` | > 85% for 5 min | OOM imminent |
| ALB `HTTPCode_Target_5XX_Count` | > 10 in 5 min | Downstream failing at scale |
| ALB `UnHealthyHostCount` | >= 1 for 2 min | A task has dropped out |
| Alloy's own `remote_write_pending_samples` | > 100k sustained | Grafana Cloud push is falling behind |

All pages go to a single PagerDuty / OpsGenie / SNS topic — no alarm spam, one pager line per incident.

---

## Runbook cross-ref

`09-runbook.md` has the "what to do when this alarm fires" decision tree. Keep alarms here (observability config) and responses there (ops), even if they duplicate names — they change at different cadences.

---

Next: [`07-secrets-and-config.md`](./07-secrets-and-config.md).
