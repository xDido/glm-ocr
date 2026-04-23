# Prompt 06 — ObservabilityStack + Alloy sidecar config

> Prerequisite: prompt 05 done (Fargate service deployed, ALB green). Expect ~30 min.

---

```
Build the Alloy sidecar image + ObservabilityStack (CloudWatch log groups, alarms,
Grafana Cloud datasource role).

Read first:
  - docs/prod/06-observability-prod.md (full)
  - docker/alloy/config.alloy (dev repo; carry over into this repo)

## Part A — Alloy sidecar image (docker/alloy/)

1. Create the directory:
     mkdir -p docker/alloy

2. Copy the dev config (adapted):
     # Copy the dev repo's docker/alloy/config.alloy first, then:
     # - change scrape target from cpu:5002 to 127.0.0.1:5002
     # - parameterize env labels with env("AWS_REGION") etc.
     # - use the cardinality allowlist verbatim
   See docs/prod/06-observability-prod.md for the template.

3. Create docker/alloy/Dockerfile:
     FROM grafana/alloy:v1.9.0
     COPY config.alloy /etc/alloy/config.alloy

4. Build + push to ECR (create a new repo glmocr-alloy if not already):
     aws ecr create-repository --repository-name glmocr-alloy
     docker build -t glmocr-alloy:$(git rev-parse --short HEAD) docker/alloy/
     # tag + push as in prompt 03

5. Update FargateStack: change the alloy container's image to the new ECR URI.
   Add the 3 GRAFANA_CLOUD_* env vars from the composite secret:

     secrets: [
       { name: "GRAFANA_CLOUD_PROM_URL",   valueFrom: grafana_cloud + ":url::" },
       { name: "GRAFANA_CLOUD_PROM_USER",  valueFrom: grafana_cloud + ":user::" },
       { name: "GRAFANA_CLOUD_PROM_TOKEN", valueFrom: grafana_cloud + ":token::" },
     ]

   Redeploy FargateStack.

6. Fill the grafana_cloud secret with real values (from the dev repo's .env):
     aws secretsmanager put-secret-value \
       --secret-id glmocr/prod/grafana_cloud \
       --secret-string '{"url":"https://prometheus-prod-....net/api/prom/push","user":"...","token":"..."}'

7. Verify: tail alloy's logs:
     aws logs tail /glmocr/prod/alloy --since 5m --follow
   Look for "scrape successful" and "remote_write: samples pushed".

## Part B — ObservabilityStack (internal/stacks/observability.go)

1. CloudWatch log groups (retention 14 days):
   - /glmocr/prod/cpu
   - /glmocr/prod/sigv4-proxy
   - /glmocr/prod/alloy
   - /aws/sagemaker/Endpoints/glmocr-sglang (already auto-created; reference for alarms)

2. Six alarms (see docs/prod/06-observability-prod.md §CloudWatch alarms):
   a. glmocr-sm-5xx-errors
   b. glmocr-sm-latency-high
   c. glmocr-fargate-cpu-high
   d. glmocr-fargate-memory-high
   e. glmocr-alb-5xx
   f. glmocr-alb-unhealthy-host

   Each wired to an SNS topic (create one: glmocr-prod-alerts) — the human
   subscribes their email/PagerDuty endpoint to it.

3. Grafana Cloud CloudWatch datasource IAM role (for cross-visualizing SM metrics
   in Grafana Cloud):
   - Role name: glmocr-prod-grafana-cw-reader
   - Trust policy: trust Grafana Cloud's account+external ID
     (Grafana Cloud UI → Data Sources → CloudWatch → "AWS IAM Role" displays
     these; hardcode after we pick them, or ask me to paste them)
   - Permissions: cloudwatch:GetMetricData, cloudwatch:GetMetricStatistics,
     cloudwatch:ListMetrics, tag:GetResources

4. Deploy:
     cdk deploy --context stage=prod glmocr-obs-prod

## Part C — Grafana dashboard import

1. Ask me for the Grafana SA token (or read it from
   `aws secretsmanager get-secret-value --secret-id glmocr/prod/grafana_sa`).

2. Update docker/grafana/dashboards/glmocr-load.json datasource UIDs to the
   Grafana Cloud Mimir UID (Settings → Data sources → lookup).

3. Import:
     curl -X POST "$GRAFANA_URL/api/dashboards/db" \
       -H "Authorization: Bearer $GRAFANA_SA_TOKEN" \
       -H "Content-Type: application/json" \
       --data-binary @docker/grafana/dashboards/glmocr-load.json

4. In the Grafana Cloud UI, open the dashboard. Confirm data is flowing on
   the main rps/latency panels. If empty: Alloy or scrape is broken.

Commit as "feat(observability): alloy sidecar + cloudwatch alarms + grafana cw datasource".

Report: 3 bullets on what's working, 1 bullet per alarm that's not in OK state.
```
