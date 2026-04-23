# Prompt 05 — FargateStack

> Prerequisite: prompt 04 done (SageMaker endpoint InService). Expect ~45 min.

---

```
Implement FargateStack in internal/stacks/fargate.go.

Read first:
  - docs/prod/04-fargate-cpu-task.md (full)
  - docs/prod/05-sigv4-proxy-sidecar.md (full)
  - docs/prod/07-secrets-and-config.md

Tasks:

## Part A — SecretsStack (do this first, in secrets.go)

1. Three secrets in Secrets Manager:
   - glmocr/prod/hf_token              — SecretString (just the raw token)
   - glmocr/prod/grafana_cloud         — JSON: {url, user, token}
   - glmocr/prod/grafana_sa            — SecretString (CI only)
   Create them with placeholder values; the human fills in real values via
   aws secretsmanager put-secret-value after this stack deploys.

2. Seed CPU knob SSM params under /glmocr/prod/cpu/*. Read reference/env-tuned.md
   for the full list. Use CfnParameter (one per knob, 3 lines each).

3. Export:
   - HfTokenSecret        (awssecretsmanager.ISecret)
   - GrafanaCloudSecret   (awssecretsmanager.ISecret)
   - CpuKnobs             map[string]awsssm.IStringParameter
   - GrantReadKnobs(role) helper

4. Deploy:
     cdk deploy --context stage=prod glmocr-secrets-prod
   Expect ~30 seconds.

## Part B — FargateStack

1. ECS cluster: glmocr-prod. Enable ecs exec (for debugging).

2. Task role + execution role:
   - Task role: grant reads on SecretsStack.*, grant reads on CPU knob SSM,
     call SagemakerStack.GrantInvoke(taskRole). Also logs:PutLogEvents.
   - Execution role: standard AmazonECSTaskExecutionRolePolicy + ECR read.

3. Task definition (Fargate, awsvpc, LINUX, X86_64):
   - cpu: 8192, memory: 16384
   - Three containers — see docs/prod/04-fargate-cpu-task.md for the FULL spec:
     a) cpu          — glmocr-cpu image + all CPU knobs from SSM + HF_TOKEN secret
     b) sigv4-proxy  — public.ecr.aws/aws-observability/aws-sigv4-proxy
                        command: per docs/prod/05-sigv4-proxy-sidecar.md
                        env: SM_ENDPOINT_NAME= <SagemakerStack.EndpointName>
     c) alloy        — (we're pushing the alloy sidecar in prompt 06; for now
                        use grafana/alloy:v1.9.0 with a placeholder config)

   Env var injection pattern:
     - Non-secret knobs:   environment: {"KEY": ssmParam.StringValue()}
     - Secrets (HF_TOKEN): secrets:     {"HF_TOKEN": Secret.FromSecretsManager(hf)}
     - Composite secret (grafana_cloud): use Secret.FromSecretsManagerWithField()

   DependsOn:
     - cpu dependsOn sigv4-proxy HEALTHY
     - alloy has no dependsOn (essential: false)

   Healthcheck on cpu: GET /health every 10s, start period 60s, 3 retries.

4. Internal ALB:
   - scheme: internal
   - listener: HTTP on 5002
   - target group: IP targets (awsvpc), health check path /health, healthy
     threshold 2, unhealthy 10, interval 5s, timeout 4s
   - SG: ingress 5002 from VPC CIDR

5. ECS service:
   - Cluster + task def above
   - Desired count: 1 (cfg.autoscaleMin)
   - launchType: FARGATE
   - assignPublicIp: false
   - Health check grace period: 120s (weights not loaded; 60s = CPU startup)
   - Attach to ALB target group

6. Autoscaling:
   - min/max per cfg
   - Target CPU utilization: 60%
   - Target ALB request count per target: 200

7. CloudFormation outputs:
   - AlbUrl           (http://<alb-dns>:5002)
   - ServiceArn
   - ClusterName
   - TaskDefArn

8. Tests:
   - Task def has 3 containers (names: cpu, sigv4-proxy, alloy)
   - cpu container has HF_TOKEN in secrets block
   - cpu container has SGLANG_HOST=127.0.0.1 (catch accidental public-URL
     misconfiguration that would break sigv4 signing)
   - task role has sagemaker:InvokeEndpoint

9. Deploy:
     cdk deploy --context stage=prod glmocr-fargate-prod
   Expect 3-5 min.

10. Smoke test (from your laptop, through SSM session manager or a bastion):
     curl -fsS http://<alb-dns>:5002/health
     # Should return {"status":"ok"}
     # Test an actual OCR call against a small PNG
     curl -fsS -X POST http://<alb-dns>:5002/glmocr/parse \
         -H 'content-type: application/json' \
         -d "$(jq -n --arg b64 "$(base64 -i sample.png)" '{"images":[("data:image/png;base64," + $b64)]}')"

Commit as "feat(fargate): ecs service with cpu + sigv4-proxy + alloy sidecars".

Report:
  - ALB URL
  - Smoke test result
  - Any error logs from cpu, sigv4-proxy, or alloy
```
