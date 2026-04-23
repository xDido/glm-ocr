# 07 — Secrets & config

**Purpose:** replace `.env` with AWS-native stores so rotation and hot-patches don't require redeploys, and secrets never land in source control.

---

## The split

| Class | Store | Rotation | Example |
|---|---|---|---|
| Secret (sensitive, rotatable) | Secrets Manager | automatic or manual | `HF_TOKEN`, `GRAFANA_CLOUD_PROM_TOKEN`, `GRAFANA_SA_TOKEN` |
| Non-secret tunable (reviewable) | SSM Parameter Store (String) | manual | `SGL_MEM_FRACTION_STATIC`, `OCR_MAX_WORKERS`, `LAYOUT_ONNX_THREADS` |
| Structural (code-owned) | CDK context (`cdk.json`) | git commit | Instance type, min/max task count, region |

Rule: if it **could change during an incident** and a human should be able to flip it without a PR, it's SSM. If it's **a secret**, Secrets Manager. If **changing it requires a code review**, CDK context.

---

## Secrets Manager layout

Three secrets, all under the `glmocr/prod/` prefix:

| Secret name | Schema | Consumed by |
|---|---|---|
| `glmocr/prod/hf_token` | `SecretString: <token>` (raw) | cpu container `HF_TOKEN` env |
| `glmocr/prod/grafana_cloud` | JSON: `{url, user, token}` | alloy sidecar `GRAFANA_CLOUD_PROM_*` envs |
| `glmocr/prod/grafana_sa` | `SecretString: <token>` | CI (dashboard import step) |

ECS task-def references them via `secrets:` blocks. JSON-key extraction syntax example:

```yaml
- name: GRAFANA_CLOUD_PROM_URL
  valueFrom: arn:aws:secretsmanager:<region>:<acct>:secret:glmocr/prod/grafana_cloud-xxxx:url::
```

The `:url::` suffix extracts the `url` field from the JSON secret.

### Rotation

- **HF_TOKEN:** rotate manually in HuggingFace, then update the secret in Secrets Manager console; Fargate `--force-new-deployment` picks up the new value in ~60 s.
- **Grafana Cloud token:** rotated via the Grafana Cloud UI → update Secrets Manager → redeploy.
- **Grafana SA token:** rotated per the Grafana team's cadence → update Secrets Manager.

Automatic Lambda-based rotation (via Secrets Manager's rotation schedules) is overkill for these; the manual path is 2 minutes per rotation.

---

## SSM Parameter Store layout

One parameter per knob. Type `String` (not `SecureString` — nothing here is sensitive). Tier `Standard` (no throughput need).

**Path scheme:**

```
/glmocr/prod/cpu/CPU_WORKERS
/glmocr/prod/cpu/OMP_NUM_THREADS
/glmocr/prod/cpu/OCR_MAX_WORKERS
/glmocr/prod/cpu/LAYOUT_BATCH_ENABLED
...
/glmocr/prod/sgl/SGL_MAX_RUNNING_REQUESTS
/glmocr/prod/sgl/SGL_CONTEXT_LENGTH
/glmocr/prod/sgl/SGL_SPECULATIVE
/glmocr/prod/sgl/SGL_MEM_FRACTION_STATIC
/glmocr/prod/sgl/SGL_SPEC_ALGORITHM
...
/glmocr/prod/sagemaker/model_data_key
/glmocr/prod/sagemaker/endpoint_name
```

The **entire** `/cpu/*` subtree becomes the CPU container's env vars. The **entire** `/sgl/*` subtree becomes the SageMaker endpoint's `Environment`.

### CDK reads params at synth time

```go
// internal/stacks/secrets.go
func LoadKnobs(scope constructs.Construct, path string) map[string]string {
    out := make(map[string]string)
    // Iterate with CloudFormation custom resource OR read from a JSON file
    // committed to the repo and write-through into SSM via CDK.
    return out
}
```

**Two ways to wire this:**

**(a) Source of truth is SSM.** CDK *reads* SSM at synth (via `ssm.StringParameter.ValueFromLookup`) and injects as plain task-def env. Pro: console edits are authoritative. Con: CDK deploy is needed to pick up param changes, or use the "force deploy on SSM change" EventBridge pattern below.

**(b) Source of truth is a JSON file in git.** CDK *writes* the file's contents to SSM at deploy time, and also injects into task-def env directly. Pro: every knob change is a PR. Con: losing the hot-patch ability.

**Recommend (a)** for prod — the whole point of SSM is the operational dial. Add a lightweight drift check: CI runs `scripts/check-ssm-drift.sh` that warns if SSM values diverge from a tracked `reference/prod-knobs.json` without a recent commit referencing it.

### Hot-patch flow

```bash
# 1. Console or CLI: update the SSM param
aws ssm put-parameter \
    --name /glmocr/prod/cpu/OCR_MAX_WORKERS \
    --value 40 \
    --overwrite

# 2. Force a rolling deploy (picks up the new env on next task)
aws ecs update-service \
    --cluster glmocr-prod \
    --service glmocr-cpu \
    --force-new-deployment
```

Rolling-deploy time: ~60 seconds. Rollback: put the old value back and redeploy.

### EventBridge on SSM param change (optional)

If you want "change the knob → deploy automatically," wire an EventBridge rule on `Parameter Store Change` events → Lambda → `UpdateService --force-new-deployment`. Not in scope for the first CDK deploy; add after ops comfort level is there.

---

## cdk.json context (structural config)

```json
{
  "context": {
    "stages": {
      "prod": {
        "region":             "us-east-1",
        "account":            "123456789012",
        "cpuImageTag":        "sha-REPLACE",
        "sglangImageTag":     "sha-REPLACE",
        "cpuInstanceCpu":     "8192",
        "cpuInstanceMemory":  "16384",
        "sagemakerInstance":  "ml.g4dn.2xlarge",
        "sagemakerMinInstances": 1,
        "autoscaleMin":       1,
        "autoscaleMax":       4
      },
      "dev": { ... }
    }
  }
}
```

CI overrides `cpuImageTag` and `sglangImageTag` to the commit SHA on every deploy.

---

## IAM — least privilege

### Task role

```go
// internal/stacks/fargate.go, taskRole
taskRole.AddToPolicy(iam.NewPolicyStatement(&iam.PolicyStatementProps{
    Actions: &[]*string{
        jsii.String("secretsmanager:GetSecretValue"),
    },
    Resources: &[]*string{
        jsii.String(hfSecret.SecretFullArn()),
        jsii.String(grafanaSecret.SecretFullArn()),
    },
}))
taskRole.AddToPolicy(iam.NewPolicyStatement(&iam.PolicyStatementProps{
    Actions: &[]*string{
        jsii.String("ssm:GetParameters"),
        jsii.String("ssm:GetParametersByPath"),
    },
    Resources: &[]*string{
        jsii.String("arn:aws:ssm:<region>:<acct>:parameter/glmocr/prod/*"),
    },
}))
sagemakerEndpoint.GrantInvoke(taskRole)
```

**Not allowed:** `secretsmanager:CreateSecret`, `ssm:PutParameter`. The task can only read.

### SageMaker execution role

```go
s3WeightsBucket.GrantRead(sagemakerRole)
logs.GrantPutLogEvents(sagemakerRole)
```

Only needs S3 + CloudWatch. **No** internet egress, no HF Hub — weights come exclusively from S3.

### CI role (GitHub Actions, OIDC)

Trust policy: `token.actions.githubusercontent.com` + repo/ref conditions. Permissions: `ecr:*` on the two repos, `cloudformation:*` on stacks named `glmocr-*`, `iam:PassRole` on the three above roles, `s3:*` on the weights bucket path, `ssm:PutParameter` on `/glmocr/prod/*` (for drift-fix deploys only).

---

## What's NOT in secrets stores

| Value | Why not |
|---|---|
| AWS credentials | Use IAM roles (Fargate task role + IMDS) — never store AWS creds in SM/SSM |
| Region, account ID | CDK context — they're structural, not tunable |
| ALB URL | CloudFormation output — visible, not secret |
| SageMaker endpoint name | CDK cross-stack ref — not a free-text secret |
| HF model IDs | CDK context — picking a different model is a PR |

---

## Checklist for a new env var

Before adding a knob in code:

1. Does the existing dev `.env` have it? If yes, you're porting; add to the relevant SSM subtree (`/cpu/*` or `/sgl/*`) with the tuned default from `reference/env-tuned.md`.
2. Is it a secret? If yes, Secrets Manager; otherwise SSM.
3. Does it need hot-patchability? If yes, SSM. If no and it's a code-review-worthy decision, CDK context.
4. Update the task-def / endpoint-config in CDK to read it.
5. Add a line in `reference/env-tuned.md` describing its prod source of truth.

---

Next: [`08-ci-cd.md`](./08-ci-cd.md).
