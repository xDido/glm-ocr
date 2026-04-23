# 02 — Go CDK project structure

**Purpose:** the layout and conventions for the AWS CDK (Go) infrastructure repo that deploys everything in `01-prod-architecture.md`.

---

## Repo layout

```
<repo-root>/
├── CLAUDE.md                          ← from 11-CLAUDE.md.template
├── README.md
├── go.mod                             ← module: github.com/<org>/glmocr-prod
├── go.sum
├── cdk.json                           ← CDK entrypoint: "app": "go mod download && go run cmd/app/main.go"
├── cdk.context.json                   ← gitignored; stores CDK environment lookups
│
├── cmd/
│   └── app/
│       └── main.go                    ← CDK app entrypoint; instantiates all stacks
│
├── internal/
│   ├── config/
│   │   └── config.go                  ← loads env-per-stage config (dev/staging/prod)
│   └── stacks/
│       ├── network.go                 ← NetworkStack: VPC, subnets, endpoints
│       ├── ecr.go                     ← EcrStack: 2 repositories
│       ├── secrets.go                 ← SecretsStack: SM + SSM params
│       ├── sagemaker.go               ← SagemakerStack: model, endpoint config, endpoint
│       ├── fargate.go                 ← FargateStack: cluster, task def, service, ALB, autoscaling
│       └── observability.go           ← ObservabilityStack: log groups, CW alarms, Grafana SA role
│
├── docker/
│   ├── cpu/                           ← built from scratch per 12-cpu-container-spec.md
│   │   ├── Dockerfile
│   │   ├── entrypoint.sh
│   │   ├── wsgi.py
│   │   ├── gunicorn_conf.py
│   │   ├── config.yaml.template
│   │   └── config.layout-off.template
│   ├── sglang/                        ← built from scratch per 03-sagemaker-sglang-byoc.md
│   │   ├── Dockerfile
│   │   ├── serve.py
│   │   ├── entrypoint.sh
│   │   └── requirements.txt
│   ├── alloy/                         ← built from scratch per 06-observability-prod.md
│   │   ├── Dockerfile
│   │   └── config.alloy
│   └── sigv4-proxy/                   ← optional Go fallback per 05-sigv4-proxy-sidecar.md
│       ├── Dockerfile
│       ├── go.mod
│       └── main.go
│
├── scripts/                           ← built from scratch per 13-loadtest-spec.md
│   ├── bake-weights.sh                ← downloads GLM-OCR from HF, tars, uploads to S3
│   ├── smoke.sh                       ← post-deploy: curl /health, POST sample doc
│   ├── omnidoc_asyncio_matrix.sh      ← multi-c matrix runner
│   ├── test_images/                   ← small sample images for smoke + matrix
│   └── lib/
│       └── loadtest_common.sh
│
├── loadtest/                          ← built from scratch per 13-loadtest-spec.md
│   ├── asyncio/
│   │   └── bench.py
│   └── results/                       ← matrix reports land here
│
├── docs/
│   ├── prod/                          ← this whole handoff package
│   ├── ARCHITECTURE.md                ← copied from dev repo (referenced)
│   └── OPTIMIZATIONS.md               ← copied from dev repo (referenced)
│
└── .github/
    └── workflows/
        ├── cdk-diff.yml
        ├── cdk-deploy.yml
        ├── build-cpu.yml
        └── build-sglang.yml
```

---

## Stack boundaries

Six stacks. Boundaries were chosen so each stack's deploy time stays under ~3 minutes, and so a blast-radius problem in one is contained.

| Stack | Owns | Depends on | Redeploy impact |
|---|---|---|---|
| `NetworkStack` | VPC, 2 public + 2 private subnets, NAT gateway, VPC endpoints (sagemaker-runtime, s3, ecr.api, ecr.dkr, secretsmanager, logs, ssm) | — | Rare; touches long-lived network. |
| `EcrStack` | 2 ECR repos: `glmocr-cpu`, `glmocr-sglang`. Lifecycle policies (keep last 20 images). | — | None (just the registry). |
| `SecretsStack` | Secrets Manager secrets, SSM Parameter Store params | — | Rotation only; no service restart required. |
| `SagemakerStack` | Model, EndpointConfig, Endpoint | `EcrStack`, `SecretsStack`, `NetworkStack` (for endpoint-in-VPC optional) | 8–15 min redeploy (instance recreation). Blue-green via `DeploymentConfig`. |
| `FargateStack` | ECS Cluster, Task Def (3 containers), Service, Internal ALB, autoscaling policies | `EcrStack`, `SecretsStack`, `NetworkStack`, `SagemakerStack` (consumes endpoint name) | ~60 s rolling deploy (task-def revision). |
| `ObservabilityStack` | CloudWatch log groups, alarms, Grafana IAM role for CloudWatch datasource | `FargateStack`, `SagemakerStack` | None (alarm-only). |

---

## `cmd/app/main.go` skeleton

```go
package main

import (
    "github.com/aws/aws-cdk-go/awscdk/v2"
    "github.com/aws/jsii-runtime-go"
    "github.com/<org>/glmocr-prod/internal/config"
    "github.com/<org>/glmocr-prod/internal/stacks"
)

func main() {
    defer jsii.Close()
    app := awscdk.NewApp(nil)

    cfg := config.LoadFromContext(app)   // reads --context stage=prod

    net := stacks.NewNetworkStack(app, "glmocr-network-"+cfg.Stage, cfg)
    ecr := stacks.NewEcrStack(app, "glmocr-ecr-"+cfg.Stage, cfg)
    sec := stacks.NewSecretsStack(app, "glmocr-secrets-"+cfg.Stage, cfg)
    sm  := stacks.NewSagemakerStack(app, "glmocr-sagemaker-"+cfg.Stage, cfg, stacks.SagemakerInputs{
        Network: net, Ecr: ecr, Secrets: sec,
    })
    fg  := stacks.NewFargateStack(app, "glmocr-fargate-"+cfg.Stage, cfg, stacks.FargateInputs{
        Network: net, Ecr: ecr, Secrets: sec, Sagemaker: sm,
    })
    stacks.NewObservabilityStack(app, "glmocr-obs-"+cfg.Stage, cfg, stacks.ObservabilityInputs{
        Fargate: fg, Sagemaker: sm,
    })

    app.Synth(nil)
}
```

`cfg` is a plain struct: `Stage`, `AccountID`, `Region`, `CpuImageTag`, `SglangImageTag`, `InstanceType=ml.g4dn.2xlarge`, `CpuCount=1`. Loaded from `cdk.json` context or `--context` flags.

---

## Cross-stack references

Pass **typed constructs**, not strings. e.g., `sagemaker.go` exports `endpointName string` and a `grantInvoke(role iam.IRole)` helper. `fargate.go` calls `sm.GrantInvoke(taskRole)` instead of manually attaching an `ecs:InvokeEndpoint` policy.

This keeps IAM least-privilege automatic and spots wiring errors at `cdk synth` time, not at deploy time.

---

## Parameter strategy

| Value | Source | Why |
|---|---|---|
| Account ID, region | CDK context (`cdk.json` or CLI `--context`) | Per-stage override without code change |
| Image tags | CDK context (`--context cpuTag=sha-abc123`) | CI sets this on deploy; ties image to commit |
| Tuned knobs (SGL_*, OCR_*, LAYOUT_*) | SSM Parameter Store (`/glmocr/prod/*`) | Hot-patchable without code change; see `07-secrets-and-config.md` |
| Rotatable secrets (HF_TOKEN, GRAFANA_*) | Secrets Manager | Actual secrets; rotation without CDK redeploy |
| Instance type, task CPU/memory | CDK context | Governed by a code review, not a console edit |
| Autoscaling bounds (min/max tasks) | CDK context | Reviewable change; rollback is a revert |

Rule of thumb: if a human might want to tweak it during an incident, it's an SSM param. If it requires a code review to change, it's CDK context.

---

## Commands you'll run

```bash
# Bootstrap the account once (per region)
cdk bootstrap aws://<account>/<region> --context stage=prod

# Iterate
cdk synth     --context stage=prod
cdk diff      --context stage=prod
cdk deploy    --context stage=prod glmocr-fargate-prod      # single stack
cdk deploy    --context stage=prod --all --require-approval never   # full deploy (CI)

# Destroy (only for dev stages; prod should be destroy-protected)
cdk destroy --context stage=dev --all
```

Enable destroy-protection on the prod stacks in code:

```go
stack.AddTerminationProtection(jsii.Bool(cfg.Stage == "prod"))
```

---

## Testing the CDK

Use `go test` + the CDK Go assertions library:

```go
func TestFargateTaskHasThreeContainers(t *testing.T) {
    app := awscdk.NewApp(nil)
    stack := stacks.NewFargateStack(app, "test", cfg, testInputs())
    template := assertions.Template_FromStack(stack, nil)

    template.HasResourceProperties(jsii.String("AWS::ECS::TaskDefinition"),
        map[string]any{"ContainerDefinitions": assertions.Match_ArrayWith(&[]any{
            map[string]any{"Name": "cpu"},
            map[string]any{"Name": "sigv4-proxy"},
            map[string]any{"Name": "alloy"},
        })})
}
```

One test per stack covering the **shape invariants** the prod repo cares about: 3 containers in the task, the Fargate task role has `sagemaker:InvokeEndpoint`, the SageMaker endpoint is `InService` after synth (metadata-only), the ALB is internal-scheme. Keep tests short — the CDK Go API is verbose enough that over-testing turns into test-maintenance drag.

---

Next: [`03-sagemaker-sglang-byoc.md`](./03-sagemaker-sglang-byoc.md).
