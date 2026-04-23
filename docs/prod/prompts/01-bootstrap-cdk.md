# Prompt 01 — Bootstrap the Go CDK project

> Copy-paste the block below into Claude Code on the new MacBook. Expect ~30 min.

---

```
Bootstrap the Go CDK project for glmocr-prod.

Prerequisites (check before starting):
  - go version       # 1.23+
  - cdk --version    # 2.x
  - aws sts get-caller-identity   # valid AWS creds

Read first:
  - docs/prod/02-cdk-go-structure.md

Tasks:

1. Initialize `go.mod`:
     go mod init github.com/<org>/glmocr-prod
   Ask me for <org> if not obvious. Then `go get github.com/aws/aws-cdk-go/awscdk/v2`
   and `go get github.com/aws/constructs-go/constructs/v10`.

2. Create `cdk.json`:
     {
       "app": "go mod download && go run cmd/app/main.go",
       "context": {
         "stages": {
           "prod": {
             "region": "<REPLACE>",
             "account": "<REPLACE>",
             "cpuImageTag": "latest",
             "sglangImageTag": "latest",
             "sagemakerInstance": "ml.g4dn.2xlarge",
             "sagemakerMinInstances": 1,
             "autoscaleMin": 1,
             "autoscaleMax": 4
           }
         }
       }
     }
   Ask me to fill the <REPLACE> slots before continuing.

3. Create `cmd/app/main.go` following the skeleton in docs/prod/02-cdk-go-structure.md.
   Instantiate all six stack constructors BUT leave the implementations as stubs
   that just create an empty Stack (we'll fill them in later prompts).

4. Create `internal/config/config.go` with a `Config` struct matching the cdk.json
   context (`Stage`, `AccountID`, `Region`, `CpuImageTag`, `SglangImageTag`,
   `SagemakerInstance`, etc.) and a `LoadFromContext(app)` function that reads
   the `--context stage=<name>` value and returns the matching substructure.

5. Create `internal/stacks/{network,ecr,secrets,sagemaker,fargate,observability}.go`
   each with an empty `NewXStack(...)` factory that returns an `awscdk.IStack`.

6. Add `.gitignore` for: `cdk.out/`, `cdk.context.json`, `.DS_Store`, `node_modules/`,
   `*.out`, `vendor/`.

7. Run:
     cdk synth --context stage=prod
   It must emit six (empty) CloudFormation templates without error.

8. Bootstrap the account (one-time; only if not already done):
     cdk bootstrap aws://<account>/<region>
   Expect ~5 minutes.

9. Commit as "chore: bootstrap Go CDK project skeleton".

At the end, give me a 3-bullet summary of what's in the repo, and confirm
`cdk synth` passes. Do NOT deploy anything.
```
