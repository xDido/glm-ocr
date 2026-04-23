# 08 — CI/CD (GitHub Actions)

**Purpose:** automate image builds, CDK diffs, and deploys so that shipping a change is "merge to main, wait ~10 minutes."

---

## Why GitHub Actions (and not CodePipeline)

- Already where the code lives.
- OIDC → AWS means no long-lived AWS creds in GitHub.
- The CDK/Docker toolchains are first-class on GH-hosted runners.
- CodePipeline is a good alternative for highly regulated shops that want everything inside AWS; not needed here.

---

## Workflows at a glance

Four workflows under `.github/workflows/`:

| Workflow | Trigger | Does |
|---|---|---|
| `cdk-diff.yml` | PR open / update | `cdk synth` + `cdk diff` — posts the diff as a PR comment |
| `build-cpu.yml` | PR, push to `main` | builds `glmocr-cpu` image, pushes to ECR tagged `<sha>` and `latest` (main only) |
| `build-sglang.yml` | PR, push to `main` | builds `glmocr-sglang` image, same pattern |
| `cdk-deploy.yml` | push to `main` (after the two builds succeed) | `cdk deploy --all` against the prod account |

`cdk-deploy.yml` requires a GitHub environment called `prod` with **required reviewers** — nobody ships without a manual approval click.

---

## OIDC setup (one-time)

In AWS:

1. Create an OIDC provider for `token.actions.githubusercontent.com`.
2. Create an IAM role `glmocr-github-actions` with trust policy:

```json
{
  "Effect": "Allow",
  "Principal": {"Federated": "arn:aws:iam::<acct>:oidc-provider/token.actions.githubusercontent.com"},
  "Action": "sts:AssumeRoleWithWebIdentity",
  "Condition": {
    "StringLike": {
      "token.actions.githubusercontent.com:sub": "repo:<org>/glmocr-prod:ref:refs/heads/main"
    },
    "StringEquals": {
      "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
    }
  }
}
```

3. Attach the permissions listed in `07-secrets-and-config.md` §"CI role".

In GitHub repo → Settings → Secrets and variables → Actions → Variables:

- `AWS_ROLE_ARN` = the role ARN
- `AWS_REGION` = `us-east-1` (or the chosen region)
- `AWS_ACCOUNT_ID` = the numeric account ID

No secrets (no `AWS_ACCESS_KEY_ID` etc.) — the role is the only handle to AWS.

---

## `cdk-diff.yml`

```yaml
name: cdk-diff
on:
  pull_request:
    paths:
      - 'cmd/**'
      - 'internal/**'
      - 'cdk.json'
      - 'go.mod'
      - 'go.sum'
permissions:
  id-token: write       # OIDC
  contents: read
  pull-requests: write  # post diff as comment
jobs:
  diff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_ROLE_ARN }}
          aws-region:     ${{ vars.AWS_REGION }}
      - uses: actions/setup-go@v5
        with: { go-version: '1.23' }
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - run: npm install -g aws-cdk@2
      - run: go mod download
      - id: diff
        run: cdk diff --context stage=prod --all 2>&1 | tee diff.txt
      - uses: marocchino/sticky-pull-request-comment@v2
        with:
          header: cdk-diff
          path: diff.txt
```

---

## `build-cpu.yml`

```yaml
name: build-cpu
on:
  pull_request:
    paths: ['docker/cpu/**']
  push:
    branches: [main]
    paths: ['docker/cpu/**']
permissions:
  id-token: write
  contents: read
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with: { role-to-assume: ${{ vars.AWS_ROLE_ARN }}, aws-region: ${{ vars.AWS_REGION }} }
      - uses: aws-actions/amazon-ecr-login@v2
        id: login
      - uses: docker/setup-buildx-action@v3
      - uses: docker/build-push-action@v6
        with:
          context: docker/cpu
          file: docker/cpu/Dockerfile.slim
          push: ${{ github.ref == 'refs/heads/main' }}
          tags: |
            ${{ steps.login.outputs.registry }}/glmocr-cpu:${{ github.sha }}
            ${{ steps.login.outputs.registry }}/glmocr-cpu:latest
          cache-from: type=gha
          cache-to:   type=gha,mode=max
```

`build-sglang.yml` is the same shape against `docker/sglang/`.

**Don't** tag `latest` on PR builds — only on `main`. PR builds still push the SHA tag so the deploy step can pin.

---

## `cdk-deploy.yml`

```yaml
name: cdk-deploy
on:
  push:
    branches: [main]
  workflow_run:
    workflows: [build-cpu, build-sglang]
    types: [completed]
permissions:
  id-token: write
  contents: read
jobs:
  deploy:
    if: github.event.workflow_run.conclusion == 'success' || github.event_name == 'push'
    runs-on: ubuntu-latest
    environment: prod    # manual approval via repo settings
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with: { role-to-assume: ${{ vars.AWS_ROLE_ARN }}, aws-region: ${{ vars.AWS_REGION }} }
      - uses: actions/setup-go@v5
        with: { go-version: '1.23' }
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - run: npm install -g aws-cdk@2
      - run: go mod download

      - name: Deploy
        run: |
          cdk deploy --all \
            --context stage=prod \
            --context cpuImageTag=${{ github.sha }} \
            --context sglangImageTag=${{ github.sha }} \
            --require-approval never

      - name: Smoke test
        env:
          ALB_URL: ${{ steps.deploy.outputs.alb_url }}   # wire as stack output
        run: bash scripts/smoke.sh "$ALB_URL"

      - name: Matrix load test
        if: github.ref == 'refs/heads/main'
        env:
          CPU_URL: ${{ steps.deploy.outputs.alb_url }}
        run: MATRIX_TOTAL=100 bash scripts/omnidoc_asyncio_matrix.sh || true

      - name: Upload matrix report to S3
        run: |
          aws s3 cp loadtest/results/ "s3://<reports-bucket>/ci/${{ github.sha }}/" \
            --recursive --exclude "*" --include "*.md"
```

---

## Rollback

Two scenarios:

**App rollback (bad image):** re-run the workflow with `cpuImageTag` / `sglangImageTag` pinned to the last-known-good SHA:

```bash
gh workflow run cdk-deploy.yml -f cpu_tag=sha-abc123 -f sglang_tag=sha-abc123
```

(Add the input params to the workflow on first cut.)

**Infra rollback (bad CDK change):** `git revert` the PR, merge, normal deploy flow re-applies the previous stack.

**Knob rollback (bad tuning):** AWS console or CLI `put-parameter` with the old value, then `ecs update-service --force-new-deployment`. No CDK involved.

---

## Matrix run as a post-deploy guard

The deploy job runs `scripts/omnidoc_asyncio_matrix.sh` against the ALB with `MATRIX_TOTAL=100` (smaller than the dev 200 to keep CI fast — ~5 minutes). The matrix result is uploaded to the reports S3 bucket at `s3://<bucket>/ci/<sha>/`.

**Pass criteria (wire into the job):**
- No failures at c=12 and c=24 (these are the "normal load" cells)
- rps at c=24 within 30% of a reference baseline committed as `reference/prod-baseline.json`

If the matrix fails, the deploy workflow posts a comment on the merge commit but doesn't automatically roll back (we don't know if it's noise yet). An engineer reviews and either re-runs, rolls back, or commits a new baseline.

---

## Secrets for CI (the small set)

In addition to OIDC credentials, CI needs:

- Nothing else. Everything else (ECR, S3, Secrets Manager, SSM) is accessed via the assumed role.

---

## What's explicitly NOT automated

- **Baking the weights tarball.** `scripts/bake-weights.sh` runs on a developer laptop (or on-demand via `workflow_dispatch`). It requires `HF_TOKEN` and touches a cross-region HF download; not a great fit for scheduled CI. Bump-weights is expected to be a manual, auditable event.
- **Grafana dashboard import.** The first-time import is manual on the MacBook. After that, dashboard changes happen in the Grafana UI and are exported back to `docker/grafana/dashboards/glmocr-load.json` via a PR for version control.
- **`cdk bootstrap`.** One-time per account-region pair; don't let CI do it.

---

Next: [`09-runbook.md`](./09-runbook.md).
