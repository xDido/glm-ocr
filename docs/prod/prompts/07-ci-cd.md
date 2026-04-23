# Prompt 07 — CI/CD (GitHub Actions)

> Prerequisite: prompt 06 done (full stack deployed green). Expect ~45 min.

---

```
Wire up GitHub Actions with OIDC + automated deploys.

Read first:
  - docs/prod/08-ci-cd.md (full)

Tasks:

## Part A — AWS OIDC setup (one-time, outside CDK if preferred)

1. Create the OIDC provider for GitHub Actions:
     aws iam create-open-id-connect-provider \
       --url https://token.actions.githubusercontent.com \
       --client-id-list sts.amazonaws.com \
       --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1

2. Create role glmocr-github-actions with trust policy scoped to the repo
   (see docs/prod/08-ci-cd.md). Attach these permissions:
   - ECR: push/pull on glmocr-cpu, glmocr-sglang, glmocr-alloy
   - CloudFormation: full on stacks named glmocr-*-prod
   - IAM: PassRole on the task role, execution role, sagemaker execution role
   - S3: read/write on the artifacts + reports buckets
   - SSM: PutParameter on /glmocr/prod/* (for one-off drift fixes; leave off
     if you want to require console edits for all knob changes)
   - STS: AssumeRole on itself (for CDK bootstrap calls)
   - logs: Create/Put log groups for CI runs

3. Record the role ARN. Add to the GitHub repo as variable AWS_ROLE_ARN.
   Also add AWS_REGION and AWS_ACCOUNT_ID as variables.

## Part B — Workflows (.github/workflows/)

Create four workflow files following the templates in docs/prod/08-ci-cd.md:

1. cdk-diff.yml — PR opens/updates → posts cdk diff as comment
2. build-cpu.yml — PR + push main → build + push (push-to-ECR only on main)
3. build-sglang.yml — same pattern for SGLang image
4. cdk-deploy.yml — push main → requires `prod` environment approval →
   deploys all stacks with the new image tags → runs smoke + matrix

Add a `prod` environment in repo Settings → Environments → Protection rules:
"Required reviewers" — add at least yourself.

## Part C — Verify

1. Make a trivial no-op PR (e.g., add a comment to README.md).
2. Confirm cdk-diff comments on the PR with "There were no differences".
3. Merge.
4. Confirm build-cpu, build-sglang, and then cdk-deploy runs.
5. cdk-deploy pauses at the approval gate — approve it.
6. Confirm deploy succeeds and smoke test passes.

## Part D — Alarm SNS subscription

1. Subscribe an email or PagerDuty endpoint to the SNS topic glmocr-prod-alerts
   (created in prompt 06).
2. Confirm the subscription (click email link / confirm in PD).
3. Test one alarm: manually trigger (e.g., terminate a task to fire
   glmocr-alb-unhealthy-host; it self-heals via autoscaling).

Commit as "feat(ci): github actions oidc + 4 workflows + prod approval gate".

Report:
  - Links to the three successful workflow runs
  - Which alarms are in OK state
  - What does NOT work yet
```
