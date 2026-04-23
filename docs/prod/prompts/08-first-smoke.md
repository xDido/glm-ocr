# Prompt 08 — First smoke + matrix + handoff complete

> Prerequisite: prompt 07 done (CI/CD wired, a full deploy was green). Expect ~30 min.

---

```
Run the full end-to-end smoke suite and baseline the matrix. Acceptance of this
step means the handoff from dev → prod is complete.

Read first:
  - docs/prod/10-claude-code-handoff.md §"Success definition"
  - docs/prod/09-runbook.md §"Run a matrix test against prod"

Tasks:

## Part 0 — Build the loadtest harness + smoke from scratch

Read docs/prod/13-loadtest-spec.md. Create:

  scripts/smoke.sh
  scripts/bake-weights.sh           (if you haven't already in prompt 03)
  scripts/omnidoc_asyncio_matrix.sh
  scripts/lib/loadtest_common.sh
  scripts/test_images/              (add 3-5 small PNG/JPG sample documents)
  loadtest/asyncio/bench.py

Each file is fully specified in 13-loadtest-spec.md — copy the blocks verbatim.
chmod +x the .sh files. Commit as "feat(loadtest): smoke + matrix harness".

Local sanity (against a running local SGLang+CPU if available):
    bash scripts/smoke.sh http://localhost:5002 scripts/test_images/sample.png
    CPU_URL=http://localhost:5002 MATRIX_TOTAL=20 bash scripts/omnidoc_asyncio_matrix.sh

## Part A — Functional smoke

1. Fetch the ALB URL from the Fargate stack outputs:
     aws cloudformation describe-stacks --stack-name glmocr-fargate-prod \
       --query 'Stacks[0].Outputs' --output table

2. Run scripts/smoke.sh against it:
     bash scripts/smoke.sh "$ALB_URL"
   Output should show a non-zero markdown length and a non-null JSON shape.

3. Check each layer:
   - curl $ALB/health                        → 200
   - curl $ALB/metrics                       → 200, full prom output
   - aws sagemaker-runtime invoke-endpoint ...  → 200 with valid chat JSON

## Part B — Matrix baseline

1. Run a reduced matrix (MATRIX_TOTAL=100 to keep it under 10 min):
     export CPU_URL="$ALB"
     export MATRIX_TOTAL=100
     bash scripts/omnidoc_asyncio_matrix.sh

2. The report lands in loadtest/results/omnidoc-<ts>-asyncio-matrix.md.
   Check:
   - c=12 rps    — similar to dev baseline (3.1 rps ± 25%)
   - c=24 rps    — similar to dev baseline (3.6 rps ± 25%)
   - c=12 fails  — 0
   - c=24 fails  — 0

3. Commit the report to the repo as reference/prod-baseline.md (rename):
     cp loadtest/results/omnidoc-*-asyncio-matrix.md docs/prod/reference/prod-baseline.md
     git add docs/prod/reference/prod-baseline.md
     git commit -m "docs: first prod matrix baseline"

## Part C — Dashboard smoke

1. Open https://dido.grafana.net/d/glmocr-load
2. Confirm the main panels show live data from the matrix run
3. Screenshot the full dashboard; attach to the commit/PR as evidence

## Part D — Alarms smoke

1. All six alarms should be in OK state after the matrix run stabilizes:
     aws cloudwatch describe-alarms \
       --alarm-name-prefix glmocr- \
       --query 'MetricAlarms[].{Name:AlarmName,State:StateValue}' \
       --output table

## Part E — Handoff complete

Write a short note in docs/prod/DEPLOYMENT-LOG.md:

  # Production deployment log

  ## First ship — <date>

  - Deployed stacks: glmocr-{network,ecr,secrets,sagemaker,fargate,obs}-prod
  - SageMaker endpoint: glmocr-sglang (ml.g4dn.2xlarge) — created in X minutes
  - Fargate service: glmocr-cpu — 1 task
  - ALB: <dns>
  - Matrix baseline (c=12/24/32): <rps/rps/rps>, 0 failures
  - Dashboard: https://dido.grafana.net/d/glmocr-load — live
  - Alarms: all 6 in OK state
  - Known-future-work:
    * Try SGLANG_USE_CUDA_IPC_TRANSPORT=1 (runbook §Future work)
    * Multi-AZ endpoint when HA is needed
    * Savings Plan after 3 months of stable load

Commit as "docs: first ship deployment log + prod baseline".

Report to the human: "Handoff complete — you're live."
```
