# 10 — Claude Code handoff

**Purpose:** the bootstrap doc for the **new MacBook's Claude Code session**. If you are that Claude Code, read this first, then read `01-prod-architecture.md`. The rest of the docs are linked at the right moment below.

If you are the human operator: print or keep this tab open while walking through the setup steps.

---

## What you're doing

You (Claude Code on the MacBook) are going to build the AWS production deployment of an OCR service that has already been designed, measured, and tuned on a Windows dev box. The dev work is **done and validated**; you are the implementation + deploy layer.

Your inputs: this `docs/prod/` tree, plus the `docker/cpu/`, `docker/gpu/`, `scripts/`, `loadtest/`, `memory/` trees that the human copies over from the dev repo.

Your outputs: a working Go CDK project that deploys a Fargate CPU service + a SageMaker SGLang endpoint, with observability, secrets, and CI wired up.

---

## One-time setup (human)

### (a) Install tooling on the MacBook

```bash
# Claude Code
curl -fsSL https://claude.ai/install.sh | bash

# Node (CDK wrapper — the Go CDK still shells out to `cdk` which is a JS CLI)
brew install node@20
sudo npm install -g aws-cdk@2

# Go
brew install go@1.23

# AWS CLI
brew install awscli

# Docker Desktop (for local image builds)
brew install --cask docker

# (Optional) gh CLI for GitHub interactions
brew install gh
```

### (b) Configure AWS

```bash
aws configure sso       # or aws configure --profile <name> with keys
aws sts get-caller-identity   # verify
```

### (c) Create an empty prod repo

```bash
# On GitHub:
gh repo create <org>/glmocr-prod --private --clone --source=. --push
cd glmocr-prod

# or without gh:
mkdir glmocr-prod && cd glmocr-prod && git init
# (create + push to remote manually)
```

### (d) Copy assets from the dev repo

Only the docs get copied. Everything else is rebuilt from scratch.

```bash
# The dev repo is at <dev-machine>:C:\Users\Dido\Desktop\GLM-OCR\
# Copy ONLY these — all docs, no code:
cp -R <dev-repo>/docs/prod              ./docs/prod         # this whole handoff
cp <dev-repo>/docs/ARCHITECTURE.md      ./docs/ARCHITECTURE.md   # deep reference
cp <dev-repo>/docs/OPTIMIZATIONS.md     ./docs/OPTIMIZATIONS.md  # deep reference
# Memory is NOT copied into the repo — it's Claude Code's per-user auto-memory
# store and lives at ~/.claude/projects/<slug>/memory/. See step (e).
```

**`docker/`, `scripts/`, `loadtest/` are NOT copied** — they get rebuilt from scratch in the new repo using:
- `docs/prod/12-cpu-container-spec.md` for `docker/cpu/`
- `docs/prod/03-sagemaker-sglang-byoc.md` for `docker/sglang/`
- `docs/prod/06-observability-prod.md` for `docker/alloy/`
- `docs/prod/05-sigv4-proxy-sidecar.md` for `docker/sigv4-proxy/` (optional)
- `docs/prod/13-loadtest-spec.md` for `scripts/` and `loadtest/`

This is intentional: the prod MVP ships simpler than the dev tuned stack and leaves the ~3 weeks of runtime optimizations (custom numpy postproc, layout coalescer, phase-augmenter report) as a documented future-port phase. The MVP gets the 1.76× ONNX layout win + all the HTTP/retry/pool tuning; the rest is follow-on work.

If you later want to port a specific dev optimization (e.g. the layout coalescer), do `git clone <dev-repo> /tmp/glmocr-dev` as a read-only reference — don't copy the files in.

Copy `docs/prod/11-CLAUDE.md.template` to the **repo root** as `CLAUDE.md`:

```bash
cp docs/prod/11-CLAUDE.md.template ./CLAUDE.md
```

### (e) Seed Claude Code's auto-memory

Claude Code's memory is **not** part of the git repo. It lives at
`~/.claude/projects/<slug>/memory/` on the local machine, where `<slug>` is
a sanitized version of the project's working-directory path.

On the new MacBook the slug will be something like
`Users-<you>-glmocr-prod` (Claude Code computes it on first session
open). The directory is created automatically the first time you run
`claude` in the new repo.

**Two paths to seed it:**

**Automatic (recommended).** Let the first Claude Code session rebuild memory
from `docs/prod/reference/memory-seed.md`. Tell it to, in the first prompt
(step f). `memory-seed.md` is a condensed, self-contained export of the
dev memories — Claude Code will read it and recreate per-topic memory
files under `~/.claude/projects/<slug>/memory/`.

**Manual (highest fidelity).** Copy the dev memory folder wholesale:

```bash
# 1. On the Windows dev box, zip the memory folder:
#    Path: C:\Users\Dido\.claude\projects\C--Users-Dido-Desktop-GLM-OCR\memory\
#    Contains ~13 files: MEMORY.md + 12 project/feedback/user markdown files
powershell Compress-Archive \
    -Path "$env:USERPROFILE\.claude\projects\C--Users-Dido-Desktop-GLM-OCR\memory\*" \
    -DestinationPath "$env:USERPROFILE\Desktop\glmocr-memory.zip"

# 2. Transfer glmocr-memory.zip to the MacBook (USB, rsync, cloud drive).

# 3. On the MacBook, after running `claude` once in the new repo so the
#    project dir exists:
SLUG=$(ls ~/.claude/projects | grep -i glmocr-prod | head -1)
DEST="$HOME/.claude/projects/$SLUG/memory"
mkdir -p "$DEST"
unzip -o ~/Downloads/glmocr-memory.zip -d "$DEST"
```

Either path ends with Claude Code's memory pre-populated before you start
implementing. Manual is more faithful; automatic is lower-effort.

### (f) First Claude Code session

```bash
cd <glmocr-prod repo>
claude
```

Paste this as the **first prompt**:

```
This is a fresh prod repo. We're porting a working GLM-OCR service from a
Windows dev box to AWS Fargate + SageMaker + Go CDK.

Bootstrap the session:
1. Read CLAUDE.md in the repo root.
2. Read docs/prod/00-START-HERE.md and docs/prod/01-prod-architecture.md.
3. List what's in docs/prod/reference/. Read reference/env-tuned.md and
   reference/memory-seed.md to understand the tuning history.
4. Run 'git status' and 'ls -la' to confirm you see: docs/prod/,
   docs/ARCHITECTURE.md, docs/OPTIMIZATIONS.md. Note: docker/, scripts/,
   loadtest/ do NOT exist yet — they get built from scratch per docs/prod/12
   and docs/prod/13. memory/ is NOT in the repo either; it lives in
   ~/.claude/projects/<slug>/memory/ and was seeded in setup step (e).
5. Summarize in 5 bullets what you understand, and what the first
   implementation chunk is (hint: docs/prod/prompts/01-bootstrap-cdk.md).

Don't make any code changes yet — just orient.
```

Claude Code should respond with a terse orientation summary. Verify it mentions Fargate, SageMaker, Go CDK, the six stacks from `02-cdk-go-structure.md`, and the sigv4-proxy pattern. **If it doesn't**, the memory seeding didn't take — re-do step (e) manually.

---

## Session cadence (for Claude Code, going forward)

Do not try to do all of it in one session. The prod implementation is nine discrete chunks, one per `prompts/` file:

| # | Prompt | Session length | Blocking on |
|---|---|---|---|
| 1 | `prompts/01-bootstrap-cdk.md` | ~30 min | — |
| 2 | `prompts/02-network-stack.md` | ~30 min | CDK bootstrap complete |
| 3 | `prompts/03-ecr-build.md` | ~45 min | Docker installed + authed to ECR |
| 4 | `prompts/04-sagemaker-stack.md` | ~30 min | ECR push of glmocr-sglang image; weights baked to S3 |
| 5 | `prompts/05-fargate-stack.md` | ~45 min | SagemakerStack deployed (we need the endpoint name) |
| 6 | `prompts/06-observability.md` | ~30 min | FargateStack deployed (we need task log groups) |
| 7 | `prompts/07-ci-cd.md` | ~45 min | OIDC role set up |
| 8 | `prompts/08-first-smoke.md` | ~30 min | All stacks deployed |

**Budget ~4-5 hours of total Claude Code time.** Real-world wall time is longer because of AWS provisioning (SageMaker endpoint create ~10 min, bootstrap ~5 min, etc.).

Between sessions: compact or close. Don't try to hold the whole implementation in one context window.

---

## At the start of every new session

Paste this:

```
Refresh context:
1. Read CLAUDE.md.
2. Read docs/prod/00-START-HERE.md (reading order).
3. Check TaskList — which prompts are done, which is next?
4. Tell me in 3 bullets: (a) where we left off, (b) what the next
   prompt expects, (c) any open questions from last time.
```

Then hand off the relevant `prompts/XX-*.md` file.

---

## Success definition

The handoff is successful when:

1. `cdk synth --context stage=prod --all` produces valid CloudFormation.
2. `cdk deploy --context stage=prod --all` deploys green end-to-end.
3. `bash scripts/smoke.sh <alb-url>` returns HTTP 200 with a non-empty response on a real PDF sample.
4. `MATRIX_TOTAL=100 bash scripts/omnidoc_asyncio_matrix.sh` completes with 0 failures at c=12 and c=24, and posts a report to Grafana Cloud.
5. Grafana Cloud dashboard `glmocr-load` shows live metrics.
6. The six CloudWatch alarms in `06-observability-prod.md` are in state `OK`.

Acceptance criterion: the human runs all six checks and confirms green.

---

## When things go wrong

- **"cdk bootstrap" errors on the MacBook:** usually an AWS role mismatch. Re-auth with `aws sso login`, verify region.
- **SageMaker endpoint stuck in `Creating` > 15 min:** check CloudWatch logs `/aws/sagemaker/Endpoints/glmocr-sglang/AllTraffic`. Most likely `model.tar.gz` shape is wrong (a nested folder at the root). Re-bake.
- **Fargate task flapping:** check sigv4-proxy dependsOn block — if CPU starts before proxy, early requests fail.
- **Grafana Cloud not receiving metrics:** Alloy sidecar logs (`aws logs tail /glmocr/prod/alloy`) — usually a missing env var from the grafana_cloud secret.
- **You (Claude Code) are confused or missing context:** stop. Read `CLAUDE.md` and `docs/prod/00-START-HERE.md` fresh. Re-read the specific prompt you're on. Check memory. Ask the human.

---

## Reading list (linear, for Claude Code)

```
CLAUDE.md
docs/prod/00-START-HERE.md
docs/prod/01-prod-architecture.md
docs/prod/reference/memory-seed.md
docs/prod/reference/env-tuned.md
docs/prod/02-cdk-go-structure.md
docs/prod/prompts/01-bootstrap-cdk.md      ← start implementing
# after each session:
docs/prod/prompts/02-*.md
docs/prod/12-cpu-container-spec.md         ← read before prompt 03 (builds docker/cpu/)
docs/prod/03-sagemaker-sglang-byoc.md      ← read before prompt 03 (also builds docker/sglang/)
docs/prod/prompts/03-*.md
docs/prod/prompts/04-*.md
docs/prod/04-fargate-cpu-task.md           ← read before prompt 05
docs/prod/05-sigv4-proxy-sidecar.md        ← read before prompt 05
docs/prod/07-secrets-and-config.md         ← read before prompts 05 & 06
docs/prod/prompts/05-*.md
docs/prod/06-observability-prod.md         ← read before prompt 06 (builds docker/alloy/)
docs/prod/prompts/06-*.md
docs/prod/08-ci-cd.md                      ← read before prompt 07
docs/prod/prompts/07-*.md
docs/prod/13-loadtest-spec.md              ← read before prompt 08 (builds scripts/ + loadtest/)
docs/prod/prompts/08-*.md
docs/prod/09-runbook.md                    ← read after deploy is green
```

---

**Next file:** [`11-CLAUDE.md.template`](./11-CLAUDE.md.template) (goes in repo root as `CLAUDE.md`).
