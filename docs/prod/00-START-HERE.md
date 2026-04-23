# 00 — Start here

**Purpose:** entry point for the production handoff package. Read this first when opening the new prod repo on the new MacBook.

---

## What this folder is

This folder (`docs/prod/`) is the **handoff package** from the dev stack (Windows + Docker Compose + RTX 3060 Ti, locally tuned + measured) to the production stack (AWS Fargate + SageMaker + Go CDK, on a MacBook in a new git repo).

It was produced on the dev machine as a self-contained blueprint so a **fresh Claude Code session on the new MacBook** can implement the prod system without needing to re-derive anything from the dev repo.

If you are reading this on the **new MacBook**: you are in the right place. Read files in order.

If you are reading this on the **dev repo**: these files are the source. Copy the whole `docs/prod/` tree into the new repo's `docs/prod/` (keeping the path), then start work on the MacBook.

**Note on code:** `docker/`, `scripts/`, and `loadtest/` are **NOT** copied from the dev repo. They get rebuilt from scratch in the new repo using the specs in `12-cpu-container-spec.md` and `13-loadtest-spec.md`. The dev-repo versions contain ~3 weeks of tuned customizations; the MVP prod rebuild intentionally starts simpler (ships with functional defaults + the 1.76× ONNX win) and leaves the tuned optimizations as a documented future-port phase.

---

## Reading order

Numbered because order matters. Each doc is short and cross-links to the next.

| # | File | When to read |
|---|---|---|
| 00 | `00-START-HERE.md` | **(this file)** |
| 01 | `01-prod-architecture.md` | Before anything else. Target system at one glance. |
| 02 | `02-cdk-go-structure.md` | Before writing any CDK. Module layout + stack boundaries. |
| 03 | `03-sagemaker-sglang-byoc.md` | Before building the SGLang image. |
| 04 | `04-fargate-cpu-task.md` | Before building the Fargate task def. |
| 05 | `05-sigv4-proxy-sidecar.md` | Paired with 04 — the sidecar that wires CPU → SageMaker. |
| 06 | `06-observability-prod.md` | Before Alloy/CloudWatch/Grafana wiring. |
| 07 | `07-secrets-and-config.md` | Before putting anything in `.env`. |
| 08 | `08-ci-cd.md` | Before setting up GitHub Actions. |
| 09 | `09-runbook.md` | On-call / incident guide. Read on first deploy, then on-demand. |
| 10 | `10-claude-code-handoff.md` | **If you are Claude Code on the MacBook, read this before 01–09.** It tells you how to bootstrap yourself. |
| 11 | `11-CLAUDE.md.template` | Copy to the new repo root as `CLAUDE.md`. Not to be read linearly. |
| 12 | `12-cpu-container-spec.md` | Read before prompts 03 (building `docker/cpu/` from scratch). |
| 13 | `13-loadtest-spec.md` | Read before prompt 08 (building `scripts/` + `loadtest/` from scratch). |

Plus two subfolders:

- `prompts/` — eight copy-pasteable prompts, one per implementation chunk. Work in order.
- `reference/` — static portability artifacts (tuned `.env`, memory seed, annotated compose). Cited from the other docs.

---

## What's NOT in this folder

- **Source code.** `docker/cpu/`, `docker/sglang/`, `docker/alloy/`, `scripts/`, and `loadtest/` are all **built from scratch in the new repo** using the specs in `12-cpu-container-spec.md` and `13-loadtest-spec.md`. No files are copied from the dev repo.
- **CDK code.** The Go CDK project is created by `prompts/01-bootstrap-cdk.md` during implementation.
- **Secrets.** Nothing confidential is in these files. The actual HF token, Grafana token, and AWS credentials get wired via Secrets Manager on deploy.

---

## The two hard-earned lessons from the dev stack

These show up in nearly every doc below. Internalize before reading further:

1. **CPU-side layout forward is the binding constraint** on throughput. SGLang/GPU tuning does not raise aggregate rps above what the CPU can feed it. In prod, this means: **right-size the Fargate task CPU first**, then tune SageMaker. See `reference/memory-seed.md` for the evidence (the GPU utilization A/B/C/D and async-sidecar experiments).
2. **Single load-test runs are noisy (±15–25% on rps).** Never ship a config change on one matrix run. Always average two or more. See `09-runbook.md` for the protocol.

---

## When in doubt

- **Architecture question?** → `01-prod-architecture.md`
- **"How do I deploy?"** → `08-ci-cd.md` + `prompts/08-first-smoke.md`
- **"What knob do I turn?"** → `09-runbook.md` + `reference/env-tuned.md`
- **"What did the dev team learn the hard way?"** → `reference/memory-seed.md`

Next stop: [`01-prod-architecture.md`](./01-prod-architecture.md).
