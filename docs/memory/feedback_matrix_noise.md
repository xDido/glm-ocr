---
name: Average matrix runs before concluding on perf tuning
description: Single asyncio-matrix runs are ±15–25% noisy on rps; always run 2+ and average before committing to a keep/rollback call
type: feedback
originSessionId: 4f68fa82-22d5-414d-8646-1b145bb7159f
---
**Rule:** Never commit to a keep/rollback decision on a single matrix run for this load-test harness. Run at least 2 matrix runs per config change and average the rps / p50 / p95 / p99 before comparing.

**Why:** In the Fix 1–4 tuning sweep (2026-04-22), single-run variance repeatedly inverted qualitative conclusions:

- Fix 3 single-run c=12 rps: 2.64 on run 1, 2.18 on run 2 — a **17% drop** with identical config.
- Fix 4b single-run c=12 rps: 2.26 on run 1, 3.10 on run 2 — a **37% swing** with identical config.
- Fix 4b single-run c=64 failures: 0 on run 1, 32 on run 2 — zero-vs-flooding with identical config.

I nearly rolled back Fix 4b citing "c=12 regresses 14%" on a single run — the rerun showed c=12 was actually the best of any config. Similarly nearly kept Fix 4 async citing "c=24 +20% and 0 failures" on a single run — the rerun showed c=32 took 24 failures, dropping rps to 1.71.

**How to apply:**
- For any matrix comparison that will drive a config change, run the matrix twice per config (baseline and candidate) and average rps/p50/p95/p99 before declaring a winner.
- The harness is `scripts/omnidoc_asyncio_matrix.sh`. User prefers running it at **N=200** (`MATRIX_TOTAL=200 bash scripts/omnidoc_asyncio_matrix.sh`) — doubles samples vs the N=100 default, shrinking the noise band and making single-run calls more credible when the gate is clearly met/missed. A full 5-trial N=200 run with observability augmented is ~15–25 min.
- Decision workflow per perf experiment: apply change → run matrix once at N=200 → if result is inside/outside the gate, decide keep-or-move-on; if ambiguous (near the gate), rerun once and average.
- Pay attention to rps **and** p50 **and** p99 — rps alone is misleading because wall-time is dominated by the worst tail (one 50s outlier at c=12 drops rps by 15%+).
- When a metric moves by less than ~10% even at N=200, treat it as noise, not signal.
- When a run shows failures that a prior run didn't, the config is right at a capacity cliff and a third data point is worth capturing.
