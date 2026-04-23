---
name: GPU utilization A/B/C/D — none of the scheduler knobs help on this workload
description: 2026-04-23 four-experiment sweep. SGLang running batch peaks at 11-16 not because scheduler is conservative, but because CPU layout stage is the binding bottleneck producing bursty arrivals. Tried spec-off, cuda_graph_max_bs=32, schedule_conservativeness=0.5, context_length=10240 — all net-negative. The only real lever for raising sustained running count is speeding up the CPU layout stage (detector swap).
type: project
originSessionId: c8845dae-98e9-4b22-a7ae-02787ef2bf30
---
**Root-cause finding:** the pattern "SGLang queue=42 while running=16" is **not** scheduler throttling — it's bursty arrivals from the CPU layout stage. 4 CPU workers complete ~1 layout forward/sec each → ~4 pages/sec → ~24 region calls/sec (pages fan out ~6 regions each). Each OCR call takes ~2–3 s GPU-side. Little's Law: running = 24 × 0.5 ≈ 12 (matches measured 11–16 peak). The layout batcher bunches completions into periodic bursts → queue briefly spikes to ~40+ → drains through 12-ish running slots → back to idle. No GPU-side knob can make running batch higher than the inflow rate × per-call residence supports.

**Four experiments, all rejected** (reports in `loadtest/results/omnidoc-20260423-{114825,120150,121324,122512}-asyncio-matrix.md`; full comparison at `loadtest/results/comparison-20260423-gpu-utilization.md`):

| # | Change | Result | Failure mode |
|---|---|---|---|
| A | `SGL_SPECULATIVE=false` | rps regressed at c=12/24/32/40 (−11 to −36%); running peak jumped 11→27 at c=40 but rps at that cell dropped 36%. Spec-off is multi-stream-friendly / single-stream-hostile, but our per-stream throughput matters more. | rps net-negative |
| B | `SGL_CUDA_GRAPH_MAX_BS=32` (spec on) | rps regressed 12–30% at most cells; 14 ServerDisconnectedErrors at c=32. Larger graph capture when spec is on just destabilizes without raising sustained running. | capacity cliff |
| C | `SGL_SCHEDULE_CONSERVATIVENESS=0.5` | mixed (c=32 +27%, c=40 +24%) but 19 total failures across c=24 + c=64. Aggressive admission overruns the stack at both ends. | capacity cliff |
| D | `SGL_CONTEXT_LENGTH=10240` (from 24576) | effective max_total_num_tokens barely moved (37710→38254) because the pool is already GPU-memory-bound not context-bound; 21 ServerDisconnectedErrors at c=40 + c=64 without material rps uplift. | capacity cliff, no KV gain |

**Plumbing kept in tree (additive, default-off):**
- `SGL_CUDA_GRAPH_MAX_BS` env knob → `--cuda-graph-max-bs` flag in `docker/gpu/entrypoint.sh`
- `SGL_SCHEDULE_CONSERVATIVENESS` env knob → `--schedule-conservativeness` flag
- Both unset by default; no change to baseline behavior. Useful for future per-deployment overrides if someone has a workload structurally different from this one.

**Don't re-run this experiment unless:**
- The CPU layout stage's throughput changes materially (e.g. DocLayout-YOLO swap) — at that point the bursty-arrival pattern changes and the scheduler knobs might respond differently.
- SGLang upstream changes the default cuda_graph_max_bs or spec-decode interaction (would show as different default server_args on restart).
- We move off the 3060 Ti to a GPU where the compute ceiling shifts.

**Real follow-up for "utilize GPU better":** speed up the CPU layout stage, not the GPU scheduler. That's the DocLayout-YOLO detector swap identified in `project_kernel_levers_2026_04_23.md`.
