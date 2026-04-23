---
name: SGLang schedule_policy=lpm is correct for this workload; fcfs regresses
description: 2026-04-23 A/B of SGL_SCHEDULE_POLICY lpm vs fcfs at MATRIX_TOTAL=200. LPM wins at every useful concurrency (fcfs -28% rps c=12, -16% c=24, capacity cliffs at c=32/40). Don't re-run without a reason.
type: project
originSessionId: c8845dae-98e9-4b22-a7ae-02787ef2bf30
---
One-paragraph verdict: **keep `SGL_SCHEDULE_POLICY=lpm`** on this stack. LPM beats fcfs 3.124→2.244 rps at c=12 (-28%), 3.595→3.013 at c=24 (-16%), and survives c=32/c=40 with zero failures where fcfs trips ServerDisconnectedError on 3 of 200 and 16 of 200 respectively. The full side-by-side is at `loadtest/results/comparison-20260423-scheduler.md`; the raw matrices are `loadtest/results/omnidoc-20260423-094350-asyncio-matrix.md` (lpm) and `loadtest/results/omnidoc-20260423-095555-asyncio-matrix.md` (fcfs).

**Why fcfs loses despite *lower* SGLang queue-wait:**

FCFS actually has lower SGLang-side queue-wait (c=24 mean 108 ms vs lpm 862 ms; c=64 mean 194 ms vs lpm 2,890 ms). That's not the whole story — on this stack the saving is consumed (and more) by the CPU layout stage:

- At c=64, lpm in-flight mean 17.1 / layout forward 9,507 ms; fcfs in-flight mean 34.8 / layout forward 17,096 ms.
- Because fcfs returns OCR region calls faster (no prefix reordering), the gunicorn workers churn more layout stages per wall second → the layout batcher (max=8, window=20 ms) sees more concurrent demand → per-request Conv time nearly doubles.
- Net: Flask end-to-end worsens at every cell. This is the layout-Conv bottleneck (76% wall-time share, see `project_layout_profile.md`) absorbing any scheduler win.

**When this verdict could change:**

- **If layout moves off CPU** (e.g. DocLayout-YOLO swap per `project_openvino_ep.md`) — fcfs's back-pressure into the layout stage disappears, and its lower-queue-wait win at SGLang might dominate. Re-run the A/B after any layout detector swap.
- **If the workload's prompt prefix stops overlapping** (e.g. new OCR pipeline with per-request custom system prompts). LPM's value is prefix-grouping; if prefixes diverge per request, LPM has nothing to group and fcfs's simpler admission wins. Current glmocr region calls share the system prompt, so LPM is correct.
- **If SGLang's LPM implementation changes materially** (upstream scheduler rewrite). The current radix-tree-based LPM is what we measured; a future heuristic might shift the comparison.

Don't re-run this A/B before one of those three trigger conditions; at MATRIX_TOTAL=200 for both runs it costs ~40 min of stack time plus a SGLang restart cycle, and the prefix-overlap argument is structural to this workload.
