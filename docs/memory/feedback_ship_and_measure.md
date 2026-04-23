---
name: Prefers ship-and-measure over analyze-and-skip for optimizations
description: When user disagrees with my "skip this optimization" recommendation, default to shipping with clear rollback criteria rather than debating
type: feedback
originSessionId: 4f68fa82-22d5-414d-8646-1b145bb7159f
---
When I recommend skipping an optimization on cost/benefit grounds and the user overrides with "let's ship it and see," don't re-litigate. Implement it with:

1. A cheap rollback path (env flag, not code revert).
2. An explicit numeric rollback criterion stated up front.
3. A head-to-head measurement against the current baseline.

**Why:** Phase 2 of the torch→ONNX study (2026-04-22). I recommended skipping Phase 2 because the asyncio-matrix showed Phase 1 captured rps +17% / p95 -31% already, and my model of residual gains said Phase 2 would buy <1% on top. User override: "let's ship Phase 2 and run the matrix; if we don't see improvement we rollback Phase 2." Their judgment: when rollback is cheap, measurement beats analysis.

**How to apply:**
- If the user explicitly says "ship it" after I've recommended skipping, don't restate the case for skipping. Acknowledge the override, spec the rollback path, and go.
- Structure the implementation so the new behaviour is gated behind an env flag with a safe default. Rollback = flip the flag.
- Before kicking off the perf run, state the ship/rollback threshold in concrete numbers the user can evaluate without my interpretation.
- My prior-probability of being wrong about "it won't help" is higher than I instinctively estimate — especially under concurrency, where my mental model of Python/GIL/allocator interactions has been off before (see Phase 1 matrix result: predicted +4%, measured +17%).
