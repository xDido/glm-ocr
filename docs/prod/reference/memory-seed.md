# reference/memory-seed.md

**Purpose:** condensed distillation of the dev Claude Code auto-memory (`~/.claude/projects/C--Users-Dido-Desktop-GLM-OCR/memory/`) — the `MEMORY.md` index plus 12 project/feedback/user memories — sized for Claude Code on the new MacBook to re-seed its own auto-memory store in one pass. Note: these memory files are **not** part of the git repo on either side; they live under Claude Code's per-project profile directory.

**How to consume (human):** paste each section below into Claude Code as a memory to save (e.g., "Save this as a feedback memory: ..."), or copy the dev-repo's `~/.claude/projects/C--Users-Dido-Desktop-GLM-OCR/memory/` wholesale as documented in `docs/prod/10-claude-code-handoff.md` §(e).

**How to consume (Claude Code):** read each section; the human may have already hand-copied the raw memory files. Either way, internalize these so you don't re-run experiments the dev session already disproved.

---

## User profile (user memory)

The user is learning load-test and observability vocabulary. They prefer concepts and charts over raw JSON dumps. They work across Windows (dev) and macOS (prod); paths and shell syntax differ between the two.

When explaining numbers, show trends and context (baseline, noise band) not just the value.

---

## Feedback: tightly scope exploration (feedback)

**Rule:** when the user gives file:line anchors in the prompt, verify from there; don't spawn broad Explore agents that re-discover known targets.

**Why:** saves context and time; the user has already done the discovery work and is handing it to you.

**How to apply:** if a prompt mentions a specific file path or line number, go straight there with Read/Grep — don't dispatch an Explore agent unless the path turns out to be wrong.

---

## Feedback: ship-and-measure for perf optimizations (feedback)

**Rule:** if the user overrides my "skip this" recommendation on a performance optimization, implement with an env-flag rollback + an explicit numeric criterion for keep/rollback. Don't re-litigate.

**Why:** the user's judgment on what's worth trying may differ from mine, and measurement is cheap.

**How to apply:** `LAYOUT_ASYNC`, `LAYOUT_POSTPROC`, `LAYOUT_GRAPH`, `LAYOUT_COMPILE` are all env-flag rollbacks shipped this way — kept or rejected on measurement, not opinion.

---

## Feedback: matrix noise — average ≥2 runs (feedback)

**Rule:** single asyncio-matrix runs are ±15–25% on rps. Never commit keep/rollback on one run. Always compare p50 and p99 alongside rps.

**Why:** rps is tail-dominated; a single outlier trial shifts the number more than any real config change. Dev saw three cases where one-run "wins" flipped on a re-run.

**How to apply:** for any config change, run the matrix twice before calling it. If they disagree by more than noise (25%), run a third. `09-runbook.md` §"Run a matrix test against prod" has the command.

---

## Project: Layout ONNX forward is Conv-dominated (project)

Profiling found 76% of per-call wall time is Conv kernels. The layout ONNX forward on PP-DocLayoutV3 is a DETR-style model with a heavy Conv backbone; the post-proc is already numpy-fast.

**Why this matters in prod:** the next layout optimization should target the kernel (OpenVINO EP post-batch=1 fix, int8 with a non-DETR head, lower input resolution with parity validation on prod document mix), **not** graph surgery. Switching HTTP frameworks (FastAPI, LitServe, RayServe) doesn't help — all downstream of the Conv kernel.

**How to apply:** if ops asks "why is rps so low?", the answer is the CPU Conv kernel. Don't waste time tuning SGLang or HTTP layers.

---

## Project: OpenVINO EP rejected for PP-DocLayoutV3 (project)

Apparent "3× win" was 15% silent empty-response inflation. The root cause is `batch=1` baked into the model's torch source. The fix is a detector swap (DocLayout-YOLO), not a provider swap.

**Why this matters in prod:** if someone proposes OpenVINO, quote this memory. Re-evaluate only if the batch=1 source is fixed.

---

## Project: SGL_SCHEDULE_POLICY=lpm wins over fcfs (project)

fcfs is -16 to -28% rps at c=12/24, trips capacity cliff at c=32/40. Layout Conv absorbs any SGLang queue-wait saving from fcfs. Don't re-run without a detector swap or prompt-prefix change.

---

## Project: Kernel-level layout optimization 2026-04-23 (project)

`LAYOUT_INPUT_SIZE=640x640` reverted (parity validated only on OmniDocBench; unsafe default for passport/ID/receipt distributions). PTQ int8 rejected (DETR head too fragile). ORT fusion verified as no-op (torch export already fused Conv+BN). Next structural lever is DocLayout-YOLO (Phase 1 on a compatible CPU).

**How to apply:** do not lower `LAYOUT_INPUT_SIZE` in prod without a parity test on the actual document mix. Same for int8.

---

## Project: GPU utilization A/B/C/D 2026-04-23 (project)

SGLang running-batch peaks at 11–16 because the CPU layout stage is binding (bursty arrivals), not scheduler throttling. Tried `SGL_SPECULATIVE=false` / `SGL_CUDA_GRAPH_MAX_BS=32` / `SGL_SCHEDULE_CONSERVATIVENESS=0.5` / `SGL_CONTEXT_LENGTH=10240` — all net-negative.

**Why this matters in prod:** the lever for raising sustained running count is a faster CPU layout stage, not GPU scheduler tuning. Framework swaps (FastAPI, LitServe, RayServe) hit the same ceiling. **Prod shortcut:** if you see low SM `InvocationsPerInstance`, the bottleneck is upstream (Fargate CPU), not the endpoint.

---

## Project: DocLayout-YOLO slower than PP-DocLayoutV3 on Ryzen 5600X CPU (project)

Probe invalidated the "3× faster" assumption. On Ryzen 5600X (Zen3, AVX2): 3× slower at imgsz=1024, 11% slower at imgsz=640, only 1.32× faster at imgsz=512 (where quality degrades). Public benchmarks were GPU-only.

**How to apply:** don't swap detectors on this class of CPU. Re-evaluate if prod hardware is Intel Xeon with AVX-512-VNNI, or GPU with spare VRAM for layout. `docker/cpu/tests/probe_doclayout_yolo.py` is in the dev repo to re-run when relevant.

---

## Project: CUDA IPC Transport + MTP findings 2026-04-23 (project)

MTP (Multi-Token Prediction) is already effectively on — SGLang internally aliases `SGL_SPEC_ALGORITHM=NEXTN` to `EAGLE` with matching sub-knobs. Don't chase MTP as a separate optimization.

CUDA IPC Transport (`SGLANG_USE_CUDA_IPC_TRANSPORT=1`) crash-loops on 8 GB dev 3060 Ti because `MmItemMemoryPool` needs multi-GB contiguous VRAM on top of the model + KV cache. Tested at mem_fraction 0.95 / 0.82 / 0.70 — all OOMed.

**How to apply in prod:** the plumbing is in the container (see `docker/sglang/` mirror of dev's `docker/gpu/`). On g4dn.2xlarge (16 GB VRAM) the pool may fit or may not — flip `SGLANG_USE_CUDA_IPC_TRANSPORT=1` after the endpoint is `InService`, watch CloudWatch logs for OOM, and roll back if it fails. See `09-runbook.md` §Future work.

---

## Project: FastAPI async sidecar rejected 2026-04-23 (project)

`LAYOUT_ASYNC=true` net-neutral to worse vs gunicorn baseline. Introduced HealthWatchdog 500s at c=40 (34 fails) + ServerDisconnectedError at c=64 (11 fails). Running-count peaks unchanged (9–11 vs baseline 9–16) — confirms HTTP admission isn't the lever.

**Why it broke:** async removes gunicorn's implicit `CPU_WORKERS × CPU_THREADS = 64` gthread back-pressure. SGLang saturated past its comfort threshold.

**How to apply in prod:** keep `LAYOUT_ASYNC=false`. Don't spend time on LitServe / RayServe; they hit the same CPU-kernel ceiling.

---

## Global instruction — honor the tuned defaults

Across all these memories the pattern is the same: **the dev session spent ~3 weeks narrowing a wide hyperparameter space to one concrete `.env` block** (see `env-tuned.md`). Prod inherits this. Changes to these values require a new hypothesis, not a fresh exploration.

If prompted to tune for perf in prod, the first answer is always: "Run two matrix runs on the current config as a baseline. Then change ONE knob, run two more. Compare." Not: "Let me try six things."
