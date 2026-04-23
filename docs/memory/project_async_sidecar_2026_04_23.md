---
name: FastAPI async sidecar tested, rejected — 2026-04-23
description: Flipped LAYOUT_ASYNC=true (uvicorn sidecar on :5003) and re-ran MATRIX_TOTAL=200. Net-neutral-to-worse vs gunicorn/Flask baseline, and introduced HealthWatchdog 500s at c=40 (34 fails) + ServerDisconnectedError at c=64 (11 fails). Running count peaks unchanged (9-11 vs baseline 9-16). Reverted. The LitServe/RayServe rungs above FastAPI would hit the same ceiling.
type: project
originSessionId: c8845dae-98e9-4b22-a7ae-02787ef2bf30
---
**Reports:** baseline `loadtest/results/omnidoc-20260423-094350-asyncio-matrix.md`; async `loadtest/results/omnidoc-20260423-161810-asyncio-matrix.md`; side-by-side in `loadtest/results/comparison-20260423-async-sidecar.md`.

## Throughput

| c | baseline rps | async rps | Δ | fails |
|--:|--:|--:|--:|--|
| 12 | 3.124 | 2.824 | −10% (noise) | 0 / 0 |
| 24 | 3.595 | 2.356 | **−34%** | 0 / 0 |
| 32 | 2.741 | 3.166 | +15% (noise) | 0 / 0 |
| 40 | 2.373 | 1.724 | **−27%** | 0 / **34** |
| 64 | 2.492 | 2.812 | +13% | 0 / **11** |

## Running count unchanged

Async sidecar peak sglang-running was 9-11 across c=12..c=64 — same band as baseline. Framework-level admission control does NOT raise the in-GPU batch. Consistent with the Little's-Law reading from the earlier GPU-utilization study: the gating constraint is CPU-side layout forward residence time, not HTTP admission.

## Why c=40/c=64 broke

Async removes gunicorn's implicit `CPU_WORKERS × CPU_THREADS = 64` gthread back-pressure. At c=40 the event loop admitted all 40 concurrent requests, each fan-out multiplied by `OCR_MAX_WORKERS=32` into SGLang, and SGLang saturated past the healthwatchdog threshold. We were inadvertently using gunicorn as a surge-protection layer.

## Plumbing was already in tree

`docker/cpu/async_app.py` + `LAYOUT_ASYNC` guard in `docker/cpu/entrypoint.sh` + `ASYNC_PORT`/`ASYNC_WORKERS` in `docker-compose.yml` were all there from a prior session. Toggle is a one-line `.env` flip. No new code this round, only `.env` flip+revert.

## Takeaway

**Why:** per `project_layout_profile.md`, Conv dominates wall time (76%); per `project_gpu_utilization_2026_04_23.md`, the running count is throughput × residence-time and capped by CPU layout stage. HTTP framework swaps can't change kernel time. FastAPI (easiest rung) → LitServe (dynamic batching) → RayServe (heaviest) would all hit the same ceiling; LitServe's batcher would only replace our existing `LAYOUT_BATCH_ENABLED` coalescer (measured 5× uplift), not add to it.

**How to apply:** don't re-run async sidecar / LitServe / RayServe experiments on this hardware unless the CPU kernel itself got faster first (OpenVINO EP post-batch=1-fix, or a detector swap on different microarchitecture). If a reviewer proposes a framework swap for perf, point here.
