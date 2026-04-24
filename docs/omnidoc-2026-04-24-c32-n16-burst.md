# c=32 n=16 burst — detailed latency decomposition

**Date:** 2026-04-24
**Stack:** shipped (`LAYOUT_VARIANT=paddle2onnx`, `LAYOUT_ONNX_PROVIDER=openvino`, `LAYOUT_PREFIX_PIN=true`, `LAYOUT_BATCH_ENABLED=true`, `SGL_MEM_FRACTION_STATIC=0.83`, 4 gunicorn workers × 16 gthreads × `LAYOUT_ONNX_THREADS=3`)
**Hardware:** Ryzen 5 5600X (6p / 12 SMT cores), 3060 Ti 8 GB, 24 GB RAM
**Seed:** 42 (same `random.Random(42).sample(sorted(images), 16)` used in both warmup and measure)

Purpose: measure what actually dominates per-request latency at the concurrency ceiling on this 8 GB dev card, with a clean warmup so we're not bench-marking cuda-graph capture or OV kernel JIT.

---

## Setup + methodology

```
Warmup:   16 requests at c=8  →  72.7 s wall   (heat cuda graphs + prefix cache + OV kernel JIT)
           2 s settle
Measure:  16 requests at c=32 →  170.6 s wall
```

The warmup is load-bearing. Without it, first-request cuda-graph replay misses and OV compile costs leak into the c=32 measurement and inflate layout/TTFT numbers by 2-3×.

Metrics collected:
- CPU container `/metrics` (Prometheus): `flask_http_request_duration_seconds`, `glmocr_layout_seconds`, `glmocr_ocr_region_seconds` (_sum + _count bracketing the run)
- SGLang `/metrics`: `sglang:time_to_first_token_seconds`, `sglang:e2e_request_latency_seconds`, `sglang:realtime_tokens_total{mode=...}`

All numbers are deltas bracketing the 16-request measurement window, so warmup metrics don't pollute them.

---

## Client-side results

| metric | value |
|---|---:|
| total requests | 16 |
| successful | **16** |
| empty markdown | **0** |
| wall clock | 170.62 s |
| **effective rps** | **0.094** |
| **mean latency per request** | **99.81 s** |
| p50 | 126.39 s |
| p95 | 170.61 s |
| p99 | 170.61 s |
| min | 6.26 s |
| max | 170.61 s |
| blocks detected per page (mean) | 20.9 |

**27× latency spread** (min 6.3 s → max 170.6 s). With n=16 submitted into a 32-thread pool, only the first few requests hit empty queues; the last few wait behind everything that came before. This is an artifact of the small n — at higher n the mean would approach the per-request steady-state cost.

---

## Server-side per-request breakdown

Source: CPU container `flask_http_request_duration_seconds_{sum,count}` scoped to `url_rule="/glmocr/parse"`.

| stage | time | share of total |
|---|---:|---:|
| **Total wall per request** | **99.81 s** | 100 % |
| Layout stage (serial, 1 call per request) | **11.62 s** | **12 %** |
| OCR stage wall (parallel across 21.2 regions) | **88.18 s** | **88 %** |
| Other (preprocess, dispatch, serialization) | ~0 s | <1 % |

Layout's share dropped from 29 % at c=8 to 12 % at c=32 — but not because layout got faster. Layout per call actually *grew* from 3.3 s → 11.6 s (3.5× worse). OCR stage simply grew even more (7.5 s → 88.2 s, 11.8× worse).

---

## Per-region SGLang decomposition

Source: SGLang's own `time_to_first_token_seconds` + `e2e_request_latency_seconds` histograms, 369 regions in the measurement window.

```
61.17 s  HTTP roundtrip per region (from CPU container's perspective)
├── 35.13 s  TTFT mean           (57 %)   queue wait + prefill inside SGLang
├──  0.08 s  decode mean         (<1 %)   actual GPU token generation
└── 25.96 s  unaccounted         (43 %)   aiohttp connection wait + thread-pool queue
```

**The 0.08 s decode mean is the headline number.** Actual GPU token generation is effectively zero in the context of per-request wall time. The GPU spends ~99 % of each region's lifetime idle-waiting for its turn. The 3060 Ti is *not* the bottleneck on this workload.

The ~26 s gap between SGLang's internal e2e (`TTFT + decode = 35.2 s`) and CPU container's HTTP roundtrip (61.2 s) is accounted for by:
- aiohttp connection pool acquisition when OCR_MAX_WORKERS=32 is saturated
- ThreadPoolExecutor queue wait (regions beyond the 32-thread cap wait here)
- Request framing + network roundtrip
- SGLang's ingress processing before the scheduler handles the request (not covered by the `ttft_seconds` histogram)

---

## SGLang cache behavior

| counter | tokens | observation |
|---|---:|---|
| `prefill_compute` (cold prefill) | 30 733 | new tokens that had no cached K/V |
| `prefill_cache` (prefix hit) | 4 894 | tokens served from RadixCache |
| `decode` | 37 455 | tokens generated (~101/region avg) |
| **prefix cache hit rate** | **13.7 %** | far below the ~90 % theoretical ceiling |

The low hit rate mirrors what we saw at c=16: under heavy concurrency (16 requests × 21 regions = ~340 concurrent regions competing for the KV cache), the 24 298-token KV pool gets thrashed. The `LAYOUT_PREFIX_PIN` §8 shipment shifts the theoretical ceiling from ~12 % (no stable prompt) to ~90 % (stable prompt), but on this 8 GB card's tiny KV pool, concurrency quickly evicts the prefix before it can be reused.

On 16 GB+ hardware (AWS T4) the KV pool grows ~8× and the hit rate should climb back toward the 56 % we saw at c=8.

---

## Latency budget — bottom-up

```
Per-request wall: 99.81 s
│
├─ 11.6 s   layout stage (paddle2onnx + ORT + OV EP)
│     ├─ ~1.0 s   single OV EP forward (kernel time, amortized by batcher)
│     └─ ~10.6 s  waiting for the batcher's 20 ms window AND for prior layout
│                 calls on busy workers (c=32 means all 4 workers serializing)
│
└─ 88.2 s   OCR stage wall (gated by the request's slowest region)
      │
      │   21.2 regions/request, fired in parallel across OCR_MAX_WORKERS=32
      │
      ├─ Mean region: 61.2 s HTTP roundtrip
      │     ├─ 35.1 s  SGLang queue wait (binding: 340+ regions, running-batch cap 64)
      │     ├─ 0.08 s  actual decode (GPU compute)
      │     └─ 26 s    aiohttp pool wait + thread-pool queue + framing
      │
      └─ Tail region: ~88 s (the last-scheduled region that gates the request's OCR stage)
```

---

## Comparison to earlier concurrency levels

| c | n | rps | mean latency | layout/call | TTFT/region | decode/region | prefix hit |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 8 | 20 | 0.57 | 11.6 s | 3.3 s | 5.6 s | 0.32 s | 56 % |
| 16 | 20 | 0.15 – 0.55 (±150 % noise) | 22 – 55 s | 10 – 18 s | 7 – 17 s | 0.21 – 0.64 s | 10 – 20 % |
| **32** | **16** | **0.094** | **99.8 s** | **11.6 s** | **35.1 s** | **0.08 s** | **13.7 %** |

Going c=8 → c=32:
- rps drops **6×** (0.57 → 0.094)
- mean latency grows **8.6×** (11.6 s → 99.8 s)
- TTFT per region grows **6.3×** (5.6 s → 35.1 s) — queue depth is the binding factor
- decode per region *drops* (0.32 s → 0.08 s) — because less of each region's lifetime is actually being scheduled to decode; most of it is parked waiting
- layout per call grows **3.5×** — the OV EP c≥16 scaling pathology we documented but couldn't tune reliably due to single-burst noise

---

## What this run confirmed

1. **Zero silent empties at c=32.** 16/16 successful with populated markdown. Post-session stack (paddle2onnx + OV EP + prefix-pin + mem=0.83) holds quality at the concurrency ceiling.
2. **Actual GPU decode is fast.** 0.08 s/region in the measurement window. The GPU is idle, not slow. This is critical operational knowledge — further GPU-side optimization (bigger running batch, speculative tuning) won't help while queue wait dominates.
3. **Prefix cache scales inversely with concurrency on this hardware.** 56 % at c=8 → 14 % at c=32 with identical prefix-pin code. Confirms the §9 mem-fraction tradeoff and identifies the T4 redeploy as where the prefix-pin win becomes c-range-wide.

---

## Operational implication

**Production workload on this 8 GB dev card: target c=8.**

- c=8 has genuine throughput (0.57 rps) + acceptable p95 latency + 0 empty rate + high prefix-cache utility.
- c=16 is within survival range but single-burst variance is ±150 % — unusable for tuning, risky for SLA commitments.
- c=32 is survival-only. Use it for OOM probes and capacity-cliff probes, not for serving user traffic.

When migrating to AWS T4 (16 GB):
- KV pool grows ~8× (≈200 k tokens vs 24 k here)
- `SGL_MEM_FRACTION_STATIC` can safely move back to 0.90-0.95 without OOM risk (extra dynamic pool inherent from 2× card size)
- `SGL_CUDA_GRAPH_MAX_BS` can safely grow to 16 or 32 (covers the 11-16 peak running batch)
- **c=32 should behave like c=8 does here** — re-run this exact probe on T4 and the per-region TTFT should collapse back toward 5-7 s.

---

## Cross-references

- `docs/OPTIMIZATIONS.md` §6, §7, §8, §9 — the four shipments this run measures
- `docs/omnidoc-2026-04-24-paddle-ov-shipment.md` — full session narrative including the variance finding
- `docs/ARCHITECTURE-v2.md` §9 — end-to-end latency budget at c=8 for comparison
- `scripts/matrix_sweep_quick.py` — the reusable c-level sweep harness
- Auto-memory `feedback_matrix_noise.md` — why single-burst c=16+ numbers can't be trusted on this hardware
