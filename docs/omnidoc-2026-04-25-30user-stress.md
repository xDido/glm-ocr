# 30-user × 16-page concurrent stress test — 2026-04-25

Worst-case load test against the fully-shipped stack (post-§6-§12). 30 concurrent users, each posting one HTTP request containing 16 distinct OmniDocBench pages. 480 unique pages total, partitioned 16-per-user with `seed=42` so users never share inputs — this is the **opposite** of a cache-friendly workload.

## Configuration measured

```ini
LAYOUT_VARIANT=paddle2onnx          # §6
LAYOUT_ONNX_PROVIDER=openvino       # §7
LAYOUT_PREFIX_PIN=true              # §8
SGL_MEM_FRACTION_STATIC=0.83        # §9
PAGE_LOADER_MAX_PIXELS=262144       # §10
PROMPT_TEXT=OCR: / Table: / Formula:  # §11
SGL_CUDA_GRAPH_MAX_BS=16            # §12

CPU_WORKERS=4 × CPU_THREADS=16        (gthread)
LAYOUT_ONNX_THREADS=3
OCR_MAX_WORKERS=32 (intra-request)
LAYOUT_BATCH_ENABLED=true (batcher coalesces across concurrent requests)

SGL_MAX_RUNNING_REQUESTS=64
SGL_SCHEDULE_POLICY=lpm
SGL_SPECULATIVE=true (NEXTN, num_steps=3)
```

**Hardware**: Ryzen 5 5600X / NVIDIA 3060 Ti 8 GB / 24 GB RAM.

## Top-line outcome

| metric | value |
|---|---:|
| total wall clock | **1241.9 s** (20.7 min) |
| successful users | **30 / 30** |
| failed users | 0 |
| total pages OCR'd | **480 / 480** |
| effective document rps | 0.024 (one doc per 41 s) |
| effective page rps | 0.386 (one page per 2.6 s) |
| effective block rps | 0.39 |
| total markdown chars | 672 437 |

**Zero failures. Zero silent empties. Every user got their 16-page document back.** Pre-session this would have crashed SGLang multiple times over.

## Per-user latency

| metric | value |
|---|---:|
| mean | **1063.9 s** (17.7 min) |
| p50 | 1075.6 s |
| p95 | 1196.8 s |
| p99 | 1241.9 s |
| min | 821.2 s (the lucky-fast user that got cache-empty queues) |
| max | 1241.9 s |

The first user finished at 821 s, the last at 1242 s — a 1.5× spread. Users that fired into a fully-loaded queue paid the 51 % more than the lead users.

## Server-side decomposition

### CPU container (per `/glmocr/parse` request, mean over 30 reqs)

| stage | n | mean |
|---|---:|---:|
| flask_http (total per-request wall) | 30 | 1064.0 s |
| glmocr_layout (per page) | 480 | **23.5 s / call** |
| glmocr_ocr_region (HTTP roundtrip per region) | 9 654 | **94.6 s / region** |
| regions per request | — | 321.8 |
| pages per request | — | 16.0 |

The 30 users together generated **9 654 OCR region calls** to SGLang (with retries: see SGLang section below where `e2e_n=11 865`). Each region waited ~95 s on average for SGLang to handle it.

### SGLang (per region)

| stage | n | mean |
|---|---:|---:|
| e2e latency (queue + prefill + decode) | 11 865 | 54.9 s |
| **TTFT (queue + prefill)** | 11 865 | **54.6 s** |
| **decode (actual GPU token generation)** | — | **0.31 s** |

99.4 % of every region's lifetime is queue wait. **Real GPU compute is 310 ms per region** — fast, idle most of the time.

### Token-throughput accounting

| token class | tokens |
|---|---:|
| prefill_compute (cold prefill) | **1 036 474** |
| prefill_cache (RadixCache hit) | 84 331 |
| decode (generated) | 601 001 |
| **prefix cache hit rate** | **7.5 %** |

This is the worst-case for prefix caching by design: **30 users × 16 distinct pages = 480 unique pages** with zero overlap. The only repeated tokens are the chat-template wrapper plus the literal `OCR:` / `Table:` / `Formula:` prompts. Hit rate at 7.5 % shows even those tiny stable bits get evicted under the 11 800-region barrage on a 24k-token KV cache.

## What this measurement reveals

### 1. The stack is correctness-stable far past its throughput sweet spot

We ran 11 865 SGLang requests over 20 minutes with **zero empties, zero TCP errors, zero SGLang crashes**. The §6-§9 work (Paddle2ONNX + OV EP + prefix-pin + mem=0.83) did its job: even under load this extreme, the failure modes from the earlier sessions (silent empty markdown at c≥8, SGLang OOM at c≥24) didn't surface.

### 2. Throughput is wall-clock-bounded, not user-count-bounded

Each user took ~17.7 minutes. Adding more users wouldn't reduce that — it would just lengthen the tail. The 8 GB card processes pages at a fixed rate (about 0.39 pages/sec under saturation); 30 users sharing that = 16 pages × 30 users / 0.39 pages/sec / 30 parallel ≈ 16 / 0.39 ≈ 41 s if perfect parallelism, but real serialization on layout + queue depth pushes per-user latency to 17 minutes.

### 3. Prefix cache collapses to near-zero on adversarial inputs

7.5 % hit rate vs the 40 % we measure on cache-friendly single-user repeated workloads. This is data: the §8 prefix-pin shipment helps when traffic has prompt-prefix locality (sustained traffic, repeated documents), but doesn't help when 30 users hammer disjoint document sets. **On the AWS T4 (16 GB) the cache is 8× larger and even adversarial workloads should retain >30 % hit rate** because the cache holds enough state to keep the chat-template wrapper resident through cycles of eviction.

### 4. Layout is now bursty, not binding

Layout per call grew to 23.5 s under c=30 stress (vs 4-9 s at c=8 warm), but SGLang queue (54.6 s TTFT) is still 2.3× larger. **Layout isn't the throughput limit anymore** — it was at the start of the session, before §10/§11/§12 cut prefill-token-per-region by 75 %. Under this load profile, layout-side optimization no longer moves the needle; everything left is GPU/cache-side and hardware-bounded.

### 5. The retry layer is firing

SGLang served 11 865 requests for 9 654 OCR regions — a **23 % retry rate**. `OCR_RETRY_MAX=2` is correctly absorbing transient slowness, but at the cost of ~25 % wasted tokens. On hardware where the cache holds together, this would settle to <5 %.

## Operational implication

**This card cannot serve 30 concurrent users in real time.** It can serve them eventually — every request completes — but the latency budget at this load is 17 minutes per document, not 30 seconds.

For the same workload at production-acceptable latency:
- **c ≤ 8 concurrent users on this 8 GB card.** Each gets ~5-15 minutes for a 16-page document (single page at warm-state c=8 is ~14 s; 16 pages serialized through layout is ~3-4 minutes; OCR fans out so doesn't add proportionally).
- **c ≤ 32 concurrent users on a 16 GB T4.** With 8× more KV cache, prefix hit rate should sustain >30 % even on adversarial inputs, and SGLang's running batch can comfortably hit 32+. The same 30 × 16 stress should run in under 5 minutes wall.

## Latency breakdown — bottom-up

```
Per-user wall: 1063.9 s
│
└─ 16 pages × ~20 regions = ~320 SGLang calls per user
      │
      ├─ Layout stage (per page): 23.5 s × 16 pages = 376 s gross,
      │     but actually amortized by the cross-request batcher down
      │     to ~150-200 s of layout-stage wall per user
      │
      ├─ OCR stage (per region): 94.6 s mean
      │     ├─ 54.6 s SGLang TTFT (queue wait + prefill)
      │     ├─ 0.31 s actual decode
      │     └─ ~40 s aiohttp pool wait + thread-pool queue + retry overhead
      │
      └─ With OCR_MAX_WORKERS=32 the per-user OCR stage wall is
            max(region_times) ≈ 1042 s — the slowest region's wait
            sets the request's OCR-stage budget.
```

The dominant term is **SGLang queue wait per region (54.6 s)**, repeated until the slowest region finishes (~17 minutes). Layout is no longer the binding stage; GPU queue depth is.

## Cross-references

- `docs/OPTIMIZATIONS.md` §6-§12 — the 12 numbered shipments this run validates
- `docs/omnidoc-2026-04-25-final-matrix.md` — c=8/16/32 baseline measurements (single-page requests) for comparison
- `docs/ARCHITECTURE-v2.md` — request lifecycle reference
- `loadtest/results/stress-30u-16d-2026-04-25.json` — raw run data
- `scripts/stress_30users_16docs.py` — the test harness
