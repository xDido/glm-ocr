# Step 1 + Step 1.5 detailed report — `OCR_MAX_TOKENS=2048` + `SGL_CUDA_GRAPH_MAX_BS=32`

**Date:** 2026-04-26 (warmup-cleaned re-run)
**Hardware:** RTX 3060 Ti 8 GB · Ryzen 5600X · 32 GB RAM
**Goal:** push `sglang:num_running_reqs` peak from baseline 11–16 → ≥25.
**Settings:**
- `OCR_MAX_TOKENS=2048` (caps per-region completion ceiling sent to SGLang)
- `SGL_CUDA_GRAPH_MAX_BS=32` (covers fast-path decode for the new running band)
- `MATRIX_WARMUP=8` per trial (excluded from latency stats by `bench.py:62`)

**Source report:** `loadtest/results/omnidoc-20260426-115540-asyncio-matrix.md`
**Containers:** sglang + cpu force-recreated immediately before this run for clean state.

---

## Headline

| metric | baseline | **Step 1+1.5 (this run)** |
|---|---:|---:|
| running peak c=12–c=40 | ~14 | **56–59** (4× target) |
| inter-token p99, c=12–c=32 | unknown / >992 ms (eager) | **93–97 ms (CUDA graph)** |
| correctness c=12, 24, 40 | 100 % | **100 %** |
| correctness c=32 | unknown | **98 %** (2 transient HealthWatchdog 500s) |
| correctness c=64 | 100 % (run-1) | **29 %** (HealthWatchdog cascade — beyond capacity) |

**Conclusion: c=12 through c=40 all pass with running batch sustained at 30s and peaking at 56–59.** c=64 is the new ceiling for this 8 GB GPU; the previous c=32 collapse from the prior matrix is gone with proper warmup + clean container state.

---

## Overall request latency per trial (bench-driver, client-perceived, ms)

_End-to-end per-`/glmocr/parse` HTTP call as observed by `loadtest/asyncio/bench.py`. Includes everything: layout, recognition fan-out, marshalling, network. **8 warmup requests per trial are excluded from these stats.** Failed requests excluded from latency stats._

| c | ok | fail | wall (s) | rps | **mean** | min | **p50** | p90 | **p95** | **p99** | max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 100 | 0 | 221.2 | **0.452** | **25,652** | 2,773 | **16,638** | 51,304 | **75,739** | **146,040** | 176,413 |
| 24 | 100 | 0 | 240.0 | **0.417** | **52,827** | 2,850 | **39,769** | 93,660 | **142,023** | **234,671** | 237,590 |
| 32 | 98 | 2 | 236.5 | **0.414** | **66,562** | 8,210 | **47,032** | 147,319 | **197,731** | **217,794** | 230,231 |
| 40 | 100 | 0 | 279.1 | **0.358** | **98,991** | 4,459 | **87,063** | 183,336 | **221,073** | **273,658** | 279,096 |
| 64 | 29 | 71 | 158.1 | 0.183 | 58,380 | 11,416 | 58,667 | 79,261 | 104,616 | 127,885 | 130,397 |

### Reading the latencies

- **c=12**: median 16.6 s, mean 26 s. p99 = 146 s. Healthy.
- **c=24**: median 40 s, mean 53 s. p99 = 235 s. Tail nearly doubles vs c=12 because SGLang queue wait grew 4.6× per region.
- **c=32**: median 47 s, mean 67 s. p99 = 218 s. 2 transient HealthWatchdog 500s but otherwise stable. Big improvement vs the previous run (10/100 ok) — clean state + 8 warmup made the difference.
- **c=40**: median 87 s, mean 99 s. p99 = 274 s. Still 100% completion. **First successful c=40 result on this stack.**
- **c=64**: collapses with HealthWatchdog cascade — running batch couldn't sustain admission rate because /health stalls under prefill load.

### Comparison vs run-1 (cap=2048, no Step 1.5, no observability) and run-warmup=2 (Step 1+1.5 with 2-warmup)

| c | run-1 p50 | run-warmup=2 p50 | **this run p50** | run-1 p99 | run-warmup=2 p99 | **this run p99** |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 17,267 | 15,373 | **16,638** | 198,896 | 129,925 | **146,040** |
| 24 | 35,840 | 37,713 | **39,769** | 208,755 | 204,413 | **234,671** |
| 32 | 49,783 | 223,217 (collapse) | **47,032** | 205,077 | 224,462 (collapse) | **217,794** |
| 40 | 62,280 | — (collapse) | **87,063** | 215,865 | — (collapse) | **273,658** |
| 64 | 111,349 | — (collapse) | 58,667 (29% only) | 232,758 | — (collapse) | 127,885 (29% only) |

c=12, c=24, c=32 are all in the noise band of run-1. c=40 is slightly slower at p50 (87 vs 62 s) because the new operating point is a higher running batch (59 vs 11–16) — extra concurrent slots = longer queue wait per slot. **All trials c=12 through c=40 are now in the "shippable" envelope, which run-1 only achieved by accident at c=12.**

---

## Per-trial running/queue (from `sglang:num_running_reqs` and `sglang:num_queue_reqs`, step=1 s)

| Trial | running **peak** | running p95 | running mean | queue peak | queue mean | correctness |
|---|---:|---:|---:|---:|---:|---|
| c=12 | **56** | 53 | 31.8 | 164 | 42.4 | 100/100 ✓ |
| c=24 | **58** | 54 | 31.4 | 383 | 180.1 | 100/100 ✓ |
| c=32 | **55** | 53 | 30.8 | 468 | 196.6 | 98/100 ✓ |
| c=40 | **59** | 55 | 36.1 | 579 | 298.9 | 100/100 ✓ |
| c=64 | 27 | 24 | 12.9 | 118 | 30.8 | 29/100 ✗ |

c=64 anomaly: running peak only 27, mean 12.9 because the CPU container's HealthWatchdog tripped early and most requests returned 500 fast — SGLang barely got fed. This is the same "scheduler can't admit because /health is unresponsive" pattern from prior runs at c=32 (when state was bad).

---

## c=12 — phase decomposition (running peak 56, 100/100 ok, 0.45 rps)

### CPU container

| Phase | scope | mean ms | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| Flask end-to-end | per HTTP request | 24,878 | 16,400 | 98,250 | 180,000 |
| Layout forward (ONNX + pre/post) | per HTTP request | 3,378 | 1,740 | 15,000 | 19,000 |
| OCR region call (CPU→SGL→back) | per region | **7,758** | 6,503 | 26,199 | 30,585 |

### SGLang container

| Phase | scope | mean ms | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| SGLang end-to-end | per SGLang request | 7,471 | 5,960 | 19,323 | 39,776 |
| Queue wait (scheduler) | per SGLang request | **3,736** | 3,372 | 9,559 | 12,841 |
| TTFT (prefill + 1st decode) | per SGLang request | 6,658 | 5,682 | 18,240 | 30,434 |
| **Inter-token latency** | per decoded token | **30** | 25 | 73 | 96 |

---

## c=24 — phase decomposition (running peak 58, 100/100 ok, 0.42 rps) — **the sweet spot**

### CPU container

| Phase | scope | mean ms | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| Flask end-to-end | per HTTP request | 47,570 | 37,857 | 156,000 | 180,000 |
| Layout forward (ONNX + pre/post) | per HTTP request | 7,530 | 1,645 | 20,000 | 20,000 |
| OCR region call (CPU→SGL→back) | per region | **20,946** | 20,985 | 52,534 | 59,312 |

### SGLang container

| Phase | scope | mean ms | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| SGLang end-to-end | per SGLang request | 20,650 | 18,633 | 38,884 | 64,473 |
| Queue wait (scheduler) | per SGLang request | **17,218** | 17,535 | 34,804 | 38,961 |
| TTFT (prefill + 1st decode) | per SGLang request | 19,631 | 18,128 | 37,844 | 39,797 |
| **Inter-token latency** | per decoded token | **33** | 29 | 76 | 97 |

---

## c=32 — phase decomposition (running peak 55, 98/100 ok, 0.41 rps)

### CPU container

| Phase | scope | mean ms | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| Flask end-to-end | per HTTP request | 57,908 | 41,667 | 180,000 | 180,000 |
| Layout forward (ONNX + pre/post) | per HTTP request | 11,819 | 2,429 | 20,000 | 20,000 |
| OCR region call (CPU→SGL→back) | per region | **22,947** | 20,691 | 53,461 | 59,418 |

### SGLang container

| Phase | scope | mean ms | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| SGLang end-to-end | per SGLang request | 22,145 | 23,256 | 48,393 | 65,773 |
| Queue wait (scheduler) | per SGLang request | **18,844** | 19,691 | 36,844 | 43,598 |
| TTFT (prefill + 1st decode) | per SGLang request | 21,182 | 22,585 | 41,582 | 56,543 |
| **Inter-token latency** | per decoded token | **31** | 29 | 72 | 93 |

---

## c=40 — phase decomposition (running peak 59, 100/100 ok, 0.36 rps) — **first successful c=40**

### CPU container

| Phase | scope | mean ms | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| Flask end-to-end | per HTTP request | 83,631 | 66,000 | 180,000 | 180,000 |
| Layout forward (ONNX + pre/post) | per HTTP request | 18,309 | 2,300 | 20,000 | 20,000 |
| OCR region call (CPU→SGL→back) | per region | **35,579** | 38,641 | 58,941 | 60,000 |

### SGLang container

| Phase | scope | mean ms | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| SGLang end-to-end | per SGLang request | 33,563 | 32,635 | 58,509 | 73,225 |
| Queue wait (scheduler) | per SGLang request | **29,551** | 30,775 | 54,707 | 58,941 |
| TTFT (prefill + 1st decode) | per SGLang request | 32,804 | 32,020 | 57,422 | 67,558 |
| **Inter-token latency** | per decoded token | **35** | 30 | 87 | 179 |

Note: c=40 inter-token p99 climbed to 179 ms (vs 93–97 ms at lower c). With running batch occasionally exceeding 32, some decode steps miss CUDA graphs and fall to eager. This is mild — affects p99 but not mean.

---

## c=64 — collapse (29/100 ok, 71 HealthWatchdog 500s)

| Trial-window mean signals | value |
|---|---:|
| sglang running mean | 12.9 |
| sglang running peak | 27 |
| in-flight peak (CPU) | 50 |
| OCR region p99 | 36,709 ms |
| Flask e2e p99 | 111,000 ms |

c=64 is beyond capacity. The CPU container's HealthWatchdog trips early, returning 500 to clients before requests reach SGLang in any sustained way. Running peak of 27 is misleading — it's the brief window before cascade started; mean 12.9 reflects the large fraction of time SGLang sat idle while CPU was failing fast.

---

## Bottleneck summary — where time goes per OCR region (c=24, the operating point)

```
Per region wall time = 21,000 ms (mean)
├── 17,200 ms  SGLang queue wait                   82 %  ← running batch saturated
├──  2,400 ms  SGLang prefill                      11 %
├──  1,000 ms  SGLang decode (33 ms/tok, fast path) 5 %
└──    400 ms  network + crop + base64 + JSON       2 %
```

Per request (one page, ~14 regions):
```
Flask end-to-end = 47,570 ms
├──  7,500 ms   Layout forward (ONNX + pre/post)
└── 40,000 ms   Recognition fan-out (parallel 14 regions @ OCR_MAX_WORKERS=32)
```

Recognition is 84 % of the per-request wall time, and SGLang queue wait is 82 % of recognition. **The dominant bottleneck across c=12 to c=40 is the SGLang scheduler queue wait — slots fill at running ~58 and additional regions wait.** Per-token decode is healthy across the entire range (CUDA graph fast path holds).

---

## Inter-token latency across the sweep — proof CUDA graph fast path is working

| c | running peak | inter-token mean | inter-token p99 |
|---:|---:|---:|---:|
| 12 | 56 | 30 ms | 96 ms |
| 24 | 58 | 33 ms | 97 ms |
| 32 | 55 | 31 ms | 93 ms |
| 40 | 59 | 35 ms | 179 ms |
| 64 | 27 (collapse) | 21 ms | 170 ms |

Pre-Step-1.5 baseline (Step 1 alone with bs=16): inter-token p99 was 992 ms because running 17–60 fell to eager kernels. **Step 1.5 cut decode tail latency 10×.**

c=40 inter-token p99 = 179 ms is the first sign the cuda_graph_max_bs=32 cap is being exceeded. A future bump to bs=48 or bs=56 would extend the fast path further at additional VRAM cost — but only worth it if c=40 becomes a target operating point.

---

## What to try next, if you want c=64 to stop collapsing

In order of recommended first try:

1. **Cap CPU-side concurrency at c=40 in production.** It's the sweet spot we just proved. A semaphore on `/glmocr/parse` would do it. Highest-leverage, simplest change.

2. **Loosen HealthWatchdog tolerance** in glmocr (upstream code). Probably 30 s default; raise to 60–90 s to ride out SGLang's prefill bursts. Would unlock c=48–c=64.

3. **Drop `OCR_MAX_TOKENS` further to 1024 or 512.** Mean per-region gen tokens = 73 (per the augment); p95 likely under 256. Smaller cap = tighter per-slot KV reservation = even more admit headroom under tight conditions.

4. **Increase SGLang prefill throughput.** Smaller image crops via `PAGE_LOADER_MAX_PIXELS` (currently 262144 / 512²) — push to 196608 (440²). Each region's prefill shrinks → queue drains faster.

5. **For 16 GB+ GPU**: re-enable `SGL_CUDA_GRAPH_MAX_BS=64` to extend fast-path coverage to running 48–60. On 8 GB this OOMs; on 16 GB it fits comfortably.

NOT recommended:

- Original plan's Step 2 (`SGL_CONTEXT_LENGTH=6144`) — at c=24+ the binding bottleneck is queue wait, not per-slot KV reservation.
- Original plan's Step 3 (`mem_fraction_static=0.90`) — graph state already tight, OOM risk.
- Disable speculative decoding — would cut decode throughput 2-2.5×.

---

## Reproduction & rollback

Current `.env`:
```
OCR_MAX_TOKENS=2048
SGL_CUDA_GRAPH_MAX_BS=32
```

Current matrix harness:
```
MATRIX_WARMUP=8     # configurable, default 8 (was 2 hardcoded before today)
```

Rollback:
```
OCR_MAX_TOKENS=0     # or unset; restores glmocr's 8192 default
SGL_CUDA_GRAPH_MAX_BS=16
```
Then `docker compose up -d --force-recreate sglang cpu`.

All knobs are pure env flips on top of the shipped image — no code changes.
