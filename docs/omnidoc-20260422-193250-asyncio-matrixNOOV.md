# GLM-OCR asyncio concurrency sweep — omnidoc-20260422-193250-asyncio-matrix

**Completed:** 2026-04-22 19:40:40 Egypt Standard Time
**Driver:** asyncio (`loadtest/asyncio/bench.py`)
**Endpoint:** `http://localhost:5002/glmocr/parse`
**Dataset:** OmniDocBench (128-image pool)
**Trials:** c=12, c=24, c=32, c=40, c=64

## Tuning recommendations (prod)

1. **Pin GPU clocks at host boot.** Prevents GPU frequency scaling from
   injecting 200-500 ms of idle-recovery variance into SGLang's
   generation loop — you see this as a fat first-request tail after
   idle. Add to the ASG's user-data script:

   ```bash
   # Enable persistence mode — driver stays loaded even when no process holds the GPU
   nvidia-smi -pm 1

   # Lock GPU clocks to max on the T4
   # T4 supported clocks: memory=5001 MHz, sm=585-1590 MHz
   nvidia-smi -lgc 1590,1590   # lock SM clock to max
   nvidia-smi -lmc 5001        # lock memory clock to max
   ```

   **Effect:** GPU stays at P0 (max clocks) forever, no throttling;
   eliminates the 200-500 ms idle-recovery penalty on the first request
   after quiet periods.

   **Tradeoff:** continuous power draw (same $/hour on AWS — billed by
   instance hour, not GPU work).

   **Requires:** root on the host. In ECS-EC2 this means user-data runs
   as root (fine). Fargate cannot do this (no host access), but SGLang
   runs on EC2 here so that's OK.

   Right fix for this workload — low effort, high impact, no latency
   regression elsewhere.

## Server runtime

| Var | env (.env) | actual (live) |
|---|---|---|
| `CPU_WORKERS` | 4 | 4 |
| `CPU_THREADS (per worker)` | 16 | [32, 32, 32, 32] |
| `OCR_MAX_WORKERS` | 32 | 32 |
| `SGL_MAX_RUNNING_REQUESTS` | 64 | 64 |
| `SGL_MAX_TOTAL_TOKENS` | 200000 | 200000 |
| `SGL_MAX_PREFILL_TOKENS` | 8192 | 8192 |
| `SGL_DTYPE` | — | — |
| `SGL_TP_SIZE` | — | — |
| `SGL_MEM_FRACTION_STATIC` | — | — |
| `SGL_MODEL_PATH` | — | — |

## Results — all levels

_Latencies in ms; rps counts successes only. The **ceiling** is the
concurrency where rps stops rising — past that, extra concurrency
just adds queuing and failures._

| c | interval (s) | ok | fail | fail % | wall s | rps | mean | min | p50 | p90 | p95 | p99 | max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | — | 200 | 0 | 100.0% | 75.6 | 2.646 | 4,441 | 1,187 | 3,143 | 8,844 | 10,335 | 12,761 | 19,856 |
| 24 | — | 200 | 0 | 100.0% | 60.7 | 3.293 | 7,038 | 1,137 | 5,145 | 14,870 | 19,720 | 21,282 | 22,840 |
| 32 | — | 200 | 0 | 100.0% | 64.7 | 3.092 | 9,701 | 1,111 | 5,810 | 24,007 | 33,656 | 42,225 | 42,251 |
| 40 | — | 200 | 0 | 100.0% | 66.4 | 3.014 | 12,082 | 1,494 | 9,279 | 25,350 | 36,800 | 39,779 | 47,811 |
| 64 | — | 200 | 0 | 100.0% | 76.7 | 2.606 | 18,526 | 2,003 | 9,993 | 50,609 | 51,900 | 56,486 | 56,501 |

## Errors per level

- **c=12**: no failures ✓
- **c=24**: no failures ✓
- **c=32**: no failures ✓
- **c=40**: no failures ✓
- **c=64**: no failures ✓

## Config tuning advisory

_Tracks observed peak usage against configured SGLang limits over the full matrix window, and advises whether each knob should be raised, lowered, or held. Numbers are read live from Prometheus._

| Knob | .env value | Observed | Advice |
|---|---:|---:|---|
| `sglang:max_total_num_tokens` (effective) | 37934 | peak used 16188 (43% util) | **Hold** — peak used 16188/37934 (43%). Comfortable middle band. |
| `SGL_CONTEXT_LENGTH` | 4096 (per .env) | mean req 232 tok (prompt 184 + gen 48) | **Consider 2048** — observed mean per-request total = 233 tok (prompt 184 + gen 49). Even 2x mean (465) fits 2048 with margin; setting context lower than the current 4096 would grow the KV pool further. |
| `SGL_MAX_RUNNING_REQUESTS` | (from .env) | peak 40 concurrent | Hold unless peak approaches the configured cap for long stretches. |

_KV-pool utilization > 90% = saturated (new requests queue); < 40% = over-provisioned (reclaim VRAM). Context length should sit ~2x the mean per-request total tokens — the current 4096 leaves a safety margin over typical OCR loads._


## Runtime signals + phase decomposition (retroactive, per trial)

_Pulled from Prometheus post-hoc. Each trial window is reconstructed from the matrix start timestamp + cumulative wall times in execution order. For each window we report two things: live gauges (in-flight / SGLang running / SGLang queued) and per-phase histogram statistics on the CPU and SGLang pipelines. Phase percentiles are computed within the scope of that phase's histogram (per HTTP request, per region call, or per decoded token)._

### c=12 (back-to-back)

**Worker concurrency (CPU container):**

| Metric | Value |
|---|---:|
| target concurrency (c) | 12 |
| in-flight mean | 11.1 |
| in-flight p95  | 16 |
| in-flight peak | 16 |
| probe samples  | 95 |


**SGLang state (from Prometheus, live gauges):**

| Signal | mean | p95 | peak |
|---|---:|---:|---:|
| sglang running (in-GPU batch) | 5.0 | 34 | 34 |
| sglang queued (waiting for slot) | 0 | 0 | 0 |


_Window: 1776879159–1776879253 (duration 94 s; step=1 s)_

#### c=12 (back-to-back) — Mean request phase decomposition

**CPU container**

| Phase | scope | mean ms | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| Flask end-to-end | per HTTP request | 4,840 | 3,943 | 15,410 | 19,082 |
| Layout forward (ONNX + pre/post) | per HTTP request | 4,517 | 3,307 | 10,757 | 14,588 |
| OCR region call | per region (N per HTTP request) | 3,253 | 2,757 | 9,540 | 25,301 |


**SGLang container**

| Phase | scope | mean ms | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| SGLang end-to-end | per SGLang request | 3,118 | 2,602 | 9,300 | 17,790 |
| Queue wait (scheduler) | per SGLang request | 12 | 50 | 95 | 99 |
| Time-to-first-token (prefill+first decode) | per SGLang request | 2,854 | 2,545 | 6,983 | 12,633 |
| Inter-token latency (decode step) | per decoded token | 15 | 8 | 53 | 61 |


_Values aggregate across all requests that completed inside the trial window. Scope indicates whether the histogram counts whole HTTP requests, per-region OCR fan-out calls, or per-token decode steps — percentiles are computed within that scope._

### c=24 (back-to-back)

**Worker concurrency (CPU container):**

| Metric | Value |
|---|---:|
| target concurrency (c) | 24 |
| in-flight mean | 21.4 |
| in-flight p95  | 24 |
| in-flight peak | 24 |
| probe samples  | 65 |


**SGLang state (from Prometheus, live gauges):**

| Signal | mean | p95 | peak |
|---|---:|---:|---:|
| sglang running (in-GPU batch) | 5.1 | 39 | 39 |
| sglang queued (waiting for slot) | 0.2 | 2 | 2 |


_Window: 1776879279–1776879343 (duration 64 s; step=1 s)_

#### c=24 (back-to-back) — Mean request phase decomposition

**CPU container**

| Phase | scope | mean ms | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| Flask end-to-end | per HTTP request | 6,793 | 4,819 | 19,742 | 27,788 |
| Layout forward (ONNX + pre/post) | per HTTP request | 6,693 | 4,846 | 19,621 | 20,000 |
| OCR region call | per region (N per HTTP request) | 5,192 | 5,347 | 9,535 | 9,907 |


**SGLang container**

| Phase | scope | mean ms | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| SGLang end-to-end | per SGLang request | 3,468 | 1,881 | 8,878 | 9,776 |
| Queue wait (scheduler) | per SGLang request | 33 | 52 | 99 | 748 |
| Time-to-first-token (prefill+first decode) | per SGLang request | 3,454 | 1,886 | 8,529 | 9,706 |
| Inter-token latency (decode step) | per decoded token | 11 | 10 | 19 | 20 |


_Values aggregate across all requests that completed inside the trial window. Scope indicates whether the histogram counts whole HTTP requests, per-region OCR fan-out calls, or per-token decode steps — percentiles are computed within that scope._

### c=32 (back-to-back)

**Worker concurrency (CPU container):**

| Metric | Value |
|---|---:|
| target concurrency (c) | 32 |
| in-flight mean | 23.7 |
| in-flight p95  | 30 |
| in-flight peak | 30 |
| probe samples  | 45 |


**SGLang state (from Prometheus, live gauges):**

| Signal | mean | p95 | peak |
|---|---:|---:|---:|
| sglang running (in-GPU batch) | 8.0 | 27 | 27 |
| sglang queued (waiting for slot) | 0.2 | 3 | 3 |


_Window: 1776879369–1776879438 (duration 69 s; step=1 s)_

#### c=32 (back-to-back) — Mean request phase decomposition

**CPU container**

| Phase | scope | mean ms | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| Flask end-to-end | per HTTP request | 9,498 | 5,978 | 30,750 | 54,150 |
| Layout forward (ONNX + pre/post) | per HTTP request | 9,360 | 5,585 | 20,000 | 20,000 |
| OCR region call | per region (N per HTTP request) | 6,282 | 6,466 | 23,900 | 28,780 |


**SGLang container**

| Phase | scope | mean ms | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| SGLang end-to-end | per SGLang request | 4,147 | 3,727 | 15,250 | 19,050 |
| Queue wait (scheduler) | per SGLang request | 182 | 71 | 929 | 2,200 |
| Time-to-first-token (prefill+first decode) | per SGLang request | 4,102 | 3,583 | 14,722 | 18,944 |
| Inter-token latency (decode step) | per decoded token | 26 | 10 | 98 | 176 |


_Values aggregate across all requests that completed inside the trial window. Scope indicates whether the histogram counts whole HTTP requests, per-region OCR fan-out calls, or per-token decode steps — percentiles are computed within that scope._

### c=40 (back-to-back)

**Worker concurrency (CPU container):**

| Metric | Value |
|---|---:|
| target concurrency (c) | 40 |
| in-flight mean | 29.4 |
| in-flight p95  | 37 |
| in-flight peak | 37 |
| probe samples  | 50 |


**SGLang state (from Prometheus, live gauges):**

| Signal | mean | p95 | peak |
|---|---:|---:|---:|
| sglang running (in-GPU batch) | 6.4 | 40 | 40 |
| sglang queued (waiting for slot) | 1.1 | 14 | 14 |


_Window: 1776879464–1776879533 (duration 69 s; step=1 s)_

#### c=40 (back-to-back) — Mean request phase decomposition

**CPU container**

| Phase | scope | mean ms | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| Flask end-to-end | per HTTP request | 11,423 | 9,746 | 41,300 | 56,260 |
| Layout forward (ONNX + pre/post) | per HTTP request | 11,280 | 8,981 | 20,000 | 20,000 |
| OCR region call | per region (N per HTTP request) | 4,187 | 3,195 | 20,125 | 28,025 |


**SGLang container**

| Phase | scope | mean ms | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| SGLang end-to-end | per SGLang request | 3,119 | 2,640 | 7,527 | 9,340 |
| Queue wait (scheduler) | per SGLang request | 384 | 65 | 2,615 | 4,646 |
| Time-to-first-token (prefill+first decode) | per SGLang request | 3,059 | 2,549 | 7,480 | 9,340 |
| Inter-token latency (decode step) | per decoded token | 9 | 7 | 22 | 38 |


_Values aggregate across all requests that completed inside the trial window. Scope indicates whether the histogram counts whole HTTP requests, per-region OCR fan-out calls, or per-token decode steps — percentiles are computed within that scope._

### c=64 (back-to-back)

**Worker concurrency (CPU container):**

| Metric | Value |
|---|---:|
| target concurrency (c) | 64 |
| in-flight mean | 29.5 |
| in-flight p95  | 49 |
| in-flight peak | 49 |
| probe samples  | 50 |


**SGLang state (from Prometheus, live gauges):**

| Signal | mean | p95 | peak |
|---|---:|---:|---:|
| sglang running (in-GPU batch) | 6.6 | 30 | 30 |
| sglang queued (waiting for slot) | 1.7 | 17 | 17 |


_Window: 1776865279–1776865358 (duration 79 s; step=1 s)_

#### c=64 (back-to-back) — Mean request phase decomposition

**CPU container**

| Phase | scope | mean ms | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| Flask end-to-end | per HTTP request | 14,756 | 14,207 | 47,283 | 57,457 |
| Layout forward (ONNX + pre/post) | per HTTP request | 14,475 | 11,923 | 20,000 | 20,000 |
| OCR region call | per region (N per HTTP request) | 5,770 | 5,121 | 20,437 | 28,087 |


**SGLang container**

| Phase | scope | mean ms | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| SGLang end-to-end | per SGLang request | 5,524 | 4,811 | 14,643 | 18,929 |
| Queue wait (scheduler) | per SGLang request | 680 | 75 | 2,984 | 3,923 |
| Time-to-first-token (prefill+first decode) | per SGLang request | 5,344 | 4,667 | 13,750 | 18,750 |
| Inter-token latency (decode step) | per decoded token | 67 | 44 | 305 | 381 |


_Values aggregate across all requests that completed inside the trial window. Scope indicates whether the histogram counts whole HTTP requests, per-region OCR fan-out calls, or per-token decode steps — percentiles are computed within that scope._


## Observability pointers

- Grafana dashboard: <http://localhost:3000/d/glmocr-load>
- Pushgateway: <http://localhost:9091/metrics> (job=`glmocr_asyncio`, run_id=`omnidoc-20260422-193250-asyncio-matrix`)
- Alloy UI: <http://localhost:12345/graph>
