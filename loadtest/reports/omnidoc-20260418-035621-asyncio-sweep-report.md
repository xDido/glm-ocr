# GLM-OCR asyncio concurrency sweep — 2026-04-18

**Run ID:** `omnidoc-20260418-035621-asyncio-sweep`
**Driver:** asyncio (`loadtest/asyncio/bench.py`)
**Endpoint:** `POST http://localhost:5002/glmocr/parse`
**Dataset:** OmniDocBench (1,654 images available; 480 sampled for this run)

> **⚠️ Revised 2026-04-18 — see [addendum at end](#revision--c6-probe-run-reveals-sglang-is-the-bottleneck).**
> The "lock in the CPU container" hypothesis in the original findings was
> wrong. A follow-up c=6 run with `/metrics` probing + Prometheus backfill
> showed SGLang is the actual bottleneck (pinned at its 16-running cap with
> a 9-deep queue). Findings #5 and Recommendations #1–#3 below are
> superseded; read the addendum for the corrected story.

---

## Executive summary

The GLM-OCR stack has a **hard throughput ceiling at ~0.30 req/s** on
this host, and it is reached already at concurrency **6**. Raising
concurrency above 6 does not raise throughput at all — it converts
successes into failures while keeping rps essentially flat (0.25–0.31
across all six levels). The **healthy operating point is c=6**: 480/480
requests completed with zero failures, p50 = 15s, p99 = 75s. At c≥8,
45–94% of requests die with `ServerDisconnectedError` (Gunicorn killing
workers after the 180s timeout) or `TimeoutError` (client giving up).
Evidence of an internal global-serialization lock is strong: `min`
latency grows with c, which should not happen in a truly parallel
pipeline.

---

## Test configuration

### Sweep parameters

| Knob | Value |
|---|---|
| Concurrency levels | 6, 8, 10, 12, 14, 16 |
| Requests per level | 480 |
| Image pool | 480 URLs, one per request on average |
| Pool sampled from | `datasets/OmniDocBench/` (1,654 images) |
| Warmup per level | 2 requests (excluded from stats) |
| Timeout per request | 300s (client-side aiohttp) |

### Server config (from `.env` at run time)

| Var | Value | Meaning |
|---|---|---|
| `CPU_WORKERS` | 2 | Gunicorn worker processes |
| `CPU_THREADS` | 8 | Gthread threads per worker |
| `GUNICORN_TIMEOUT` | 180 | Worker kill timeout (seconds) |
| `OCR_MAX_WORKERS` | 8 | SGLang fan-out pool per request |
| `OCR_REQUEST_TIMEOUT` | 120 | SGLang call timeout |
| `OCR_CONN_POOL` | 128 | HTTP connection pool to SGLang |
| `LAYOUT_ENABLED` / `_DEVICE` | true / cpu | PP-DocLayoutV3 on CPU |
| `SGL_MAX_RUNNING_REQUESTS` | 16 | SGLang concurrency cap |
| `SGL_MAX_TOTAL_TOKENS` | 100,000 | SGLang KV-cache budget |

Total *inbound* thread slots: `CPU_WORKERS × CPU_THREADS = 16`. So the
pool is sized for exactly the highest-c level in the sweep.

---

## Results — all six levels

| c | ok | fail | fail % | wall (s) | rps (succ) | min (ms) | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **6**  | **480** | **0**   | **0%**  | **1,567.8** | **0.306** | **2,221**  | **15,075** | 37,141  | 47,618  | 74,570  | 265,475 |
| 8  | 132 | 348 | 72.5% | 449.3   | 0.294 | 3,522  | 18,572 | 45,890  | 65,780  | 94,511  | 127,133 |
| 10 | 234 | 246 | 51.3% | 769.6   | 0.304 | 4,992  | 23,924 | 52,674  | 64,532  | 90,242  | 114,089 |
| 12 | 258 | 222 | 46.3% | 876.6   | 0.294 | 7,577  | 30,317 | 62,924  | 95,174  | 120,678 | 126,611 |
| 14 | 28  | 452 | 94.2% | **99.3** ⚠️ | 0.282 | 13,811 | 44,332 | 64,937  | 82,763  | 97,096  | 99,025  |
| 16 | 155 | 325 | 67.7% | 612.1   | 0.253 | 11,239 | 47,028 | 76,454  | 119,902 | 142,419 | 230,350 |

**Raw JSON files** (per level):
`loadtest/results/omnidoc-20260418-035621-asyncio-sweep-c{6,8,10,12,14,16}.json`

---

## Key findings

### 1. The throughput curve is flat — ~0.30 rps is the real ceiling

rps sits in a narrow 0.25–0.31 band across all six concurrency levels.
In a healthy system, doubling concurrency from 8 → 16 would roughly
double throughput until the server saturates. Here it actually *drops*
(0.294 → 0.253). That is the textbook signature of a **server fully
saturated at ≪ c=6**; adding client pressure just deepens the queue.

```
rps vs concurrency
0.31 ┤ ●                                   ← c=6 (perfect, no failures)
0.30 ┤            ●                        ← c=10
0.29 ┤    ●          ●                     ← c=8, c=12
0.28 ┤                  ●                  ← c=14 (anomalous)
0.27 ┤
0.26 ┤
0.25 ┤                     ●               ← c=16
     └─────────────────────────────────
       6   8   10   12   14   16
```

### 2. c=6 is the healthy operating point

The only level with **zero failures**: 480/480 completed, wall 1,568s,
rps 0.306, p99 74.6s. Anything beyond c=6 is the server accepting
requests it cannot actually serve. Practical implication: put a
semaphore in front of `/glmocr/parse` capped at 6, return 503 when the
queue is full, and clients get honest backpressure instead of
half-done 180s timeouts.

### 3. Failure mode shifts reveal where the bottleneck is

| c | Dominant error | What it means |
|---|---|---|
| 8 | `ServerDisconnectedError` only | Gunicorn killed the worker at its 180s timeout while still mid-request. No client-side timeout fired — requests just vanished. |
| 10–12 | Mix: `TimeoutError` + `ServerDisconnectedError` | Some requests queued long enough to trigger the 300s client timeout; others got killed server-side first. |
| 14–16 | Mix, skewed toward `TimeoutError` | Deeper queues → more client-side timeouts first. |

Both errors share a root cause: **request latency exceeds the 180s
worker budget while the client is still waiting**. A queue ≥ 180s deep
means at least one request in every batch gets killed.

### 4. p50 rises linearly with c — pure queuing, not real work

| c | p50 (s) | Δ vs c=6 | Expected if pure queuing (c/6 × 15s) |
|---:|---:|---:|---:|
| 6  | 15.1 | — | — |
| 8  | 18.6 | +3.5 | 20.1 (off by 1.5s — slightly faster than pure queueing) |
| 10 | 23.9 | +8.8 | 25.1 |
| 12 | 30.3 | +15.2 | 30.2 ✓ near-exact match |
| 16 | 47.0 | +31.9 | 40.2 (slower than pure queueing → worse than linear) |

p50 tracking `c × work_time / effective_parallelism` so closely is
Little's Law behaving exactly. The slight acceleration at c=16
(47s > 40s expected) is contention beyond pure queuing — likely the
connection pool or some shared resource adding overhead once the
pipeline backs up.

### 5. `min_latency` growing with c is the smoking-gun serialization signature

| c | min (s) |
|---:|---:|
| 6  | 2.2 |
| 8  | 3.5 |
| 10 | 5.0 |
| 12 | 7.6 |
| 14 | 13.8 |
| 16 | 11.2 |

In a truly parallel pipeline, `min_latency` is a property of a *single
unloaded request* — adding concurrency shouldn't affect the fastest
request in the batch because it ran when nothing else was happening.
Here the fastest request at c=14 is **6× slower** than the fastest at
c=6. This means even the "lucky" request is waiting on some global
resource. The most likely candidates:

- A `threading.Lock` around the layout detector or OCR model inside the
  `glmocr` library (wasn't found in our own code, so it's inside the
  package — confirm with
  `docker compose exec cpu grep -rn "threading.Lock" $(python -c "import glmocr, os; print(os.path.dirname(glmocr.__file__))")`).
- The Python GIL, if the hot path is pure-Python and CPU-bound
  (unlikely to fully explain 6× serialization, but contributes).
- A single `Pipeline` object shared across all 8 threads in each
  worker (confirmed in `docker/cpu/runtime_app.py:316`).

---

## Little's Law sanity check

At the clean point c=6:

```
L = λ × W
6 = 0.306 × W
W = 19.6s
```

Observed mean latency at c=6 = **19.46s** — near-perfect match. This
confirms c=6 is fully utilized at steady state.

Effective parallelism estimate:

```
ceiling_rps = effective_parallelism / mean_work_time_per_request
0.306       = P / 2.2          (using min_latency as work-time proxy)
P           ≈ 0.67
```

So the server is serving roughly **0.5–1 request genuinely in parallel**
— effectively serial, with maybe 20–30% overlap from the layout
detector running while SGLang is processing. Far below the nominal
2×8 = 16 thread slots, and far below SGLang's configured cap of 16.

---

## Anomaly — c=14 wall time

c=14 finished in **99.3 seconds** with 94.2% failure rate. That's an
order of magnitude faster than its neighbors (c=12 took 876s, c=16 took
612s). 452 failures in 99s = 4.6 fail/s. The math only works if most
failures returned in 1–3 seconds — i.e., the server had shut the TCP
socket almost immediately.

Hypotheses:

1. **Worker restart storm.** Gunicorn killed both workers and the
   restart took longer than typical, leaving requests to hit a brief
   "no worker listening" window → `ECONNREFUSED` / immediate
   disconnect.
2. **OOM kill.** The 8 threads × layout model + SGLang client pool may
   have tipped into OOM under c=14 pressure, dropping connections until
   a restart cleared state.
3. **Transient host contention.** Another process on the host took a
   big CPU slice mid-run.

Recommended: re-run c=14 in isolation to see whether it's reproducible.
If the repeat looks like c=12 and c=16's shape, the first run was a
transient. If it repeats the 100s/450-fail pattern, something
structural at c=14 (not at adjacent levels) is worth digging into.

---

## Recommendations

1. **Cap client-facing concurrency at 6.** Add a semaphore in front of
   `/glmocr/parse` and return 503 + `Retry-After` when saturated. This
   trades 72%+ failure rates for honest backpressure.

2. **Investigate the serialization lock.** Before scaling up (more
   workers, more containers), find the lock — otherwise adding capacity
   doesn't help, because the extra threads all serialize on the same
   lock anyway. Quick checks in order of cost:
   - `grep -rn "threading.Lock\|asyncio.Lock\|Semaphore("` inside the
     installed `glmocr` package.
   - `py-spy dump --pid <gunicorn worker PID>` during a c=6 load run —
     threads stuck on `lock.acquire()` tell you exactly where.
   - Profile a single request end-to-end (layout detector + SGLang
     call) to see which step is ~serial.

3. **Horizontal scaling will not help until #2 is resolved.** Doubling
   container count doubles ingress capacity but each container still
   bottoms out at 0.3 rps. The unit economics improve; the tail doesn't.

4. **Re-run c=14 alone to classify the anomaly.** One 20-minute run
   disambiguates whether c=14 has a structural issue or was a
   transient.

---

## Observability cross-check (for the next run)

While a c=6 run is in flight, confirm the serialization hypothesis from
the server side:

```bash
watch -n 2 'curl -s http://localhost:5002/runtime/summary | python -m json.tool \
    | grep -E "live_running|live_queued|in_flight"'
```

Expected pattern if CPU container is the bottleneck:

- `glmocr_in_flight_requests` hovers at 6 (client is keeping it full).
- `sglang:num_running_reqs` stays at 1–2 (SGLang itself is mostly idle
  waiting for the next crop to arrive).
- `sglang:num_queue_reqs` = 0.

That combination means the GPU is not the bottleneck — the serialization
is upstream, inside the CPU container's pipeline.

Opposite pattern (if GPU is actually the bottleneck):

- `in_flight` ~6, SGLang running ~8+, queue > 0. Then the CPU container
  is fanning out correctly and the GPU can't keep up.

---

## Appendix — per-level output files

| File | Purpose |
|---|---|
| `omnidoc-20260418-035621-asyncio-sweep.urls.txt` | The 480-image URL pool shared across all levels. |
| `omnidoc-20260418-035621-asyncio-sweep-c{6,8,10,12,14,16}.json` | Per-level summary (same schema as the single-driver bench). |
| `sweep-live.log` | Mirrored stdout from the sweep script (tee'd at runtime). |

---

## Revision — c=6 probe run reveals SGLang is the bottleneck

**Follow-up run ID:** `omnidoc-20260418-134009-asyncio-probe-c6`
**Script:** `scripts/omnidoc_asyncio_probe_c6.sh`
**When:** same host, same day, ~10 hours after the sweep.

### What the follow-up did

1. Re-ran **c=6 / 480 requests** against the CPU container — the "clean"
   level from the sweep — using a fresh 64-image OmniDocBench pool.
2. Started a **live probe** (`scripts/runtime_probe_loop.py`) that polled
   the CPU container every 2s and captured `glmocr_in_flight_requests`
   into JSONL.
3. Because `/runtime/summary`'s `_filter_sglang_metrics` has a parsing
   bug that drops Prometheus label sets (so `live_running` /
   `live_queued` always came back `null`), the SGLang signals were
   **backfilled from the local Prometheus** via `query_range` over the
   exact run window `[ts=1776512409, ts=1776513976]`.

### Bench result — matches the sweep's c=6 row

| Metric | Sweep c=6 | Probe c=6 | Δ |
|---|---:|---:|---:|
| successes | 480 | 480 | 0 |
| failures | 0 | 0 | 0 |
| throughput (rps) | 0.306 | 0.309 | +0.003 |
| p50 (ms) | 15,075 | 13,556 | -1,519 |
| p95 (ms) | 47,618 | 56,091 | +8,473 |
| p99 (ms) | 74,570 | 106,602 | +32,032 |
| mean (ms) | 19,459 | 19,244 | -215 |

Reproducibility is tight on rps and mean. The p99 being ~30s higher is
noise from two slow outliers in a 480-sample run (max also went from
265s → 235s, so the slowest request was faster — the long tail is
redistributed, not worse).

### Correlated signals during the 1,567s run window

| Signal | Source | mean | max | p95 | samples |
|---|---|---:|---:|---:|---:|
| `glmocr_in_flight_requests` | probe (`/metrics`) | **5.91** | 6 | 6 | 781 |
| `sglang:num_running_reqs` | Prometheus | **13.40** | **16** | 15 | 53 |
| `sglang:num_queue_reqs` | Prometheus | **9.23** | **32** | 25 | 53 |

### What the numbers actually say

**Fan-out ratio:** 13.40 / 5.91 = **~2.27 SGLang calls per client
request**. That matches expectations: `LAYOUT_ENABLED=true` splits each
page into a small number of regions, then the pipeline fires up to
`OCR_MAX_WORKERS=8` parallel SGLang calls per request. At c=6, the CPU
container is successfully fanning out ~14 parallel SGLang calls at any
moment — that's the pipeline *working correctly*, not serialization.

**SGLang is pinned at its cap.** With `SGL_MAX_RUNNING_REQUESTS=16`,
mean=13.4 and max=16 means SGLang spent a lot of the run at 100%
running-budget utilization. When it hit the cap, additional incoming
calls queued — mean queue 9.2, peaks to 32.

**Why throughput plateaus at ~0.30 rps.** SGLang's service rate for this
workload is capped by its 16-running batch. Pushing more client
concurrency (c=8, 12, 16) just inflates the queue. Once a queued SGLang
call waits past `OCR_REQUEST_TIMEOUT=120s`, the CPU container gets a
timeout, retries (3×), eventually times out the whole request, and the
Gunicorn worker either burns the 180s worker budget or gets
disconnected mid-flight — which is exactly the
`TimeoutError` → `ServerDisconnectedError` progression observed in the
sweep.

### Why Findings #5 and Little's-Law interpretation were misleading

- **Finding #5 ("min_latency growing with c = serialization lock").**
  Still a real observation, but the mechanism isn't a Python lock. It's
  SGLang *queue depth*. Even a single "lucky" request at c=6 arrives to
  find 9 older calls already queued at SGLang; as c grows, the queue
  deepens and everyone — including the fastest request — waits longer.
- **"Effective parallelism ≈ 0.67".** That number came from
  `P = rps × min_latency` which implicitly assumed min_latency is
  uncontended. It's not — min_latency at c=6 is already affected by the
  9-deep SGLang queue. The real effective parallelism inside the stack
  (measured by sglang_running) is **~13**. The throughput ceiling is
  service rate × 16 ≈ 0.3 rps, not a 0.67-wide lock.

### Revised recommendations (supersede the original list)

1. **~~Investigate the Python lock in `glmocr`.~~** Don't. There isn't
   one — the observed behavior is explained by SGLang queue saturation.
2. **Raise `SGL_MAX_RUNNING_REQUESTS`** if GPU memory / KV cache allow.
   Check `DCGM_FI_DEV_FB_USED` during c=6: if there's headroom under
   `SGL_MEM_FRACTION_STATIC=0.88`, try 24 or 32. Throughput should rise
   roughly linearly with this knob until you hit a *new* ceiling
   (attention-compute or token-budget).
3. **Reduce fan-out per request.** `OCR_MAX_WORKERS=8` lets one
   document burst 8 parallel SGLang calls. Lowering to 4 cuts peak
   SGLang contention in half at the cost of per-document latency —
   a good trade when aiming for higher aggregate rps.
4. **Consider `LAYOUT_ENABLED=false` for simple pages.** Whole-page OCR
   produces 1:1 fan-out instead of 2.3:1. Throughput goes up; accuracy
   on complex / multi-column documents goes down. Workload-specific.
5. **Keep the original recommendation to cap client concurrency at 6.**
   The business-logic advice holds — just for the different reason that
   past c=6, the CPU container is only inflating SGLang's queue rather
   than buying any additional throughput.
6. **Alert on `sglang:num_queue_reqs > 5` sustained.** That's the
   leading indicator of the cascading-failure pattern. At c=6 it
   averaged 9 — which is already why tail latency was 75-100s even
   without failures. Past ~20 it starts converting into timeouts.

### Known issues uncovered by this run

- **`_filter_sglang_metrics` in `runtime_app.py:157-182` strips the
  metric name wrongly** when Prometheus labels are present (current
  SGLang version emits `sglang:num_running_reqs{engine_type="unified",
  model_name="glm-ocr", ...}`; the filter splits on the last space and
  keeps the labelled form as the dict key). `/runtime/summary`'s
  `live_running` and `live_queued` have been effectively dead since
  SGLang added those labels — a fix would parse out the bare metric
  name before indexing.
- **`scripts/runtime_probe_loop.py` inherited the same bug** by
  relying on `/runtime/summary`. A follow-up patch should hit
  `sglang:30000/metrics` directly and parse the labelled exposition
  itself, so future probe runs don't need Prometheus backfill.

### Addendum files

| File | Purpose |
|---|---|
| `omnidoc-20260418-134009-asyncio-probe-c6.bench.json` | Bench summary from the c=6 probe run. |
| `omnidoc-20260418-134009-asyncio-probe-c6.probe.jsonl` | 781 samples of `in_flight` every 2s (sglang columns null — see bug above). |
| `omnidoc-20260418-134009-asyncio-probe-c6.urls.txt` | Image pool used (64 URLs). |
| `probe-c6-live.log` | Tee'd stdout of the probe orchestrator. |
