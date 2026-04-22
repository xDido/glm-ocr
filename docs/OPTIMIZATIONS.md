# GLM-OCR CPU container — optimization cheat-sheet

Standalone reference for replicating the optimized setup in another repo. No cross-links to this project's tree; all values and knobs are spelled out so this file can be grep-pasted into the next codebase.

Target workload: document-OCR pipeline where a CPU container runs the layout detector (PP-DocLayoutV3 via transformers) and dispatches region crops to a GPU-side VLM server (SGLang serving GLM-OCR). Layout inference on CPU is the bottleneck.

Most numbers below are measured on an 8-vCPU / 24 GB cgroup matching a `g4dn.2xlarge`-class host. A follow-up retest at 12-vCPU (local AMD Ryzen 5 5600X max) is noted separately in the 12-core section. Load harness is an asyncio bench over 100/200 requests × 128-image pool at c = 12/24/32/40/64.

---

## TL;DR — the final `.env` block

```ini
# --- layout inference backend ---
LAYOUT_ENABLED=true
LAYOUT_DEVICE=cpu              # CUDA rejected on 8 GB cards (see rejected list)
LAYOUT_BACKEND=onnx            # ORT instead of torch eager — 1.76x on forward
LAYOUT_POSTPROC=numpy          # numpy post-proc — +6–17% rps, -14–31% p95
LAYOUT_GRAPH=raw               # fused graph rolled back — regressed at c=24/32
LAYOUT_USE_POLYGON=false       # polygon mode not needed for block-level OCR
LAYOUT_COMPILE=false           # torch.compile regresses +19% mean
LAYOUT_ONNX_THREADS=2          # saturates 8-core cgroup: 4 workers x 2 ORT = 8
                               # grow with cores: at 12c use 3 (measured +34% rps at c=12, +31% at c=24 vs 8c baseline)
LAYOUT_ONNX_PROVIDER=openvino  # OpenVINOExecutionProvider via onnxruntime-openvino wheel
                               # 3x+ rps on CNN-dominated layout vs default MLAS CPU EP
                               # device_type=CPU (no Intel iGPU on host); fall through to CPUExecutionProvider
LAYOUT_BATCH_ENABLED=true      # cross-request layout coalescer
LAYOUT_BATCH_MAX=8             # larger than 8 causes c=64 ServerDisconnectedError
LAYOUT_BATCH_WINDOW_MS=20
LAYOUT_ASYNC=false             # FastAPI sidecar didn't reliably improve

# --- CPU container sizing ---
CPU_WORKERS=4                  # gunicorn processes
CPU_THREADS=16                 # gthread per worker — 32 caused c=64 oversubscription
OMP_NUM_THREADS=1              # prevents MKL/OMP oversubscribing cgroup
MKL_NUM_THREADS=1

# --- HTTP fan-out + pool ---
OCR_MAX_WORKERS=32             # intra-request parallel region calls
OCR_CONN_POOL=2048             # must be >= CPU_THREADS * OCR_MAX_WORKERS
OCR_CONNECT_TIMEOUT=10
OCR_REQUEST_TIMEOUT=60
OCR_RETRY_MAX=2                # 1 -> 2 eliminated c=64 ServerDisconnectedError at n=200
OCR_RETRY_BACKOFF_BASE=0.5
OCR_RETRY_BACKOFF_MAX=8
GUNICORN_TIMEOUT=480           # must exceed OCR_REQUEST_TIMEOUT
```

**Process model:** gunicorn `--worker-class gthread --workers $CPU_WORKERS --threads $CPU_THREADS`. Async flask isn't the right model — the blocking C extensions under layout make async cooperatively starved. Gthread gives you OS-level preemption per request.

**Math-thread caps:** without `OMP_NUM_THREADS=1` + `MKL_NUM_THREADS=1`, each of the 4 gunicorn workers spawns a per-kernel OpenMP pool sized to the host core count. On an 8-vCPU cgroup that's 4×8 = 32 math threads fighting 8 cores under `CPU_THREADS=16` = 64 Python threads — immediate catastrophic oversubscription. Pin both to 1 and let `LAYOUT_ONNX_THREADS` own the intra-op parallelism for the only kernel that benefits from it.

---

## Shipped optimizations (kept — measured wins)

### 1. ONNX Runtime backend for layout forward

Replace torch-eager inference of PP-DocLayoutV3 with an ONNX Runtime session. Export once on first container boot (`torch.onnx.export`), then load the `.onnx` and run via ORT for every subsequent request.

Why: torch eager on CPU carries significant dispatch + allocator overhead per op. ORT's CPU kernel fusion gives ~1.76× on the forward pass of this detector with zero algorithmic change.

Cost: one-time export (~60s first boot; idempotent), plus ORT session load at worker startup. Disk footprint ~600 MB (fp32 graph).

Parity: export uses `torch.onnx.export` from the same weights → output tensors are numerically identical to torch eager within 1e-6.

Knob: `LAYOUT_BACKEND=onnx` (`torch` is the fallback).

### 2. numpy post-processing (drop torch from request path)

Replace upstream `PPDocLayoutV3ImageProcessor.post_process_object_detection` (which takes torch tensors) with a pure-numpy reimplementation. Call it directly on the ORT outputs; skip the `torch.from_numpy` wrap entirely.

Why: the ORT forward returns numpy, but upstream post-proc wants torch. The round trip (`np → torch wrap → torch post-proc → np again`) adds per-op Python overhead that holds the GIL in long chunks under concurrent load. Removing torch from the request path lets the GIL release more frequently and unblocks other gthreads.

Measured impact vs. torch post-proc, head-to-head on the asyncio matrix (100 requests, pinned pool):

| c | rps | mean | p50 | p95 | p99 |
|---:|---:|---:|---:|---:|---:|
| 12 | **+6%** | **-18%** | **-49%** | **-21%** | +17% |
| 24 | **+17%** | **-15%** | – | **-15%** | **-32%** |
| 32 | **+9%** | **-12%** | – | **-31%** | **-28%** |

Parity: bit-for-bit on smoke-test page; 30/30 OmniDocBench pages pass with max score Δ 1.19e-07 and max box Δ 0.00 px.

Scope: ~560 LOC numpy port (the deterministic-arithmetic math + a vendored copy of a few glmocr postproc helpers). Standalone module, no glmocr/transformers imports on the hot path.

Knob: `LAYOUT_POSTPROC=numpy` (`torch` is the fallback). Keep the torch branch available behind the flag for ~1-week soak before deleting.

### 3. Cross-request layout batcher (coalescer)

A thin batch-window coalescer in front of the ORT call. Any request arriving within `LAYOUT_BATCH_WINDOW_MS=20` ms of another gets stacked into one forward pass up to `LAYOUT_BATCH_MAX=8` images.

Why: under load, many concurrent requests each call the detector independently at small batch size. Batching amortizes per-forward fixed cost (session enter, memory allocator warmup, Python dispatch).

Measured: +29% rps at c=12; `c=64` failure count 14 → 0.

Tradeoff: `BATCH_MAX > 8` causes `ServerDisconnectedError` at c=64 (request holds the thread longer than keepalive tolerance under the downstream API). Don't push past 8 without also extending keepalive on the client.

Knobs: `LAYOUT_BATCH_ENABLED=true`, `LAYOUT_BATCH_MAX=8`, `LAYOUT_BATCH_WINDOW_MS=20`.

### 4. `LAYOUT_ONNX_THREADS=2` (saturate cgroup with ORT intra-op)

Bump ONNX Runtime intra-op threads from 1 to 2 per worker. With `CPU_WORKERS=4`, that's 4 × 2 = 8 ORT threads — exactly matches the 8-core cgroup and leaves one ORT kernel call per worker running on 2 cores in parallel.

Why: `LAYOUT_ONNX_THREADS=1` leaves cores idle because there's only one layout-forward per worker and each forward was single-threaded. Doubling releases headroom the cgroup already had.

Measured: +16–63% rps across all concurrency, p95/p99 -16–44%, peak rps 2.21 → 3.16. **No tradeoffs.**

Do-the-arithmetic rule: `CPU_WORKERS × LAYOUT_ONNX_THREADS` should equal the cgroup core count. Grow `LAYOUT_ONNX_THREADS` when you add cores.

**Validated at 12 cores**: on a 12-core cgroup, `LAYOUT_ONNX_THREADS=3` (4 × 3 = 12) gave +34% rps at c=12, +31% at c=24, +7% at c=32 vs the 8-core ORT=2 baseline — measured head-to-head on the same pool-seeded matrix. The c=12 p99 dropped from ~22,000 ms to ~9,000 ms (-59%) — the largest single win of the whole tuning arc.

**c=64 capacity-cliff fix at 12c**: with `OCR_RETRY_MAX=1` (default), c=64 at n=200 showed 22–40 `ServerDisconnectedError` failures with max latency hitting `OCR_REQUEST_TIMEOUT=60` s. The errors are server-side disconnects from SGLang overload, not client-side timeouts — so bumping `OCR_REQUEST_TIMEOUT` does *not* help (tested 60 → 120, failures got worse as slow requests piled up longer under overload). **Bumping `OCR_RETRY_MAX=1 → 2`** eliminated them: n=200 c=64 dropped from 40 → 0 failures with max latency 53,541 ms (comfortably under 60s). Rps is essentially unchanged (2.575 → 2.544 averaged). At 8 cores this cliff lives past c=64 because there's less throughput to overrun SGLang in the first place.

### 5. Supporting knobs (load-bearing, easy to miss)

- `OMP_NUM_THREADS=1` + `MKL_NUM_THREADS=1` — see "Math-thread caps" above. **Required**; without this, oversubscription masks every other fix.
- `OCR_CONN_POOL=2048` — must be **≥** `CPU_THREADS × OCR_MAX_WORKERS` or you get pool-exhaust 503s that look like downstream (SGLang) failures. Shoot for 2× buffer.
- `GUNICORN_TIMEOUT=480` — must exceed `OCR_REQUEST_TIMEOUT=60`. Gunicorn kills stuck workers before the app's own timeout fires otherwise, and you lose the tail-latency samples.
- Gunicorn `gthread` worker class, not `sync` or `gevent`. Sync is single-request-per-worker (no fan-out); gevent/eventlet monkey-patch breaks torch's threaded CPU kernels.
- Torch CPU-only wheel installed **before** `pip install glmocr` from PyTorch's dedicated index (`https://download.pytorch.org/whl/cpu`). Skips ~5 GB of unused CUDA dependencies pulled by the default PyPI wheel.
- Docker multi-worker Prometheus needs a shared tmpfs dir (`PROMETHEUS_MULTIPROC_DIR`) wiped on each container start; otherwise stale dead-worker values pollute the aggregation across reboots.

### 6. OpenVINO Execution Provider for layout (the 3× win)

Swap the `onnxruntime` wheel for `onnxruntime-openvino` in the CPU image and initialize the layout `InferenceSession` with `OpenVINOExecutionProvider` (device `CPU`) before the default `CPUExecutionProvider` fallback.

Why: per-op profile (ORT `enable_profiling`, chrome-trace) shows **Conv is 76% of every layout forward** (2,990 Conv invocations per call, 859 ms cumulative out of 1,134 ms mean wall time at intra-op=3, 800×800 input). The default ORT CPU EP uses MLAS; OpenVINO ships Intel's oneAPI Conv primitives, which beat MLAS on both Intel and AMD x86 — solo warm calls measured **755 ms/call vs 1,100 ms/call (−31%)**, and under 4-worker concurrency + the coalescer, the win amplifies to ~3× end-to-end throughput.

Measured vs `LAYOUT_ONNX_PROVIDER=cpu` (everything else identical), averaged across 3 × N=200 matrix runs:

| c | base rps | OV avg rps | Δ | base Flask p99 | OV p99 | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 3.58 | 9.12 | **+155%** | 6,376 | 8,152 | +28% |
| 24 | 3.13 | 13.78 | **+340%** | 24,151 | 6,489 | **−73%** |
| 32 | 2.79 | 14.81 | **+431%** | 31,223 | 10,448 | **−67%** |
| 40 | 3.06 | 13.20 | **+331%** | 29,583 | 11,954 | **−60%** |
| 64 | 2.79 | 9.40 | **+237%** | 53,537 | 20,004 | **−63%** |

Layout-forward p50 at c=12 dropped **3,744 → 691 ms (−81%)**. The gate is passed by enormous margins — this is the largest single win in the project's history, and it validates E4's "attack the Conv kernel, not the graph" conclusion.

Cost: image size grows ~200 MB (OpenVINO runtime libs baked into the wheel). ORT version inside the image becomes `1.24.1` (from `1.24.4`) because the openvino wheel is pinned one patch behind mainline ORT — no compat issues seen. First-request per-worker has a ~1 s OpenVINO graph-compile warmup; pre-warm with a burst of 16 concurrent hits before matrix measurement.

Tradeoff: the ~13 rps ceiling at c=24–40 surfaces a **new** capacity cliff — 20/3000 requests across 3 runs raised `ServerDisconnectedError` at non-deterministic c levels (c=24 on one run, c=64 on another). This is a downstream SGLang keepalive/batching event that the upstream speedup *exposes* but doesn't cause. Layout CPU is no longer binding; the next bottleneck is the GPU pipeline, which is finally worth tuning (speculative decoding, `max_tokens` reservation, etc.).

Device_type knob: set to `CPU` explicitly to avoid accidentally dispatching to an Intel iGPU the host doesn't have. On an Intel iGPU/NPU host, flipping `device_type` to `GPU`/`NPU`/`AUTO` would be a one-line config change, no rebuild. NVIDIA GPUs are invisible to OpenVINO EP (that's an OpenVINO limitation — the plugin only speaks Intel OneAPI).

Knob: `LAYOUT_ONNX_PROVIDER=openvino` (`cpu` is the rollback). The onnxruntime-openvino wheel ships **both** providers, so rollback is an env flip + `docker compose up -d --force-recreate cpu`; no rebuild.

---

## Rejected — documented so you don't re-try them without a new hypothesis

### `LAYOUT_COMPILE=true` (torch.compile on the detector)

+19% mean per-call, +18% p95, 2% failure rate from recompile-triggered timeouts. The compile recompiles on new input shapes, and OCR regions are wildly variable. Leave off.

### Dynamic int8 ONNX quantization (`quantize_dynamic`, `QuantType.QInt8`)

Single-forward **3× slower** (2.4s → 6.8s) and `pred_boxes` outputs broken (max |Δ| = 1.0 on normalized coords, which scale-wise is full-page-width). Likely cause: the detector's bounding-box head is extremely sensitive to quantization granularity and the dynamic approach quantizes everything. Needs per-layer static calibration with exclusions, which we didn't pursue.

### Phase 2 fused ONNX graph (bake post-proc math into ONNX)

Export with sigmoid + top-K + box decode + rescale + order decoder all as ONNX ops so the graph returns `(scores, labels, boxes, order, masks)` already ranked. Skips the ~5 ms of numpy arithmetic per page at the cost of a structurally larger graph.

Measured:
- c=12: **+29% rps, -34% p99** — real win.
- c=24: **-23% rps, +24% p95** — regression.
- c=32: **-33% rps, +45% p95** — severe regression.

Rolled back.

**Retried 2026-04-22** with the documented recipe: pinned opset 17 request, `onnxsim.simplify` post-export, bool-mask output (threshold baked in as a graph input, masks emitted as `tensor(bool)` — 4× smaller I/O). Parity cleared (30/30 OmniDocBench pages, worst IoU 1.0000, score Δ 1.19e-07). Matrix at N=200 × 2 still failed: c=12 rps −23% averaged, c=24 rps −15%, c=32 tripped a 14-failure capacity cliff on run 2. Root cause: `torch.onnx`'s exporter cannot emit a Pad op at opset 17 (no opset-17 adapter in the onnx C API's version converter; fallback to opset 18), which reintroduces exactly the concurrent-session overhead the first attempt hit. **Do not retry this again until a torch.onnx update adds opset-17 Pad support, or someone writes a standalone opset-18→17 pass for Pad.** Code kept dormant behind `LAYOUT_GRAPH=fused` in commit `b0cb780` (subsequently reverted in `cd4ae66`).

### Async FastAPI sidecar (`asyncio.to_thread(pipeline.process)`)

Over 2 averaged runs: only c=24 rps +11% (inside the ±15–25% single-run noise envelope); c=32 regressed -16% with 12 avg failures; c=12 tails widened. The async boundary doesn't actually unblock anything on this workload because `pipeline.process` is GIL-bound under the hood — `asyncio.to_thread` just trampolines into the same pool gthread would have used.

Left in the repo behind `LAYOUT_ASYNC=true` in case the workload shape changes enough for asyncio to pay off.

### `CPU_THREADS=32` (was 16)

+7–27% at c=24/32 but **-36% at c=64** (oversubscription past admission ceiling). Adding gthreads doesn't help once you're past the point the downstream API can absorb concurrent requests — you just move the pile-up from the inbound queue to the outbound.

### `LAYOUT_BATCH_MAX=12` (was 8)

Avg rps +15–27% at c=32/40 across 2 runs but **16 failures at c=64** (`ServerDisconnectedError`). Larger batches hold a thread for the batch window + forward time, exceeding keepalive at the outbound SGLang client under c=64 queue depth. 8 is the ceiling at this config.

### Layout on CUDA (`LAYOUT_DEVICE=cuda`)

Evaluated on an 8 GB dev GPU. Torch's CUDA context (~1 GB per worker × 4 workers) steals VRAM from SGLang's KV cache, which costs more end-to-end than the layout speedup. Revisit only on 16 GB+ cards.

### jemalloc + balanced-decay allocator

-15% throughput on this workload (aiohttp + torch small-alloc density fights jemalloc's arena-per-thread model). Sticking with glibc. Bound memory growth via gunicorn `--max-requests` recycling instead.

### Paced c=1 baselines in the matrix

Removed because they produce the same p50/p95/p99 within noise as the low-percentile tail of loaded trials. Saves ~10 min per matrix run with no loss of signal.

---

## Queued — impactful but not yet shipped

### SGLang-side tuning is finally worthwhile

With layout unblocked by OpenVINO EP (#6), the RPS ceiling now sits at SGLang. Two experiments previously dismissed because layout was binding are now live candidates:
- **Speculative-decoding sweep.** Grid `SGL_SPEC_NUM_STEPS ∈ {3,4,5} × SGL_SPEC_NUM_DRAFT_TOKENS ∈ {4,5,6}`. Greedy OCR output (top_k=1, top_p=1e-5) is a textbook fit for NEXTN acceptance; never measured at the current ingress rate.
- **`OCR_MAX_TOKENS` reduction.** glmocr defaults `max_tokens=8192` per region; observed mean is ~55 generated tokens. At the old RPS ceiling this didn't move the needle (queue collapsed but layout bound the ceiling); at 13 rps it should free meaningful SGLang slot headroom.
- **c=24/c=64 `ServerDisconnectedError` investigation.** 20/3000 in the OpenVINO matrix. Tunables: `OCR_REQUEST_TIMEOUT`, `OCR_RETRY_MAX`, SGLang `--max-running-requests`, keepalive on the aiohttp client.

### Preprocess into `onnxruntime-extensions` (half a day, only if profiled as bottleneck)

Move `PPDocLayoutV3ImageProcessor.resize + normalize` into the ORT session as a preprocessor op. Currently preprocess is sub-10 ms per page and not visible in the p50/p95/p99 tail decomposition — don't attempt without profile evidence.

### Scale `LAYOUT_ONNX_THREADS` with cgroup

If you grow the cgroup past 8 cores, re-tune `LAYOUT_ONNX_THREADS` to maintain `CPU_WORKERS × LAYOUT_ONNX_THREADS ≈ cgroup_cores`. The knob is the one with the cleanest scaling behavior in the whole setup.

### Post-soak cleanup (~1 hour)

Delete the `LAYOUT_POSTPROC=torch` fallback branch and the `LAYOUT_COMPILE` block from the runtime app (~80 LOC). Drops torch as a runtime dependency entirely. No perf delta, just dependency hygiene.

---

## Measurement methodology (the meta-lessons)

These burned days of measurement time; replicate them before tuning.

### 1. Average ≥2 matrix runs before any keep/rollback

Single-run variance on the asyncio matrix is **±15–25% on rps at low concurrency**. Concrete cases we hit:

- Fix 3 at c=12: 2.64 → 2.18 rps on a rerun (**-17%**) with identical config.
- Fix 4b at c=12: 2.26 → 3.10 rps on a rerun (**+37%**) with identical config.
- Fix 4b at c=64: 0 → 32 failures on a rerun with identical config.

We nearly rolled back a *winning* fix and nearly kept a *losing* fix from single-run numbers. Always run 2+ and average rps/p50/p95/p99 before committing.

### 2. `rps` is tail-dominated — always read p50 **and** p99 alongside

Matrix `rps` = `total_requests / wall_time` where wall_time is first-to-last completion. A single 50 s outlier at c=12 drops rps by 15%+ even when the p50 is unchanged. Never interpret rps in isolation — if rps moves and p99 doesn't, it's a single outlier artifact.

### 3. Inter-trial gaps ≥20 s are load-bearing for segmented reports

If you post-process Prometheus timelines per trial (in-flight gauges, queue depth), the segmenter needs ≥20 s of idle between trials to draw the boundary. Without the gap all trials merge into one segment and per-trial phase decomposition is silently dropped from the report. Don't skip the sleeps "to save time" — they're the report's only way to know where one trial ends and the next starts.

### 4. Apply fixes one at a time, matrix between each

The only way to attribute a delta to the right knob. Bundling two fixes means you can't tell which one helped if the result is mixed, and you're forced into an expensive A/B/A/B decomposition later.

---

## Architectural dependency order (why this order matters)

1. **ONNX backend first.** Unblocks `LAYOUT_ONNX_THREADS` and all alternative execution providers. Without the ORT move, none of the thread tuning or provider-swap applies.
2. **numpy post-proc second.** Prerequisite for deleting torch from the request path, and for accurate GIL accounting under load (you can't see the GIL ceiling while torch still holds it in long chunks).
3. **Batch coalescer third.** Independent of ORT/numpy ordering, but compounds the ORT gain.
4. **Thread tuning fourth.** `LAYOUT_ONNX_THREADS`, `CPU_THREADS`, `LAYOUT_BATCH_MAX` are all cgroup-sensitive and depend on which kernels are live. Don't tune them before the pipeline stack is finalized; you'll tune to the wrong baseline.
5. **Alternative EP last (once you have a profile).** Per-op profile before switching providers — the Conv share was 76% on this workload, making OpenVINO the obvious pick. Don't swap providers speculatively; the wheel size, import cost, and graph-compile warmup are real, and the win only materialises if the bottleneck op-type matches the EP's strength (OpenVINO for CNN Conv, TensorRT for GPU if available, DnnlEP for Dnnl-friendly shapes, etc.).

---

## Net impact (pre-optimization → final)

Two checkpoints are useful, because the arc has a clear before/after at the OpenVINO switch.

**Original baseline** (torch eager, torch post-proc, no batcher, `LAYOUT_ONNX_THREADS=1`, `LAYOUT_BATCH_MAX=4`):

| c | original rps | post-ORT+numpy+batcher rps | Δ |
|---:|---:|---:|---:|
| 12 | 1.55 | ~3.6 | +132% |
| 24 | 2.38 | ~3.1 | +30% |
| 32 | 2.17 | ~2.8 | +29% |

**After OpenVINO EP** (onnxruntime-openvino wheel, device=CPU; the final TL;DR `.env` block above), averaged 3 × N=200:

| c | post-ORT baseline rps | final (OV) rps | Δ vs post-ORT | Δ vs original |
|---:|---:|---:|---:|---:|
| 12 | 3.58 | 9.12 | **+155%** | **+487%** |
| 24 | 3.13 | 13.78 | **+340%** | **+479%** |
| 32 | 2.79 | 14.81 | **+431%** | **+582%** |
| 40 | 3.06 | 13.20 | **+331%** | — |
| 64 | 2.79 | 9.40 | **+237%** | — |

Flask p99 at c=32 went **31,223 ms (post-ORT baseline) → 10,448 ms (−67%)**. Layout-forward p50 at c=12: **3,744 → 691 ms (−81%)**.

The largest single win of the whole arc is OpenVINO EP (3–4× on its own). The ORT+numpy+batcher stack was the *prerequisite*: without a stable profile-ready pipeline, the provider swap would have been an uninformed guess. Apply them in the dependency order above; skipping steps 1–4 to jump straight to #6 works from a pure-throughput angle, but you lose the observability (histograms, GIL accounting) that tells you the provider swap is doing what you think it's doing.
