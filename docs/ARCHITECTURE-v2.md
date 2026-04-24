# GLM-OCR Architecture v2 — End-to-End Request Lifecycle

**Scope.** v1 of this document (`docs/ARCHITECTURE.md`) described the file layout and deployment shape. v2 goes deeper: a single HTTP request's complete journey from the client's TCP socket through every thread, every kernel, every cache layer, and back. Written after the 2026-04-24 shipments (Paddle2ONNX backend §6, OpenVINO EP §7, prefix-pin §8, `SGL_MEM_FRACTION_STATIC=0.83` §9 in `docs/OPTIMIZATIONS.md`). Cross-references into `OPTIMIZATIONS.md` are noted inline.

**Audience.** You already know Docker and Flask. You want to know what happens between `curl localhost:5002/glmocr/parse` and the JSON coming back.

---

## 0. Top-down overview

```
                       CLIENT
                          │  HTTP POST /glmocr/parse  {"images":[url,...]}
                          ▼
┌────────────────────── glmocr-cpu (Docker, cgroup: 12 vCPU / 24 GB) ──────────────────────┐
│                                                                                          │
│  Linux TCP stack  →  Docker userland proxy on :5002  →  gunicorn master (pid 1)          │
│                                                          │ dispatches to one of 4        │
│                                                          │ gthread workers via accept()  │
│                                                          ▼                               │
│  gunicorn worker (pid 12-15)  —  one Flask WSGI app per worker, shared nothing           │
│    │                                                                                     │
│    ├─ Flask routing  →  @app.route("/glmocr/parse")  in  glmocr/server.py:75             │
│    │                                                                                     │
│    ├─ pipeline.process() is a generator; spins up three daemon threads per request:      │
│    │     ┌─────────────┐  page queue   ┌──────────────┐  region queue  ┌──────────────┐  │
│    │     │ data-loading│──────────────▶│  layout      │───────────────▶│  recognition │  │
│    │     │  worker     │ (bounded)     │  worker      │ (bounded)      │  worker      │  │
│    │     └─────────────┘               └──────────────┘                └──────────────┘  │
│    │         PDF→PIL                    ONNX inference                   ThreadPool of   │
│    │         URL fetch                  (Paddle2ONNX +                   OCR_MAX_WORKERS │
│    │                                    ORT + OV EP)                     aiohttp sessions│
│    │                                                                                     │
│    └─ + _health_watchdog thread polls SGLang /health                                     │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                          │                                     │
                          │ 14 parallel OCR requests per page   │ layout: CPU only
                          │ via aiohttp connection pool         │
                          ▼                                     
┌─────────────────────── glmocr-sglang (Docker, GPU passthrough) ──────────────────────────┐
│                                                                                          │
│  HTTP 30000  →  uvicorn  →  OpenAI-compatible /v1/chat/completions                       │
│    │                                                                                     │
│    ├─ Chat template renders {"role":"user","content":[text, image]} to flat tokens       │
│    ├─ RadixCache lookup: does this token prefix exist in the KV cache already?           │
│    │     (prefix-pin §8 targets this specifically)                                       │
│    ├─ Scheduler (LPM policy) places request into running batch                           │
│    ├─ Prefill (chunked, 8192 tokens/chunk) computes K/V for new tokens                   │
│    ├─ Decode loop:                                                                       │
│    │     • bs ≤ 8   →   CUDA graph replay (fast)                                         │
│    │     • bs >  8   →   eager execution (slow fallback)                                 │
│    │     • speculative: EAGLE/NEXTN draft model generates 4 tokens, target verifies      │
│    ├─ Token stream out → tokenizer decode → text                                         │
│    ▼                                                                                     │
│  HTTP 200 {"choices":[{"message":{"content":"..."}}]}                                    │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                          ▲
                          │ response stream
                          │
┌────────────────────── back on glmocr-cpu ────────────────────────────────────────────────┐
│                                                                                          │
│  recognition worker merges per-region text  →  result_formatter builds markdown  →       │
│  pipeline.process() yields PipelineResult  →  Flask handler jsonify()  →  WSGI response  │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
                       CLIENT receives {"json_result":[...], "markdown_result":"..."}
```

End-to-end wall-clock at c=8 on the shipped stack: **~11 s per request** for a typical OmniDocBench page (14 detected regions). The breakdown below allocates every millisecond.

---

## 1. Network + TCP ingress

### 1.1 Client → Docker published port

Container `glmocr-cpu` publishes 5002 via `docker-compose.yml: ports: ["5002:5002"]`. Docker Desktop's VPNKit (on Windows/Mac) or iptables DNAT rules (on Linux) forward `host:5002 → container:5002`. This adds ~0.1 ms of NAT overhead per request on Linux, more on Windows (hence the bigger `docker run --rm -v` benchmark discrepancy we see vs bare-metal numbers).

### 1.2 gunicorn master socket

`docker/cpu/entrypoint.sh` launches:

```
exec gunicorn --worker-class gthread \
              --workers ${CPU_WORKERS:-4} --threads ${CPU_THREADS:-16} \
              --timeout ${GUNICORN_TIMEOUT:-480} \
              --bind 0.0.0.0:5002 wsgi:app
```

Gunicorn master (pid 1) calls `socket(), bind(), listen()` on 0.0.0.0:5002 and then forks `CPU_WORKERS=4` workers. Each worker inherits the listening socket. When a client connects, the kernel wakes exactly one worker (thundering-herd avoidance via `SO_REUSEPORT` is NOT used by default; gunicorn uses its own round-robin). The master itself never touches application traffic — it exists to supervise workers (respawn on crash, graceful restart on `SIGHUP`, rotate max-requests).

### 1.3 gthread worker acceptance

Worker class `gthread` is critical (see OPTIMIZATIONS.md supporting-knob §5):

- `sync` would dedicate one entire worker to one request at a time — with 4 workers, max in-flight = 4. At `c=8` half our requests block.
- `gevent`/`eventlet` monkey-patch stdlib threading, which breaks PyTorch and ONNX Runtime's C-level thread pools. Their TBB/OpenMP kernels assume OS threads, not greenlets.
- `gthread` uses a pool of `--threads 16` OS threads per worker. `accept()` returns in a thread, the thread runs the WSGI app synchronously for that request, then returns to the pool. Max in-flight per worker = 16; across the container = **64 concurrent requests**.

In our config at c=8 each of 4 workers gets ~2 concurrent requests on average, well under the 16-thread cap.

### 1.4 WSGI adapter

`wsgi.py` is a one-liner: `from glmocr.server import create_app; from glmocr.config import load_config; app = create_app(load_config("/app/config.yaml"))`. This runs **once per worker at import time**, well before the first request — it's how all the pipeline state gets initialized (threads, ORT sessions, aiohttp pools, model weights) before the worker is added to the accept pool.

---

## 2. Request entry: Flask routing and handler

### 2.1 Route dispatch

Flask's Werkzeug routing maps `POST /glmocr/parse` to `glmocr/server.py:75 @app.route("/glmocr/parse", methods=["POST"])`. The `prometheus-flask-exporter` middleware wraps the handler to record `flask_http_request_duration_seconds` histogram buckets keyed by `url_rule + method + status` (see OPTIMIZATIONS.md §5 for why the shared-tmpfs multiproc dir matters across workers).

### 2.2 `def parse()` — body validation

```python
# glmocr/server.py:92-117 (paraphrased)
if request.headers.get("Content-Type") != "application/json":
    return jsonify(error=...), 400
data = request.json                        # werkzeug lazy-parses
images = data.get("images", [])
if isinstance(images, str): images = [images]
if not images and "file" in data:          # MaaS-client back-compat
    images = [data["file"]]
```

A few μs total; synchronous; runs in the gthread that caught the accept.

### 2.3 Building the pipeline request

```python
# server.py:124-131 — the handler packs every input URL as an image content item:
messages = [{"role": "user", "content": []}]
for image_url in images:
    messages[0]["content"].append({
        "type": "image_url",
        "image_url": {"url": image_url}
    })
request_data = {"messages": messages}
```

Note the handler receives image **URLs**, not base64 blobs. A URL can be `file:///app/...` (bind-mounted dataset), `http://...`, `https://...`, or `data:image/png;base64,...`. The loader worker downstream will resolve whatever form the URL takes.

### 2.4 `pipeline.process(request_data)` — generator call

Here's where the magic starts. `pipeline.process` is a **generator** (`glmocr/pipeline/pipeline.py:108`) — calling it returns an iterator without running any work yet. The handler forces execution with `list(pipeline.process(...))`. Each yielded value is one `PipelineResult` = one input page's OCR.

---

## 3. The three-worker pipeline

`pipeline.process()` is the orchestration core. It launches three daemon threads per request and emits results via the generator protocol.

```python
# pipeline.py:140-172 (paraphrased)
state = PipelineState(page_maxsize=_page_maxsize, region_maxsize=_region_maxsize)
tracker = UnitTracker(num_units)
state.set_tracker(tracker)

t1 = Thread(target=data_loading_worker, args=(state, page_loader, image_sources), daemon=True)
t2 = Thread(target=layout_worker,       args=(state, layout_detector, ...),       daemon=True)
t3 = Thread(target=recognition_worker,  args=(state, page_loader, ocr_client, self.max_workers), daemon=True)
# + _health_watchdog thread
t1.start(); t2.start(); t3.start(); t_watchdog.start()

try:
    yield from self._emit_results(state, tracker, ...)
finally:
    state.request_shutdown()
    t1.join(timeout=10); t2.join(timeout=10); t3.join(timeout=10)
```

**Why three threads instead of inline synchronous code?** Because each stage has a different CPU/IO shape:

| stage | bottleneck | contention |
|---|---|---|
| data-loading | IO (URL fetch or PDF rasterize) | GIL-light |
| layout | CPU (ORT+OV kernels, releases GIL during C calls) | burstier |
| recognition | IO (HTTP to SGLang) | many parallel aiohttp |

Running them as a pipeline lets the stages overlap: while layout runs on page N, the loader already fetched page N+1 and recognition is already OCR'ing page N-1's regions. Bounded queues (`page_maxsize`, `region_maxsize`) provide back-pressure so a slow stage can't infinity-accumulate work behind it.

All three threads run **inside the same gthread** that accepted the request. They share memory. On a single `/glmocr/parse` call with one image, the data-loading work is trivial, so you mainly see t2 (layout) and t3 (recognition) serialized per page — but across concurrent requests multiple pipelines run simultaneously, and t2 from request A can overlap t3 from request B.

---

## 4. Data-loading worker (Stage 1)

### 4.1 URL → bytes

`data_loading_worker` (in `glmocr/pipeline/data_loading.py`, wrapped as a method on `PageLoader`) resolves each URL:

- **file://** — direct open
- **http/https** — `requests.get` with a timeout (inherited from `page_loader` config)
- **data:** — base64 decode in-memory
- **PDF byte sniff** — if first bytes are `%PDF-`, invoke `pdfium2` to rasterize pages

### 4.2 PDF rasterization

PDF pages are rasterized to PIL images at a configurable DPI (default 200). `pdfium2` is a C library with Python bindings — it releases the GIL during rendering, so concurrent data-loading workers across requests actually benefit from multi-core. For our current OmniDocBench workload all inputs are pre-rasterized `.png`/`.jpg`, so this path is dormant.

### 4.3 PIL open and enqueue

For images: `PIL.Image.open(bytes)` creates a lazy handle (no pixel decode yet). The worker pushes `(unit_id, page_index, PIL.Image)` onto the page queue and proceeds.

### 4.4 Page queue

A `queue.Queue(maxsize=page_maxsize)`. When the downstream layout worker is slow, `put()` blocks and the loader thread idles. Default `_page_maxsize` is typically 2×workers × n_pages_per_pdf; at our single-image workload this queue never backs up.

---

## 5. Layout worker (Stage 2) — the star

This is where our 2026-04-24 shipments live. The layout worker pulls a `(unit_id, page)` from the page queue, runs the detector, pushes one-or-more `(unit_id, region)` tuples onto the region queue.

Runtime dispatch is built at gunicorn worker startup via `install_pipeline_gauges(app)` in `docker/cpu/runtime_app.py:540-750`. At import time, `runtime_app.py` examines `LAYOUT_VARIANT` and `LAYOUT_BACKEND` env vars and **monkey-patches** `ld.process` (where `ld = pipeline.layout_detector`) to replace glmocr's default torch-eager path with one of three optimized implementations:

1. `LAYOUT_VARIANT=paddle2onnx` → Paddle2ONNX + ORT + optionally OV EP (§6/§7)
2. `LAYOUT_BACKEND=onnx + LAYOUT_POSTPROC=numpy` → torch-exported ONNX + numpy post-proc (older path)
3. Default → upstream torch eager (slowest; used as fallback if above two fail)

In the shipped config all four gunicorn workers take path (1). Let's trace it.

### 5.1 PIL → pixel tensor (`layout_paddle2onnx.py:_preprocess_batch`)

```python
# For each PIL image in the batch:
resized = im.resize((800, 800), Image.BILINEAR)      # shrink to model input size
arr = np.asarray(resized, dtype=np.float32) / 255.0  # RGB / 255 — see §6 gotcha
arr = arr.transpose(2, 0, 1)                         # HWC → CHW
return np.stack(batch, axis=0)                       # (B, 3, 800, 800)
```

The `/255` is critical — `config.json` declares `norm_type=none` but Paddle's C++ loader actually does `/255` before `NormalizeImage` sees the tensor. Without it, max detection score is 0.014 (nothing passes threshold). This was a real session bug before it became a comment.

### 5.2 3-input feed construction (`_build_session_inputs`)

Paddle2ONNX expects three inputs:

| input | shape | semantics |
|---|---|---|
| `image` | `(B, 3, 800, 800)` float32 | the preprocessed pixel tensor |
| `im_shape` | `(B, 2)` float32 | original `[H, W]` per image |
| `scale_factor` | `(B, 2)` float32 | nominally `(H_orig/800, W_orig/800)` |

The `scale_factor` input is effectively a no-op in this export — the graph ignores it. We pass the "correct" values anyway for semantic cleanliness, but the output boxes come back in **800² space** and the adapter rescales them to original image space manually (see §6 coord-space gotcha).

### 5.3 ORT session invocation

`sess.run(None, feed)` is a single C call into ONNX Runtime. Python's GIL is **released** for the duration — which is why multiple concurrent requests can actually run layout in parallel on different worker gthreads.

What happens inside:

```
┌─ onnxruntime.InferenceSession.run() ─────────────────────────────────────┐
│                                                                          │
│  1. OrtApi::Run dispatches to the InferenceSession C++                   │
│  2. Session checks providers in order: [OpenVINOExecutionProvider]       │
│  3. OV EP's CompiledModel takes ownership of the graph (partial or full) │
│  4. OV's CPU plugin runs the compiled IR:                                │
│     - MKLDNN kernels for Conv, MatMul, BatchNorm                         │
│     - Optimized attention fusion for the DETR decoder heads              │
│     - The 24 graph passes OV captured at session init fire here          │
│  5. Output tensors flow back up the stack as numpy arrays                │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

At c=8 with 4 workers × 3 intra-op threads (via `LAYOUT_ONNX_THREADS=3`), this call takes ~0.8-1.2 s per single-image forward (measured by `scripts/bench_paddle_ep.py`). The **unpinned OV default** uses what appears to be a shared global thread pool across sessions, which scales better than a fixed per-session pool (we verified this the hard way — see session report's "Experiment A").

Output is a tuple of three numpy arrays:

| output | shape | meaning |
|---|---|---|
| `fetch_name_0` | `(N_det, 7)` float32 | detection table: `[class_id, score, x1, y1, x2, y2, extra]`, ragged across batch |
| `fetch_name_1` | `(B,)` int32 | per-image detection counts (how to un-ragged `fetch_name_0`) |
| `fetch_name_2` | `(N_det, 200, 200)` int32 | mask tensors (unused by our adapter — we build rectangular polygons from boxes instead) |

### 5.4 Ragged → per-image regrouping

`_ungroup_detections` reads `fetch_name_1` as per-image counts, slices `fetch_name_0` into per-image arrays, and applies the pre-NMS score threshold (from `ld.threshold`, typically 0.5).

### 5.5 800² → original image coord rescale

Because the `scale_factor` input is a no-op, boxes come back in 800² space. `_rescale_and_build_raw` does the manual rescale:

```python
sx = w_orig / 800.0
sy = h_orig / 800.0
x1, y1, x2, y2 = dets[:,2]*sx, dets[:,3]*sy, dets[:,4]*sx, dets[:,5]*sy
```

Also builds a rectangular `polygon_points` per detection (corners of the box), and synthesizes `order_seq` as `arange(N)` — Paddle2ONNX doesn't have glmocr's DETR reading-order head, so downstream reading-order is just the model's natural output order (not a semantic reading order, but stable).

### 5.6 NMS and label routing

Hands off to the shared `np_apply_layout_postprocess` in `layout_postprocess.py` (reused from the torch path). This does:

- **NMS** with `iou_same=0.6, iou_diff=0.98` — aggressive within a class, permissive between classes.
- **Oversized-image filter** — a detection labeled `image` occupying >82/93 % of page area (orientation-dependent) is dropped (heuristic against spurious whole-page detections).
- **Unclip** (optional, off by default).
- **Merge_bboxes_mode** (off by default).

Then `paddle_to_all_results` maps the class IDs through glmocr's native `id2label` (captured at startup — **not** Paddle's config.json mapping, which uses more granular names that would miss glmocr's routing table; see §6 id2label gotcha). Output is JSON blocks with `bbox_2d` normalized to 0-1000 per-image coords (not original pixels — another subtle point that had me chasing ghosts for 20 min).

### 5.7 Region emission

For each detection that survives post-proc, the layout worker pushes a region tuple onto the region queue. A typical OmniDocBench page yields **12-20 regions**, mostly `text` with occasional `table`, `formula`, `footer`. Our measured mean is 14.3 regions/page at c=8.

---

## 6. Recognition worker (Stage 3) — the fan-out

`recognition_worker` pulls `(unit_id, region, task_type)` tuples from the region queue. Per-region:

1. Crop the region out of the original PIL image using the box coords.
2. Call `page_loader.build_request_from_image(crop, task_type)` to build an OpenAI-compatible request.
3. Submit the request to `ocr_client.process()` via a ThreadPoolExecutor of size `max_workers=OCR_MAX_WORKERS=32`.

The ThreadPoolExecutor is the **intra-request** fan-out — all 14 regions of a single page fire concurrently to SGLang.

### 6.1 Request construction with prefix-pin (§8)

This is where the 2026-04-24 monkey-patch lives. `runtime_app.py:549-597` replaces `PageLoader.build_request_from_image` at worker startup with:

```python
content = [
    {"type": "text", "text": "Transcribe the text in the image."},  # stable per task
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
]
```

**Text first, image second.** This is the opposite of upstream glmocr. It matters because SGLang's RadixCache only dedupes on **leftmost common tokens** — with image-first, every region's prefix is different image tokens → 0 % cache hit. With text-first, the prompt tokens ("Transcribe the text..." + chat template wrapper) are identical across regions → cache hit. Measured win: 12 % → 56 % hit rate, TTFT cut in half at c=8.

### 6.2 Base64 encoding

`load_image_to_base64` (also in `page_loader.py`) JPEG-encodes the PIL crop at quality 85 (configurable), base64-encodes, builds the `data:image/jpeg;base64,...` URL. Typical crop is 10-30 KB of base64. This runs on the worker thread and does NOT release the GIL (pure Python path).

### 6.3 aiohttp request

`OCRClient.process()` uses a shared `aiohttp.ClientSession` with `connector=TCPConnector(limit=OCR_CONN_POOL=2048)`. The pool is dimensioned to `CPU_THREADS × OCR_MAX_WORKERS × safety = 2048` so no request ever waits for a connection slot.

The actual HTTP payload is OpenAI-compatible:

```json
POST http://sglang:30000/v1/chat/completions
{
  "model": "glm-ocr",
  "messages": [{"role": "user", "content": [{"type": "text", ...}, {"type": "image_url", ...}]}],
  "max_tokens": 2048,
  "temperature": 0.0,
  ...
}
```

### 6.4 Retry logic

`OCRClient` has a built-in retry loop (from glmocr config: `retry_max_attempts: 2`, exponential backoff 0.5-8 s, retries on 429/500/502/503/504). Under SGLang overload this fires, and on the 3rd attempt if SGLang is still unhealthy the request comes back empty — **this is the silent-empty path that fooled load drivers before solution (b) was queued** (see OPTIMIZATIONS.md "Driver body-content assertion" Queued entry).

---

## 7. The SGLang GPU side

Now we leave the CPU container and enter `glmocr-sglang`. This container has GPU passthrough (`deploy.resources.reservations.devices` with `capabilities: [gpu]` in compose).

### 7.1 uvicorn → OpenAI-compatible handler

SGLang serves a FastAPI app via uvicorn on :30000. `POST /v1/chat/completions` hits SGLang's request dispatcher. The request gets an ID, queued into SGLang's tokenizer thread.

### 7.2 Tokenization + chat template

The tokenizer (from `zai-org/GLM-OCR`) renders the message list via its Jinja chat template. For our prefix-pin request:

```
<|begin_of_sentence|><|start_of_role|>user<|end_of_role|>Transcribe the text in the image.<image><|end_of_role|>...
```

The `<image>` token is a placeholder; the VLM's vision encoder replaces it with the actual image token embeddings at prefill time. Tokenization is fast (~0.1 ms), CPU-side, and critically produces a **deterministic token sequence for the prompt prefix** — this is what makes prefix caching work.

### 7.3 RadixCache lookup

Before scheduling, SGLang's `TokenToKVPoolAllocator` checks whether the leftmost N tokens of this sequence already have K/V computed and in the KV cache. The cache is a **radix tree** keyed on token sequences, so if our prompt prefix has been seen before, the first ~20-30 tokens of prefill are SKIPPED — `prefill_cache` counter increments, `prefill_compute` stays flat.

Measured at c=8 post-prefix-pin: 56.5 % of prefill tokens are cache hits, 12.3 % post-0.83-mem (see §9 — smaller KV cache thrashes harder).

### 7.4 Scheduler

`SGL_SCHEDULE_POLICY=lpm` (Longest-Prefix-Match) — preferred over `fcfs` because at our workload it clusters requests with the same prompt prefix, increasing cache reuse. `fcfs` regressed rps 16-28 % at c=12/24 in earlier memory (`project_sgl_schedule_policy`). The scheduler maintains:

- **running batch** — requests currently being decoded; capped by `SGL_MAX_RUNNING_REQUESTS=64`.
- **waiting queue** — requests that arrived but haven't been batched yet.

At c=8 with 112 regions in-flight competing for the 64-cap running batch, most regions spend 10+ seconds in the waiting queue before being scheduled. This is the **TTFT component** in our per-region metrics.

Under `SGL_MEM_FRACTION_STATIC=0.83` the KV cache holds 24,298 tokens total; at c=8 with ~300 concurrent tokens per region × 16+ concurrent regions, that's tight. Under `0.95` it was 37,710 tokens (larger cache but less dynamic memory for activations, which caused crashes at c≥24).

### 7.5 Prefill

New token K/V (those not found in RadixCache) go through **chunked prefill** — `SGL_CHUNKED_PREFILL_SIZE=8192` tokens per forward pass. For a typical ~600-token prompt this is one chunk; larger documents may span multiple. Each chunk is a forward pass through the model on the GPU:

- Image embeddings come from the VLM's vision encoder (a small CNN+projector).
- Text tokens get their usual embedding lookup.
- Combined sequence runs through the transformer stack (attention + MLP × N layers).
- K/V for all positions is written to the KV cache.

The compute here is attention-heavy, ~1-2 s for a 600-token new prefill on the 3060 Ti when the GPU isn't queue-bound.

### 7.6 Decode loop

Once prefill is done, SGLang enters the decode loop. Every step generates one token per request in the running batch. On each step:

**Fast path — CUDA graphs** (`SGL_CUDA_GRAPH_MAX_BS=8` default): pre-captured CUDA graphs for batch sizes 1..8 replay in ~8-12 ms per batch. This is the typical case for our c≤8 working concurrency.

**Slow path — eager execution** (when running_batch > 8): graph replay misses, falls back to `torch.cuda` kernel launches per-op. ~3-5× slower. Per `project_gpu_utilization_2026_04_23` memory, running batch peaks at 11-16 on our workload — so some decode steps hit the slow path. This is an unshipped optimization: bumping `SGL_CUDA_GRAPH_MAX_BS=16` would cover the peak but adds ~200-400 MB of graph memory, which on the 8 GB card at 0.83 mem fraction is a tight fit. Not shipped, but documented.

**Speculative decoding** (`SGL_SPECULATIVE=true, SGL_SPEC_ALGORITHM=NEXTN`): EAGLE-style draft model predicts `SGL_SPEC_NUM_STEPS=3` tokens ahead; target model verifies all 3 in one forward pass. Success rate depends on content predictability — for repetitive text it's 2-2.5× speedup, for dense formulas it can actually slow things down. On GLM-OCR's doc text it seems net-positive (memory note: "MTP already effectively on (NEXTN→EAGLE alias)"). The `SGL_SPEC_EAGLE_TOPK=1, SGL_SPEC_NUM_DRAFT_TOKENS=4` knobs tune the branching factor.

Actual decode throughput on our 3060 Ti: **~0.5 s per region** (measured via SGLang's `e2e_request_latency_seconds_sum - time_to_first_token_seconds_sum` delta). That's the real GPU work. Everything above 0.5 s in per-region wall time is queue wait (see ARCHITECTURE-v2 §8 budget below).

### 7.7 Token streaming out

Tokens are streamed through SGLang's output processor: detokenize, apply any stop strings, buffer into the response. The OpenAI-compatible endpoint can stream via SSE, but our CPU client uses non-streaming mode — SGLang buffers the full response then returns `{"choices":[{"message":{"content":"..."}}]}`.

---

## 8. Back to the CPU container: response assembly

### 8.1 OCRClient receives per-region text

`OCRClient.process()` parses the OpenAI response, extracts `choices[0].message.content`, returns it as the region's text.

### 8.2 Result formatter

As each region completes, `result_formatter.py` threads it back into the page's result. For regions labeled `text`, the text goes into markdown as-is. For `formula`, it's wrapped in `$$...$$`. For `table`, the model is expected to output HTML `<table>` tags verbatim (GLM-OCR is trained to produce this). Headers become `## ...`, footers/page-numbers are appended at the end.

Final markdown is a concatenation in reading order — which, because our Paddle2ONNX adapter synthesizes `order_seq` as the model's natural output order rather than a true reading-order head, is sometimes subtly suboptimal on multi-column layouts. Acceptable tradeoff for the batching correctness win.

### 8.3 PipelineResult yield

The recognition worker writes the finished page into the state's output slot. `_emit_results` in `pipeline.py` reads from the output side in order (if `preserve_order=True`) and yields one `PipelineResult` per input unit back to the handler.

### 8.4 Flask `jsonify` + gunicorn send

Back in `server.py:_build_response` → `jsonify()` → WSGI app returns. Gunicorn's gthread sends the response bytes. Connection closes (HTTP/1.1 `Connection: close` by default).

---

## 9. Per-stage latency budget at c=8 (measured, shipped stack)

For one 20-page burst at c=8 (n=20 seed=42):

```
16.5 s  per request (client-measured mean)
├── 3.3 s   layout stage (serial)                      ~20 %
│     ├── 0.05 s  PIL preprocess + build ORT feed
│     ├── ~0.8 s  ORT + OV EP forward (amortized if batched)
│     ├── ~0.05 s numpy NMS + postproc
│     └── ~2.4 s  waiting for prior layout call to return (batch window)
│
├── 12.0 s  recognition stage wall (parallel 14 regions @ OCR_MAX_WORKERS=32)   ~73 %
│     Per-region (mean over 14 × 8 concurrent = 112 in-flight regions):
│     ├── ~6.3 s  TTFT (queue wait + prefill on SGLang)     ~90 % of region time
│     │         Decomposes further into:
│     │         - queue wait: ~5.5 s  (the binding factor)
│     │         - prefill:    ~0.8 s  (varies with cache-hit %)
│     └── ~0.3 s  decode (actual GPU token generation)      ~5 %
│
└── ~1.2 s  other (preprocess, dispatch, serialization)                       ~7 %
```

Three load-bearing invariants:

1. **Layout is the CPU bottleneck**, but at 20 % it's not the binding bottleneck — SGLang queue wait is.
2. **Actual GPU compute is ~5 %** of wall time. The GPU mostly waits.
3. **Prefix caching is the big latency lever** — every % of cache hit shaves directly off prefill. Our current 12-56 % hit rate (varies with load) has significant headroom.

At c=16 the queue depth roughly doubles and TTFT climbs to ~13 s, but actual decode stays ~0.5 s. See OPTIMIZATIONS.md §8 and the session report for measurements.

---

## 10. Failure modes and how they surface

### 10.1 Silent empty-markdown

Most dangerous class of failure — HTTP 200 with `markdown_result=""`. Three known causes:

- Layout ONNX crashes mid-batch (the `node_view_320` Reshape bug on the torch export; fixed in §6).
- SGLang unavailable / refuses connection → glmocr's internal `OCRClient` retry exhausts, returns empty.
- Layout detects 0 regions (edge case for almost-blank pages; legitimately empty).

All three look identical to a naive load driver that only checks status code. The shipped drivers still have this hole — see OPTIMIZATIONS.md "Driver body-content assertion" Queued.

### 10.2 Timeouts

- `OCR_REQUEST_TIMEOUT=60` bounds each CPU→SGLang HTTP call.
- `OCR_RETRY_MAX=2` allows up to 3 total attempts per region.
- `GUNICORN_TIMEOUT=480` bounds the whole `/glmocr/parse` call end-to-end from the outside.

If SGLang is truly down, the handler takes up to 3 × 60 = 180 s before giving up per region. Under the 20 ms layout batch window, this can cascade into multiple pages' worth of regions timing out together. Mitigated by the `_health_watchdog` thread which proactively aborts all in-flight work if SGLang `/health` goes non-200 for > N seconds.

### 10.3 SGLang crashes / OOMs

On the 8 GB dev card at `SGL_MEM_FRACTION_STATIC=0.95`, c≥24 used to crash SGLang mid-run (OOM on the dynamic pool). Fixed by dropping to 0.83 (§9). The `_health_watchdog` on the CPU side detects this within 5-10 s and surfaces it in logs. SGLang auto-restarts (Docker `restart: unless-stopped`), re-captures CUDA graphs (~10 s), rebuilds KV cache, and resumes — but all in-flight requests die.

### 10.4 gunicorn worker hangs

If a worker hits a deadlock inside PIL or numpy (historical bug in some versions), the `gunicorn --max-requests N` setting will recycle the worker after N requests regardless. Set via `GUNICORN_TIMEOUT` separately for per-request deadlocks. `faulthandler` is registered on `SIGTERM/SIGUSR1` so a stuck worker can be introspected via `kill -SIGUSR1 <pid>`.

### 10.5 KV cache thrash

Not a failure per se, but a performance pathology. When many concurrent requests evict each other's prefix K/V, prefix-cache hit rate crashes (we saw 56 % → 12 % going from 0.95 mem to 0.83). This doesn't surface as an error — it surfaces as climbing TTFT. The `sglang:realtime_tokens_total{mode="prefill_cache"}` counter is the canonical signal.

---

## 11. Cross-reference map (where to find what)

| topic | deep-dive in |
|---|---|
| Paddle2ONNX adapter | `docker/cpu/layout_paddle2onnx.py` + `OPTIMIZATIONS.md §6` |
| OV EP wiring | `docker/cpu/runtime_app.py:595-635` + `OPTIMIZATIONS.md §7` |
| Prefix-pin monkey-patch | `docker/cpu/runtime_app.py:549-597` + `OPTIMIZATIONS.md §8` |
| Mem-fraction tradeoff | `.env:SGL_MEM_FRACTION_STATIC` + `OPTIMIZATIONS.md §9` |
| Rejected paths & why | `OPTIMIZATIONS.md Rejected` section + `docs/omnidoc-2026-04-24-paddle-ov-shipment.md` |
| Session experiment log | `docs/omnidoc-2026-04-24-paddle-ov-shipment.md` |
| Matrix-noise calibration | auto-memory `feedback_matrix_noise.md` |
| File-by-file tour | `docs/ARCHITECTURE.md` (v1) |

---

## 12. Threading model — quick recap

**Per CPU container at c=8**:

```
4 gunicorn workers × 16 gthreads = 64 max concurrent request-handling threads
  → each concurrent /glmocr/parse call uses ONE gthread
  → that gthread spawns 3 daemon threads (loader, layout, recognition)
  → the recognition thread uses a ThreadPoolExecutor(OCR_MAX_WORKERS=32)
  → each of those 32 pool threads holds an aiohttp Session connection (from pool of 2048)

At c=8: ~8 gthreads busy, ~24 daemon threads per pipeline active,
        ~112 aiohttp requests in-flight to SGLang, ~300 Python threads total.

Inside each layout call: OV CPU plugin spawns its own MKLDNN thread pool,
effectively shared across concurrent sessions (unpinned default).
```

**SGLang side**:

```
1 uvicorn worker (async) handles all HTTP requests
  → tokenizer thread (sync, fast)
  → scheduler thread runs the running-batch loop
  → GPU compute is single-process, single-device

Running batch cap 64, actual peak 11-16 (CPU-layout-limited arrival rate).
KV cache 24 k tokens total at 0.83 mem fraction.
```

The whole system is stateful-per-worker (aiohttp pools, ORT sessions, ld.process monkey-patches) and stateless-across-workers (no shared memory between gunicorn workers beyond the Prometheus multiproc dir). A gunicorn worker restart loses its local state but not request history.

---

**End.** If a piece of the path is still opaque after reading this, grep the referenced file:line, or add a new section and file a PR — this doc is meant to stay in sync with shipped reality, not lead it.
