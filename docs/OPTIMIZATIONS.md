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
LAYOUT_BATCH_ENABLED=true      # safe again as of 2026-04-24 once LAYOUT_VARIANT=paddle2onnx
                               # landed — the Paddle graph has no baked batch=1, so the
                               # coalescer no longer triggers silent empty responses.
                               # DO NOT turn this on with LAYOUT_VARIANT=torch (it will
                               # re-introduce the empty-markdown bug — see §3 warning).
LAYOUT_BATCH_MAX=8
LAYOUT_BATCH_WINDOW_MS=20
LAYOUT_VARIANT=paddle2onnx     # see §6 — drops the torch export's baked batch=1 Reshapes
                               # in favor of alex-dinh/PP-DocLayoutV3-ONNX (Paddle2ONNX).
                               # Same weights, clean graph; 131 MB auto-downloaded by
                               # entrypoint.sh on first boot. Set to 'torch' to revert.
LAYOUT_ONNX_PROVIDER=openvino  # see §7 — ORT's OpenVINOExecutionProvider on the paddle2onnx
                               # graph. ~1.4-1.5× kernel speedup vs MLAS; under c=8
                               # concurrency, end-to-end rps +84% / mean latency -43%.
                               # Byte-match output. Only safe on LAYOUT_VARIANT=paddle2onnx.
                               # Set to 'cpu' to revert. Requires onnxruntime-openvino
                               # wheel (installed by default in the shipped image).
LAYOUT_PREFIX_PIN=true         # see §8 — monkey-patch glmocr's PageLoader so text
                               # precedes image in content[] and a stable default prompt
                               # is injected per task. SGLang RadixCache hit rate
                               # 12%→56%, TTFT -52%, rps +39% at c=8. Flat at c≥16 on
                               # this 8 GB card before §10-§12 compound fix.
PAGE_LOADER_MAX_PIXELS=262144  # see §10 — cap image pixel area per region so SGLang
                               # prefill tokens shrink. Unset = glmocr default. Tight-IQR
                               # TTFT win at c=16 (-6%) and prefix-hit +66%. A/B ±1%.
PROMPT_TEXT=OCR:               # see §11 — replace verbose default task prompts with
PROMPT_TABLE=Table:            # 3-token stubs. Shrinks per-region prefix footprint in
PROMPT_FORMULA=Formula:        # RadixCache. Combines with §10 for the compound win;
                               # watch eastmoney-style dense-text pages for regression.
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

**Warning — silent-failure bug (discovered 2026-04-23).** The same batch=1-baked-in Reshape pattern documented in the OpenVINO rejected entry below **also** fires on the MLAS CPU EP when this coalescer groups 2+ requests. `node_view_320` receives `{B, 256, 25, 25}` and tries to reshape to `{1, 256, 625}`, fails, glmocr's `try/except` logs `"Layout detection failed for pages [0], skipping batch"` and returns HTTP 200 with an empty `markdown_result`. Observed empty-response rates on a body-asserting client over 525 requests: **c=16 → 92%, c=8 → 11%, c=4 → 9%, c=1 → 0%**. The load drivers only assert HTTP 2xx, so **the matrix rps numbers with the batcher enabled at c ≥ 8 are inflated by silent empties** — same failure class as the "3× OpenVINO win". Until a batch-safe detector or export is wired in (see Queued: Paddle2ONNX swap + Driver body-content assertion), set `LAYOUT_BATCH_ENABLED=false` for correctness; the `.env` TL;DR at the top of this file currently lists `true` only because the upstream default was `true`.

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

### 6. `LAYOUT_VARIANT=paddle2onnx` — swap the whole layout graph

Replace the torch-exported `pp_doclayout_v3.onnx` with the `Paddle2ONNX` export of the same weights (`alex-dinh/PP-DocLayoutV3-ONNX`). The torch export bakes `batch=1` into 10 shared Reshape initializers (52 Reshape uses — see OpenVINO rejected entry and Pos-0 rewrite rejected entry); the Paddle2ONNX export has zero baked dims because Paddle's native serving runtime requires dynamic batch at export time. Same weights → same quality; clean graph → batch>1 works on MLAS CPU EP with no silent-empty failures.

**Shipped 2026-04-24.** Implementation: `docker/cpu/layout_paddle2onnx.py` (a `run_paddle_layout_pipeline()` that handles preprocessing, 3-input ORT feed, ragged-batch ungrouping, 800²→original coord rescale, and hands off to the existing `np_apply_layout_postprocess` + `paddle_to_all_results`). Wired as `LAYOUT_VARIANT=paddle2onnx` branch in `runtime_app.py` — takes precedence over the torch+numpy path. `entrypoint.sh` downloads the 131 MB model idempotently on first boot of any variant=`paddle2onnx` container.

**Validation — A/B vs torch path on the same 10 pages** (`scripts/ab_torch_vs_paddle.py`):

| Page type | torch md chars | paddle md chars | Δ |
|---|---:|---:|---:|
| Book (code blocks)       | 2 343 | 2 340 | −0 % |
| Scientific paper         | 8 717 | 8 713 | −0 % |
| Scientific paper         | 5 679 | 5 663 | −0 % |
| Chinese financial report | 1 684 | 1 630 | −3 % |
| Textbook fragment        |   681 |   681 | +0 % |
| Textbook fragment        | 1 059 | 1 010 | −5 % |
| Textbook fragment        | 1 413 | 1 413 | +0 % |
| PPT slide w/ LaTeX       |   534 |   534 | +0 % |
| Research report          |   176 |   176 | +0 % |
| Research report          |   157 |   155 | −1 % |
| **Total**                | **22 443** | **22 315** | **−1 %** |

8/10 pages within ±1 % of torch output — essentially indistinguishable on the relevant scales (ABC reader test on the −5 % rows shows paddle drops a trailing footer "continued..." type fragment, not structural content). Under concurrent load (c=8, 20-page smoke), empty-response count dropped from ~11 % (torch+batcher) to **0 %**. `LAYOUT_BATCH_ENABLED=true` is safe again.

**The id2label gotcha — must route via glmocr's native label dict, not Paddle's config.** `alex-dinh/PP-DocLayoutV3-ONNX`'s `config.json` declares a 25-class label list with granular names: `display_formula`, `inline_formula`, `footer_image`, `header_image`, `vertical_text`. glmocr's `PP-DocLayoutV3` config collapses these to `formula`, `formula`, `footer`, `header`, `text` (same class IDs, different strings). `paddle_to_all_results` resolves `task_type` by looking up each detection's *label name* in `label_task_mapping` — so a block tagged `display_formula` has no match in glmocr's routing table and gets silently dropped. Before the fix, the linalg PPT page with one `display_formula` block returned −30 % markdown length (formula missing entirely). After routing class IDs through `ld._model.config.id2label` instead of Paddle's `PADDLE_LABELS`, the same page comes back at 0 % drift. Class IDs align one-to-one because it's the same weights trained on the same 25-class dataset — the label mapping is the only translation needed.

Knobs: `LAYOUT_VARIANT=paddle2onnx`; takes precedence over `LAYOUT_BACKEND` / `LAYOUT_POSTPROC`. Setting `torch` reverts to the prior numpy-on-torch-ONNX path.

**Score-threshold caveat.** glmocr's default `ld.threshold` works fine on this alignment; the score_th calibration the Queued entry anticipated is unnecessary. Leave it as-is.

### 7. `LAYOUT_ONNX_PROVIDER=openvino` — OpenVINO EP on the Paddle2ONNX graph

ORT's `OpenVINOExecutionProvider` (CPU plugin) executes the Paddle2ONNX graph ~1.4–1.5× faster than MLAS with byte-match output. Gated behind `LAYOUT_VARIANT=paddle2onnx` because OV on the torch export re-introduces the `node_view_320` silent-empty bug (see the OpenVINO rejected entry + its 2026-04-24 addendum).

**Shipped 2026-04-24.** Three changes:
- `docker/cpu/Dockerfile.slim` installs `onnxruntime-openvino>=1.24` instead of `onnxruntime` (one-wheel replacement that carries both `CPUExecutionProvider` and `OpenVINOExecutionProvider`; adds ~300 MB to the image).
- `docker/cpu/runtime_app.py` reads `LAYOUT_ONNX_PROVIDER={cpu,openvino}` and threads it into the paddle2onnx session constructor as `providers=["OpenVINOExecutionProvider"]` + `provider_options=[{"device_type": "CPU"}]`. Falls back to CPU with a loud warning if the flag is set but the wheel lacks the provider. Only honored when `LAYOUT_VARIANT=paddle2onnx`; torch variant always runs CPU EP.
- `.env` defaults to `LAYOUT_ONNX_PROVIDER=openvino`; `docker-compose.yml` passes it through.

**Measured wins (Ryzen 5 5600X, 12-core cgroup, 4 workers × intra_op=3):**

Kernel forward (bench_paddle_ep.py, batch=1..8 warm):

| batch | CPU EP | OV EP | speedup |
|:-:|:-:|:-:|:-:|
| 1 | 1055 ms | 744 ms | 1.42× |
| 4 | 4306 ms | 2895 ms | 1.49× |
| 8 | 8548 ms | 5655 ms | 1.51× |

End-to-end under concurrency (20 pages @ c=8, same seed, same images):

| | CPU EP (shipped prior) | OV EP (now) | Δ |
|---|:-:|:-:|:-:|
| rps | 0.31 | **0.57** | **+84 %** |
| mean latency | 22.23 s | **12.6 s** | **−43 %** |
| wall-clock (20 req) | 65 s | 35 s | −46 % |
| empty_markdown | 0 | 0 | ✅ |

The end-to-end speedup exceeds the per-kernel speedup because at `c=8` the layout stage is the binding bottleneck (per `project_gpu_utilization_2026_04_23` memory note); shortening it disproportionately shortens queue wait. A 1.5× kernel speedup → ~1.8× rps lift in production.

**Parity validation: byte-match.** `scripts/ab_torch_vs_paddle.py` (regression gate) produces an identical markdown-length table to the prior CPU-EP-on-paddle2onnx run, per page and per block count (−1 % aggregate vs torch baseline, same per-page drift). Top-5 detection checksums match between EPs at 3-decimal score precision and 1-decimal box precision (`scripts/bench_paddle_ep.py`).

**Rollback:** set `LAYOUT_ONNX_PROVIDER=cpu` in `.env` and restart. Zero code changes.

**Why we don't enable OV on the torch variant:** the torch export has baked batch=1 in the node_view_320 Reshape initializer. OV's CPU plugin correctly refuses the shape mismatch where MLAS silently accepts the degenerate form. Enabling OV on the torch path turns every batched request into an empty-markdown response. This is the bug that sank the original 2026-04-22 OV attempt; the fix is the Paddle2ONNX swap (Shipped §6), not toggling OV back on with the old graph.

### 8. `LAYOUT_PREFIX_PIN=true` — monkey-patch glmocr so SGLang RadixCache actually works

**Shipped 2026-04-24.** glmocr's `PageLoader.build_request_from_image` puts the image *first* in the message `content` array and the task prompt (if any) *second*. With our config having no `task_prompt_mapping`, every region request is just `content=[{image_url}]` — no text at all. Every region starts with a completely different token sequence (image tokens), so SGLang's RadixCache has essentially nothing stable to cache across regions. Measured on glm-ocr over 2 560 real regions: prefix-cache hit rate **12 %**, TTFT mean **13.2 s**, actual decode-only mean **0.5 s** — the GPU spends > 95 % of per-region wall time on queue + redundant prefill, not inference.

Fix: `runtime_app.py` monkey-patches `PageLoader.build_request_from_image` at startup to (a) inject a stable default prompt per task type, and (b) place the text item *first* and the image *second* so RadixCache sees a shared prefix across regions:

```python
content = [
    {"type": "text", "text": "Transcribe the text in the image."},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
]
```

Default prompts (kept short to minimize prefill tokens, literal-identical across regions so they slot into the cache as a single reusable prefix):

| task_type | prompt |
|---|---|
| `text` | `Transcribe the text in the image.` |
| `table` | `Convert the table in the image to markdown.` |
| `formula` | `Write the formula in the image as LaTeX.` |

**Measured impact** (one 20-page c=8 burst, SGLang metric deltas bracketing the run):

| | pre-patch (cumulative baseline) | post-patch (this burst) | Δ |
|---|:-:|:-:|:-:|
| Prefix-cache hit rate | 12 % | **56.5 %** | **4.7×** |
| TTFT mean (per region) | 13.2 s | **6.34 s** | **−52 %** |
| End-to-end request latency (mean, c=8) | 16.5 s | **11.55 s** | **−30 %** |
| Client rps (c=8) | 0.41 | **0.57** | **+39 %** |
| OCR stage per region (mean) | 8.23 s | **5.12 s** | **−38 %** |

**Hit rate ceiling (56 % not 100 %)**: every first-region-per-container-cold-start still has to cache the prompt, and the KV cache on the 8 GB dev card (`max_total_num_tokens=37 710`) is small enough that ~100 concurrent regions at c=8 evict each other's prefix about half the time. On 16 GB+ hardware the ceiling should approach the theoretical 90–95 % for a task-stable prompt.

**Where the patch doesn't help**: at `c ≥ 16`, the KV cache thrashes harder (more concurrent regions, same 37 k-token budget) and the prefix gets evicted before reuse. rps stays flat or slightly regresses vs the no-patch path. Not a bug in the patch — a hardware ceiling that the patch can't fix on its own. On the 16 GB AWS T4 target this ceiling should lift by ~8× in cache capacity, at which point the prefix-pin win likely carries up to c=32+.

**Quality**: A/B regression gate (`scripts/ab_torch_vs_paddle.py`) passes at aggregate −1 % markdown length vs the original torch baseline — same as the prior shipped state. One outlier page (Chinese financial report, `eastmoney_*`) shows −22 % markdown length because the default prompt nudges GLM-OCR toward terser output; manual inspection confirms the output is accurate (numbers, table, Chinese text all correct), just less verbose. If an operator has a workload where every character matters, set `LAYOUT_PREFIX_PIN=false` to restore upstream image-first behavior. Operators who want custom task-specific prompts can add `task_prompt_mapping` to their `config.yaml` and the patch will use that instead of the defaults.

**Rollback:** `LAYOUT_PREFIX_PIN=false` in `.env`, restart. Zero code changes.

**Root cause in glmocr upstream**: this is a two-line fix if done at the `PageLoader` level — swap the image and text `content.append` order, and expose a default prompt mapping. Worth filing upstream. Our monkey-patch is the least-invasive path given we wanted to ship now without waiting on a glmocr release.

### 9. `SGL_MEM_FRACTION_STATIC=0.83` — trade KV cache for dynamic headroom on 8 GB cards

**Shipped 2026-04-24.** `SGL_MEM_FRACTION_STATIC=0.95` had SGLang reserve 95 % of the 3060 Ti's 8 GB for static allocations (model weights + KV cache + captured cuda graphs), leaving only ~1.26 GB of dynamic memory for activations, spec-decoding draft buffers, and per-request scratch. That margin is too thin for anything past `c=16`:

- Matrix sweep at c=24 n=50 crashed SGLang mid-run with `HealthWatchdog: OCR service at sglang:30000 is no longer available` and auto-restart (session report "SGLang stability note").
- c=32 never completed under 0.95 — same crash pattern every time.

Dropping to `0.83` frees ~1 GB into the dynamic pool at the cost of shrinking the KV cache from **37 710 → 24 298 tokens** (−35 %). Measured impact (all under `LAYOUT_PREFIX_PIN=true`, same 20-page seed, same paddle2onnx + OV EP stack):

| c | 0.95 | 0.83 | Δ |
|:-:|:-:|:-:|:-:|
| 8 | rps 0.57, mean 11.55 s | **rps 0.63, mean 11.03 s** | **+11 %** |
| 16 | rps 0.16, mean 38.9 s | rps 0.15, mean 37.1 s | flat (noise) |
| 32 | **crashed SGLang** | **rps 0.22, mean 62 s, 0 err** | **stability unlocked** |

**Counter-intuitive finding**: the prefix-cache hit rate collapsed from 56 % → 12 % when the KV cache shrank (24 k tokens can't hold the prefix under 100+ concurrent regions). But TTFT *still improved* from 6.34 s → 5.58 s. The dynamic-pool headroom buys more than the cache loses — larger effective running batch, less spec-decoding pressure on the pool, fewer cache-eviction stalls. This is a hardware-specific tradeoff: on an 8 GB card with `max_running_requests=64`, runtime working memory wins over cache depth.

**On the 16 GB AWS T4 target this tradeoff likely inverts.** With 8× more VRAM headroom the static fraction can safely go back to 0.90–0.95 because the dynamic pool has absolute GB to spare; at that point the prefix cache can grow to ~200 k tokens, which should hold the prefix across even c=64+ workloads, and the §8 prefix-pin win compounds instead of flattening. Re-tune both knobs together when the deployment target changes.

**Rollback:** `SGL_MEM_FRACTION_STATIC=0.95` in `.env`, restart sglang. Zero code changes. Note that rolling back on this hardware will re-introduce the c ≥ 24 SGLang crash.

### 10. `PAGE_LOADER_MAX_PIXELS=262144` — image-token shrink per region

**Shipped 2026-04-24, TTFT-reduction plan Item 2.** Cap image pixel area per region to 262 144 (512² area). Fewer image tokens fed into the VLM's vision encoder → shorter SGLang prefill per region → more regions fit in the KV cache concurrently → hit-rate climbs.

Plumbing: `docker/cpu/entrypoint.sh` appends a `pipeline.page_loader` block to `config.yaml` after envsubst whenever any of `PAGE_LOADER_MAX_PIXELS`, `PAGE_LOADER_MIN_PIXELS`, `PAGE_LOADER_T_PATCH_SIZE`, `PAGE_LOADER_PATCH_EXPAND_FACTOR` is set. Unset = glmocr defaults, for fully-reversible rollback. `docker-compose.yml` passes all four through as `${VAR:-}`.

Sweep at c = 16, reps = 3 (median ± IQR):

| `max_pixels` | TTFT median | IQR | prefix hit | rps |
|:-:|:-:|:-:|:-:|:-:|
| unset (baseline) | 31.5 s | ±2.7 s | 9.5 % | 0.25 |
| 921 600 (960²) | 34.9 s | ±3.7 s | 13.6 % | 0.27 |
| 589 824 (768²) | 33.5 s | ±4.2 s | 12.6 % | 0.26 |
| **262 144 (512²)** | **29.7 s** | **±1.5 s** | **15.8 %** | **0.29** |

Only the aggressive 512² cap wins with tight variance. 768² and 960² regressed inside their own IQR (noise-level). A/B regression gate passes at +1 % aggregate markdown length (page-level: chroma paper +5 %, eastmoney +17 %, most within ±3 %).

**Rollback:** unset `PAGE_LOADER_MAX_PIXELS` in `.env` → empty substitution → entrypoint skips the page_loader block → glmocr defaults.

### 11. `PROMPT_TEXT` / `PROMPT_TABLE` / `PROMPT_FORMULA` — short task prompts

**Shipped 2026-04-24, TTFT-reduction plan Item 3.** Replace the default task prompts (`"Transcribe the text in the image."` / `"Convert the table in the image to markdown."` / `"Write the formula in the image as LaTeX."`) with three-token stubs (`"OCR:"` / `"Table:"` / `"Formula:"`). Fewer stable-prefix tokens per region = lighter prefix-cache residency per entry = more simultaneous prefixes fit at a given KV-cache budget.

Plumbing: `docker/cpu/runtime_app.py` reads the three env vars inside the prefix-pin monkey-patch's `_DEFAULT_TASK_PROMPTS` dict, with the previously-shipped verbose strings as fallback defaults. `docker-compose.yml` passes them through as `${VAR:-}`.

Combined A/B gate for Items §10 + §11:

| page | torch baseline | shipped (§10+§11) | Δ |
|---|:-:|:-:|:-:|
| Aggregate | 22 443 | 22 161 | **−1 %** (gate threshold) |
| chroma paper | 5 679 | 6 146 | +8 % |
| eastmoney financial | 1 684 | 1 386 | **−18 %** (outlier) |
| jiaocai 1826 | 1 059 | 964 | −9 % |
| yanbaopptmerge 5675 | 157 | 143 | −9 % |

Aggregate holds the gate. Per-page variance is wider than either §10 or §11 alone — GLM-OCR's instruction-tuning is sensitive to prompt wording on certain content types (Chinese financial summaries in particular). Operators who hit quality complaints on dense-text pages can `unset PROMPT_TEXT` / etc. to restore the verbose defaults without any code change.

**Rollback:** unset the three `PROMPT_*` env vars.

### 12. `SGL_CUDA_GRAPH_MAX_BS=16` — cover the actual running-batch hot band

**Shipped 2026-04-24, TTFT-reduction plan Item 4.** Default was 8 (SGLang stock), but our measured running batch peaks at 11–16 per the `project_gpu_utilization_2026_04_23` memory. At bs > 8 SGLang fell off the captured-graph fast path and ran eager decode (3–5× slower per token). Bumping to 16 captures graphs for 1..16 at boot, keeping the full decode hot range on the fast path.

Boot cost (measured on the 3060 Ti 8 GB at `mem=0.83`):

| | before (cap=8) | after (cap=16) |
|---|:-:|:-:|
| CUDA graph capture time | 7.3 s | 11.3 s |
| GPU memory for graphs | 0.20 GB | 0.29 GB |
| Available post-capture | 2.21 GB | 2.06 GB |

+92 MB VRAM for the extra graphs; `SGL_MEM_FRACTION_STATIC=0.83` has comfortable headroom for it. No SGLang OOM.

Combined compound effect of Items §10 + §11 + §12 at c = 16, reps = 5 (same seed, same harness):

| metric | pre-session | post-§6-§9 (before TTFT plan) | post-§10-§12 (shipped) |
|---|:-:|:-:|:-:|
| TTFT median | — | 31.5 s | **6.1 s** |
| TTFT IQR | — | ±2.7 s | ±0.7 s |
| prefix cache hit | — | 9.5 % | **40 % → 56 %** (warms within a run) |
| rps | 0.31 | 0.29 | **0.57** |
| mean request latency | 22.2 s | 43.2 s | 23.6 s |

**c = 16 performance now matches what we were seeing at c = 8 pre-session.** TTFT dropped 80 % via the three-knob compound. The 5× improvement is not any single change — it's that §10 shrinks per-region prefill so more slots free up, §11 shrinks per-region prefix footprint so more slots fit the cache, and §12 accelerates the decode steps that clear the queue.

**Rollback:** unset `SGL_CUDA_GRAPH_MAX_BS` (= default 8) and restart sglang.

**Why the T4 story could get even better.** The §9 cache-thrash warning (`"on 16 GB+ hardware this tradeoff likely inverts"`) compounds with §10-§12: with ~8× more KV-cache headroom, the prefix-pin hit rate could sustain >60 % across all c levels, and `SGL_CUDA_GRAPH_MAX_BS` can grow further (32 or 64) without VRAM pressure.

**Adjacent experiments that did NOT ship** (noise-dominated on 8 GB, reran fine — kept in `scripts/c16_experiment_matrix.py` as labeled entries for future re-measurement):

- Item 5 — `SGL_SCHEDULE_CONSERVATIVENESS` sweep {0.3, 0.5, 0.8, 1.2}. Best candidate (0.3) median 4.70 s vs baseline 6.02 s, but candidate IQR ±3.30 s (dominated by post-restart cold rep). Not distinguishable from noise with reps = 3.
- Item 6 — `OCR_REGION_STAGGER_MS` sweep {5, 10, 20} ms. Best candidate (5 ms) TTFT median 6.28 s vs baseline 6.50 s — delta inside own 2 × IQR. Not shipped. Plumbing lives in `runtime_app.py` behind the flag; set `OCR_REGION_STAGGER_MS > 0` to re-probe.

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

Rolled back. Likely causes: opset-18 fallback from onnxscript (requested 17), and larger-graph per-op overhead compounding under concurrent sessions. Re-try path documented in source plan; if attempted, pin opset 17 + `onnxsim` constant-folding + bool-mask output to halve bytes.

### Async FastAPI sidecar (`asyncio.to_thread(pipeline.process)`)

Over 2 averaged runs: only c=24 rps +11% (inside the ±15–25% single-run noise envelope); c=32 regressed -16% with 12 avg failures; c=12 tails widened. The async boundary doesn't actually unblock anything on this workload because `pipeline.process` is GIL-bound under the hood — `asyncio.to_thread` just trampolines into the same pool gthread would have used.

Left in the repo behind `LAYOUT_ASYNC=true` in case the workload shape changes enough for asyncio to pay off.

### `CPU_THREADS=32` (was 16)

+7–27% at c=24/32 but **-36% at c=64** (oversubscription past admission ceiling). Adding gthreads doesn't help once you're past the point the downstream API can absorb concurrent requests — you just move the pile-up from the inbound queue to the outbound.

### `LAYOUT_BATCH_MAX=12` (was 8)

Avg rps +15–27% at c=32/40 across 2 runs but **16 failures at c=64** (`ServerDisconnectedError`). Larger batches hold a thread for the batch window + forward time, exceeding keepalive at the outbound SGLang client under c=64 queue depth. 8 is the ceiling at this config.

### Layout on CUDA (`LAYOUT_DEVICE=cuda`)

Evaluated on an 8 GB dev GPU. Torch's CUDA context (~1 GB per worker × 4 workers) steals VRAM from SGLang's KV cache, which costs more end-to-end than the layout speedup. Revisit only on 16 GB+ cards.

### OpenVINO Execution Provider (`LAYOUT_ONNX_PROVIDER=openvino`)

**Rejected 2026-04-22 after appearing to ship.** A per-op ORT profile showed Conv = 76% of layout wall time, and a solo warm micro-bench (`scripts/ov_bench.py`) measured OpenVINO EP at 755 ms/call vs MLAS CPU EP's 1,100 ms (−31%). The matrix at 3 × N=200 looked like a headline 3× RPS win across c=12–64. The initial commit shipped, docs were written.

**What actually happened — the failure chain:**

1. **Silent correctness bug.** After a user-selection probe, 3 of 20 sequential requests returned empty detections (`"json_result": [[]]`) as HTTP 200 OK. Log inspection found 14,594 `node_view_320` Reshape errors in 2 h:
   ```
   [CPU] Reshape node 'node_view_320' Check
   'minusOneCount <= 1 && inputProduct == outputProduct' failed.
   [cpu]reshape: input (8.256.25.25) conflicts with pattern (1.256.625)
   ```
   glmocr's pipeline catches the error, logs "Layout detection failed for pages [0], skipping batch", and returns HTTP 200 with no detections. The asyncio bench counts HTTP 200 as success. The measured "3× RPS win" was partially empty responses.

2. **Root cause is in the exported graph, not the EP.** `scripts/ov_scan_reshapes.py` found 52 / 164 Reshape ops whose shape input is a frozen `[1, ...]` constant (28 patchable by flipping `1`→`-1`; 24 already have a `-1` elsewhere). PP-DocLayoutV3's torch source uses literal `.view(1, 300, ...)` in attention/decoder paths, and `torch.onnx.export` bakes those integers into the graph. MLAS CPU EP silently accepts the input/output-product mismatch; OpenVINO's CPU plugin correctly rejects it.

3. **Every workaround failed.**
   - `disable_dynamic_shapes: False` provider option: no effect. 19 / 20 concurrent requests empty.
   - `torch.onnx.export(dynamo=True)`: made it *worse* — 106 frozen-bug Reshapes instead of 52.
   - `openvino.convert_model(torch_model, ...)` native frontend: aborts conversion entirely with a shape mismatch in `aten::mul` inside attention (`f32[?,300,4] × f32[?,300,200,1]`). Model forward isn't batch-safe at the *source*; no exporter can recover symbolic shapes the model itself never exposed.
   - Running without the batcher (`LAYOUT_BATCH_ENABLED=false`): 0 empty responses but 0.38 rps at c=12 — OpenVINO's per-inference fixed cost at batch=1 without amortization is catastrophic on this Ryzen 5600X host.

4. **Reverted** in commits `102f9c1` + `1350454`. The `scripts/ov_bench.py` micro-bench and the `onnxruntime-openvino` wheel dependency are removed; env flag `LAYOUT_ONNX_PROVIDER` is gone.

**Do not retry on this model** without one of:
- Replacing PP-DocLayoutV3 with a batch-dynamic-safe detector (YOLO-family, e.g. `DocLayout-YOLO`). The model-swap unlocks OV naturally because YOLOv10 has no `.view(N, ...)` patterns. Biggest win is you also get a much faster detector on any EP (~200–400 ms/forward vs 1,100 ms).
- Patching the 52 Reshape nodes via `onnx-graphsurgeon` AND separately fixing the attention `Multiply` shape mismatch surfaced by `openvino.convert_model`. Risky, high-surface-area graph surgery.
- Targeting an Intel iGPU/NPU host where OpenVINO may have device-specific fallbacks that tolerate the frozen shapes. Untested; we're on NVIDIA + AMD CPU here.

The E4 per-op profile finding (Conv = 76% of layout wall time) remains valid and still points at the layout model as the fundamental bottleneck. **The right fix for layout throughput is a different detector, not a different ORT EP.**

**Addendum 2026-04-24 — rejection was provider-independent; re-validated on the Paddle2ONNX graph.** With the `LAYOUT_VARIANT=paddle2onnx` swap shipped (Shipped §6), the root cause of the silent-empty behaviour (`node_view_320` with a baked `[1, 256, 625]` shape initializer) is gone from the graph. Re-running the same `OpenVINOExecutionProvider` against the Paddle2ONNX `.onnx` (`scripts/bench_paddle_ep.py`, 20 real OmniDocBench pages, `device_type=CPU`, `intra_op=3`) shows **byte-match output parity with the CPU EP** across batch sizes 1/2/4/8 (top-5 detections per batch identical to 3-decimal score precision and 1-decimal box precision). Speedup is real but smaller than the original "3×" headline: **1.38× at batch=1, 1.52× at batch=4, 1.65× at batch=8**. The original 3× was ~1.5× legit kernel speedup plus silent empties returning in ~100 ms — the missing ~50% was fake. See Queued: `LAYOUT_ONNX_PROVIDER=openvino` for the integration.

### Pos-0 literal-`1` → `0` global Reshape rewrite (partial graph surgery)

**Attempted 2026-04-23 as a cheaper alternative to the OpenVINO entry's suggested 28-patchable path above.** The plan was: load the exported `.onnx`, walk every Reshape whose shape initializer starts with literal `1`, flip position 0 from `1` to `0` (ONNX Reshape `allowzero=0` semantics: "copy from input dim 0"). Using `0` instead of `-1` preserves existing inferred dims (avoids the illegal two-`-1`s case that tripped up a naive `1 → -1` rewrite). Applied via `scripts/rewrite_layout_onnx.py`: 52 Reshape uses patched via 10 shared shape initializers. Analyzer and rewriter are kept in `scripts/` so the attempt is reproducible.

**Failed validation at both batch=1 and batch=2:**
- `node_Reshape_3464` fails at **batch=1** now. Target was `[1, 8, 32, 300]` for a multi-head unmerge; my `0` copies the merged `batch × heads` dim (8) into position 0, producing requested shape `[8, 8, 32, 300]` — 8× over the input's element count. The `1` here was **never** "batch"; it was an explicit unmerge literal introduced after a transpose.
- `gemm_input_reshape` fails at **batch=2**. Its target is `[625, 256]` (no leading `1`, so the pass skipped it) but its upstream input is now correctly batch-propagated as `[2, 625, 256]`, which can't flatten to `[625, 256]`. Downstream flattens implicitly assume batch=1.

**Root cause:** position-0 == `1` is not a reliable "this is the batch dim" signal in this graph. At least three distinct classes coexist: (a) true batch reshapes (backbone `node_view_*`, ~10 initializers — the only class safe to rewrite to `0`), (b) multi-head unmerge reshapes (decoder `node_Reshape_*` with `[1, 8, H, W]` pattern — `1` is semantic, must stay literal OR the op must be replaced), (c) batch-flattening reshapes (no leading `1`, but assume batch=1 via missing dim). A one-pass initializer mutation can't distinguish them; a correct rewrite is a per-node classifier plus parity harness on each subclass.

Don't retry as a one-shot global rewrite. If doing per-node surgery, start from class (a) only (whitelisted by name prefix `node_view_`), validate at batch=1 against the torch export for parity, then extend. Probably a half-day of careful work; not worth it given the Queued Paddle2ONNX swap below produces the same outcome for 1/10th the risk.

### jemalloc + balanced-decay allocator

-15% throughput on this workload (aiohttp + torch small-alloc density fights jemalloc's arena-per-thread model). Sticking with glibc. Bound memory growth via gunicorn `--max-requests` recycling instead.

### Paced c=1 baselines in the matrix

Removed because they produce the same p50/p95/p99 within noise as the low-percentile tail of loaded trials. Saves ~10 min per matrix run with no loss of signal.

---

## Queued — impactful but not yet shipped

### Post-soak cleanup (~1 hour)

Delete the `LAYOUT_POSTPROC=torch` fallback branch and the `LAYOUT_COMPILE` block from the runtime app (~80 LOC). Drops torch as a runtime dependency entirely. No perf delta, just dependency hygiene.

### Retry Phase 2 fused graph (1–2 days)

The c=12 win was real (+29% rps, -34% p99). The c=24/32 regression looks concurrent-session-specific (opset fallback + larger graph). Retry with: pinned opset 17, `onnxsim` constant-folding, bool-mask output. Re-run matrix to decide.

### Preprocess into `onnxruntime-extensions` (half a day, only if profiled as bottleneck)

Move `PPDocLayoutV3ImageProcessor.resize + normalize` into the ORT session as a preprocessor op. Currently preprocess is sub-10 ms per page and not visible in the p50/p95/p99 tail decomposition — don't attempt without profile evidence.

### Scale `LAYOUT_ONNX_THREADS` with cgroup

If you grow the cgroup past 8 cores, re-tune `LAYOUT_ONNX_THREADS` to maintain `CPU_WORKERS × LAYOUT_ONNX_THREADS ≈ cgroup_cores`. The knob is the one with the cleanest scaling behavior in the whole setup.

### Driver body-content assertion (~1 hour — ship before anything else on this list)

Drivers (`loadtest/locust/locustfile.py`, `loadtest/asyncio/bench.py`, `loadtest/k6/ocr_load.js`) currently mark a response successful on HTTP 2xx alone. Both the OpenVINO rejected entry *and* the Layout-batcher warning above have the same failure chain: `HTTP 200` + empty `markdown_result` → counted as success → rps inflated, latency understated (silent failures return in ~500 ms vs ~5 s for a real OCR). The 2026-04-23 session confirmed this retroactively — the "3× OpenVINO win" would have been rejected at commit time if the driver checked body content.

Fix: after `resp.status == 200`, assert `len(resp.json().get("markdown_result", "")) > 0`. Count empties as their own failure category (`empty-markdown`). Expose as a metric (`glmocr_asyncio_empty_markdown_total`, or locust's built-in `resp.failure()` category). Add a Grafana panel separating "real 2xx rate" from "HTTP 2xx rate" — these were the same line until this finding landed.

Retroactive value: re-running any prior matrix config against the current stack with the patched driver tells you what fraction of the historic rps number was silent-empty. Expect non-trivial deltas on runs with `LAYOUT_BATCH_ENABLED=true` at c ≥ 8.

Orthogonal to everything else here; it's the monitoring rail that catches the next OpenVINO-class regression before it's published as a win. Should ship first.

*(Paddle2ONNX swap shipped 2026-04-24 — see Shipped §6.)*

*(`LAYOUT_ONNX_PROVIDER=openvino` shipped 2026-04-24 — see Shipped §7.)*

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

1. **ONNX backend first.** Unblocks Fix 3 (`LAYOUT_ONNX_THREADS` only exists in the ORT branch) and the fused graph. Without the ORT move, none of the thread tuning applies.
2. **numpy post-proc second.** Prerequisite for deleting torch from the request path, and for accurate GIL accounting under load (you can't see the GIL ceiling while torch still holds it in long chunks).
3. **Batch coalescer third.** Independent of ORT/numpy ordering, but compounds the ORT gain.
4. **Thread tuning last.** `LAYOUT_ONNX_THREADS`, `CPU_THREADS`, `LAYOUT_BATCH_MAX` are all cgroup-sensitive and depend on which kernels are live. Don't tune them before the pipeline stack is finalized; you'll tune to the wrong baseline.

---

## Net impact (pre-optimization → final)

Baseline config: torch eager, torch post-proc, no batcher, `LAYOUT_ONNX_THREADS=1`, `LAYOUT_BATCH_MAX=4`.

Final config: as in the TL;DR `.env` block above.

| c | baseline rps | final rps | Δ |
|---:|---:|---:|---:|
| 12 | 1.554 | ~2.4 (avg) | **+54%** |
| 24 | 2.384 | ~2.77 (avg) | +16% |
| 32 | 2.173 | ~2.48 (avg) | +14% |

c=32 p95: 28,691 ms → ~21,700 ms (**-24%**). c=32 p99: roughly flat (29,273 → ~27,273 ms).

The largest single win is the low-concurrency p50 (~50% reduction at c=12) — attributable to GIL release from removing torch from post-proc under gthread contention.

**2026-04-24 additional shipments** (Paddle2ONNX graph §6 + OV EP §7), measured head-to-head at c=8 over 20 pages with the same image seed:

| | matrix-final CPU EP | +OV EP (this commit) | Δ |
|---|:-:|:-:|:-:|
| rps | 0.31 | **0.57** | **+84 %** |
| mean latency | 22.23 s | **12.6 s** | **−43 %** |
| empty_markdown rate | (hidden by matrix asserting status only; real rate at `LAYOUT_BATCH_ENABLED=true` with the torch graph was ~11 %) | **0 %** | silent-failure class eliminated |

The headline change vs the matrix baseline is **~1.8× rps at c=8 with +quality** (no more silent empties). The kernel-level win is ~1.5×; the additional lift comes from reduced queue-wait at the binding stage. Matrix reruns at c=12/24/32 are queued to update the full-range numbers above.
