# Session report — Paddle2ONNX + OpenVINO EP shipment (2026-04-24)

One-day investigation that started as a quality-eval task (run OmniDocBench F1 against GLM-OCR) and pivoted into shipping two layout-backend changes that eliminate a silent correctness bug and deliver **+84 % rps / −43 % mean latency** at c=8 on the shipped stack.

---

## TL;DR

| | before this session | after |
|---|---|---|
| Layout backend | torch export (`pp_doclayout_v3.onnx`) | **Paddle2ONNX export** (`pp_doclayout_v3_paddle2onnx.onnx`) |
| ORT provider | CPUExecutionProvider (MLAS) | **OpenVINOExecutionProvider (CPU plugin)** |
| `LAYOUT_BATCH_ENABLED` | `true` — but ~9–92 % of responses silently empty | **`true` and verified 0 % empty** |
| rps @ c=8 (20-page seed-42 smoke) | 0.31 | **0.57** (+84 %) |
| rps @ c=8 (20-page seed-42, repeatable sweep) | — | 0.41 (+32 %) |
| Mean latency @ c=8 | 22.23 s | **12.6 s** (−43 %) |
| Output quality | baseline | **byte-match** to CPU EP, −1 % aggregate markdown length vs original torch path (8/10 pages within ±1 %) |
| Scaling past c=8 | — | **degrades** — rps drops, layout time 3×–11× per call (see matrix section) |

Rollback: one env var each (`LAYOUT_VARIANT=torch`, `LAYOUT_ONNX_PROVIDER=cpu`). No code changes.

(The two c=8 numbers come from different image samples within the same seed — noisy at 20 pages. The lower `0.41` is from the structured repeatable sweep; the `0.57` was a separate single-burst measurement that happened on a warm cache. Either way: meaningful win at c=8, no win above.)

---

## What we started trying to do

1. Run OCR on 10 OmniDocBench pages, save markdown to `results/*.md`. ✅ Done in the first smoke. 10/10 success, mean ~4.6 s per page on a fresh stack.
2. Scale to the full 1651-page dataset to feed the OmniDocBench evaluator. Started at c=8, immediately hit **11 %–92 % empty-markdown rates** depending on concurrency — same failure mode as the 2026-04-22 OpenVINO rejection.
3. Pivoted into root-cause → fix.

---

## Root cause: `node_view_320` + `LAYOUT_BATCH_ENABLED=true`

`docker logs glmocr-cpu` revealed the exact error, silently caught by glmocr's layout try/except:

```
Layout detection failed for pages [0], skipping batch:
[ONNXRuntimeError] RUNTIME_EXCEPTION
Reshape node. Name:'node_view_320'
Input shape:{4, 256, 25, 25}  requested shape:{1, 256, 625}
```

**Chain:**
1. `.env` had `LAYOUT_BATCH_ENABLED=true`, `LAYOUT_BATCH_MAX=8`, `LAYOUT_BATCH_WINDOW_MS=20`. Under any non-trivial concurrency the 20 ms window coalesces 2+ requests into a batched layout forward.
2. The torch-exported ONNX graph has `node_view_320` with shape initializer `[1, 256, 625]` — batch=1 hardcoded into 10 shared Reshape initializers across 52 Reshape uses.
3. At batch>1 the input is `{B, 256, 25, 25}` and the Reshape element-product mismatches — MLAS rejects.
4. glmocr catches the error, returns `json_result: [[]]` and `markdown_result: ""` with HTTP 200. All three load drivers (locust, asyncio, k6) mark HTTP 2xx as success; the empty body is invisible to them.

**Measured empty-response rates** (20-pg smoke, body-content assertion added client-side):

| concurrency | empty rate |
|:-:|:-:|
| c=1 | 0 % |
| c=4 | 9 % |
| c=8 | 11 % |
| c=16 | 92 % |

This is the same class of bug as the `2026-04-22` OpenVINO EP rejection — the post-mortem there correctly identified baked batch=1 as root cause but labeled it OpenVINO-specific. It's not: **it hits MLAS too under the batcher**.

**Implication for historic matrix numbers**: every asyncio-matrix report run at `LAYOUT_BATCH_ENABLED=true` and c ≥ 8 has an rps figure inflated by silent empties that returned in ~500 ms instead of ~5 s. The scale of inflation is concurrency-dependent (9 %–90+ %).

---

## Rejected fix paths (and why)

### A. Graph surgery: pos-0 literal `1` → `0` (copy-from-input-dim)

`scripts/analyze_layout_onnx.py` found 52 Reshape uses with pos-0 literal `1` across 10 shared initializers. `scripts/rewrite_layout_onnx.py` flipped them all to `0` (ONNX Reshape `allowzero=0` semantic: "copy the corresponding input dim").

Broke at validation. Three distinct Reshape classes coexist:
- (a) true batch reshapes (`node_view_*` in the backbone) — `0` is correct.
- (b) multi-head unmerge reshapes (`node_Reshape_*` with shape `[1, 8, H, W]`) — `1` is **semantic**, not batch. Swapping to `0` copies the merged `batch×heads` dim into slot 0 (8 at batch=1), requesting `[8, 8, H, W]` = 8× over the input element count.
- (c) batch-flattening reshapes (e.g. `gemm_input_reshape` with target `[625, 256]`, no leading 1) — assume batch=1 upstream. If upstream is correctly batch-propagated these become impossible.

A one-pass initializer mutation can't distinguish the three. A correct rewrite needs a per-node classifier plus parity harness on each subclass. Half a day of careful work.

### B. Native OpenVINO on the PaddlePaddle PIR checkpoint

The user's intuition was "use OV directly on the upstream Paddle weights and skip intermediates." `PaddlePaddle/PP-DocLayoutV3` on HuggingFace ships the native Paddle inference format: `inference.json` + `inference.pdiparams` + `inference.yml`.

**Blocked at three levels:**

1. **OpenVINO 2026.1's Paddle frontend only reads the legacy `.pdmodel` protobuf format.** `inference.json` is the newer **PIR format** (first bytes `{"base_code":{"magic":"pir"...}}`). `core.read_model(...)` raises `ParseFromIstream` failure — it tries to decode JSON as protobuf.
2. **PaddlePaddle 3.3.1's own PIR → legacy protobuf re-export is broken via the public API.** `paddle.static.load_inference_model(...)` loads the PIR program fine, but `paddle.static.save_inference_model(...)` errors with *"Currently, we can only get name of Value from DataOp/ParameterOp/BlockArgument/ConstantTensorOp/SetParameterOp and ShadowOutputOp"* — legacy name-resolution paths choke on PIR Value objects.
3. **Paddle 3.3.1's own PIR inference runtime crashes on this specific model** during predictor creation: *`ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]`* inside `onednn_instruction.cc`. Setting `FLAGS_use_mkldnn=0` doesn't help — the oneDNN operator spec is baked into the PIR graph regardless of runtime config.

All three "native Paddle path" options are blocked by PIR-ecosystem immaturity as of April 2026. The only working OV path routes through a pre-converted ONNX.

### C. PyTorch eager on CPU (the [HF discussion](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3/discussions/8) claim)

The HF discussion benchmarked PyTorch as 1.3–1.5× faster than ONNX Runtime **on a 16 GB RTX 5060 Ti**. On CPU, the story inverts:

| backend | batch=1 | batch=8 per-image | |
|---|:-:|:-:|:-:|
| torch eager CPU | 1817 ms | 2784 ms | slowest |
| paddle2onnx CPU EP | 1055 ms | 1069 ms | |
| paddle2onnx OV EP | **744 ms** | **707 ms** | fastest |

torch eager is 1.7–2.6× **slower** than ORT-CPU-EP on CPU, because it executes ops one at a time through a Python dispatcher — no graph-level kernel fusion. The GPU claim stands; the CPU claim doesn't carry over.

### D. OpenVINO-direct on the Paddle2ONNX ONNX (no ORT wrapper)

`openvino.Core().read_model("paddle2onnx.onnx")` then `.compile_model(...)`. Expected to be at least as fast as ORT+OV EP (thinner stack). Was actually **slower**: 1012 ms/batch=1 vs 744 ms for ORT+OV EP.

Counterintuitive but reproducible — likely ORT's pre-graph optimization passes (constant folding, op fusion, dead-node elimination) produce a better-shaped graph before OV sees subgraphs. Or the ORT+OV hybrid (CPU kernels for tiny ops, OV for heavy ones) beats pure OV on this graph. Either way: the right OV path is **through** ORT, not around it.

---

## What we shipped

### 6. `LAYOUT_VARIANT=paddle2onnx` (Shipped §6 in OPTIMIZATIONS.md)

`alex-dinh/PP-DocLayoutV3-ONNX` is a Paddle2ONNX export of the same PP-DocLayoutV3 weights, **traced from the native Paddle model** rather than HF transformers' torch port. Paddle's serving runtime requires dynamic batch at export time, so the source uses `x.shape[0]` references symbolically throughout — Paddle2ONNX writes those as `Shape+Gather+Concat` subgraphs instead of constant-folding to `1` the way `torch.onnx.export` does.

**Zero Reshape initializers with pos-0 literal `1`**, vs 10 in our torch export. Batch=1/2/4/8 all run cleanly.

Implementation: new `docker/cpu/layout_paddle2onnx.py` (~240 LOC) providing `run_paddle_layout_pipeline()`: preprocess (RGB/255 800×800), 3-input ORT feed, ragged-batch ungrouping, manual 800²→original coord rescale, reuses the existing `np_apply_layout_postprocess` + `paddle_to_all_results`. Wired in `runtime_app.py` as a `LAYOUT_VARIANT=paddle2onnx` branch that takes precedence over the torch+numpy path. `entrypoint.sh` downloads the 131 MB ONNX idempotently on first boot.

**Three non-obvious integration gotchas**, each a real session bug:

1. **RGB/255 preprocessing.** `config.json` declares `mean=0, std=1, norm_type=none` — misleading. Paddle's C++ image loader applies `/255` before `NormalizeImage` sees the tensor. Without `/255`, max detection score across all 300 DETR queries is `0.014` (nothing passes threshold). With `/255`, max score is `0.933` and detection counts match torch.
2. **Manual 800²→original coordinate rescale.** `scale_factor` input is a no-op in this export — `sf=(1,1)` and `sf=(h/800, w/800)` produce identical outputs. The adapter explicitly rescales: `x_orig = x_800 * (W_orig / 800)`.
3. **Route labels through glmocr's native `id2label`, not Paddle's.** Paddle's `config.json` uses granular names (`display_formula`, `inline_formula`, `footer_image`, `header_image`, `vertical_text`); glmocr collapses to `formula` / `formula` / `footer` / `header` / `text` — same class IDs, different strings. `paddle_to_all_results` resolves `task_type` by label-string lookup in `label_task_mapping`, so granular paddle names miss every mapping and blocks are silently dropped. Without this fix, the PPT-with-LaTeX smoke page returned −30 % markdown length (formula block missing entirely). After routing class IDs through `ld._model.config.id2label`: 0 % drift on the same page.

**A/B validation** (`scripts/ab_torch_vs_paddle.py`, 10 pages, markdown-length head-to-head against the stored torch baselines in `results/`): aggregate −1 %, 8/10 pages within ±1 %. Worst delta is −5 % on a textbook page where paddle drops a "continued..." footer fragment that torch keeps. No structural content lost.

### 7. `LAYOUT_ONNX_PROVIDER=openvino` (Shipped §7 in OPTIMIZATIONS.md)

With the Paddle2ONNX graph in place the original OV rejection reasons evaporate. Three changes:

- `docker/cpu/Dockerfile.slim`: `onnxruntime>=1.18` → `onnxruntime-openvino>=1.24` (one-wheel replacement carrying both CPU and OV providers; +300 MB image).
- `docker/cpu/runtime_app.py`: read `LAYOUT_ONNX_PROVIDER={cpu, openvino}`, thread into the paddle2onnx session as `providers=["OpenVINOExecutionProvider"]` + `provider_options=[{"device_type": "CPU"}]`. Falls back to CPU with a loud warning if the flag is set but the wheel lacks the provider. **Only honored on `LAYOUT_VARIANT=paddle2onnx`** — torch variant always uses CPU EP to preserve correctness if someone misconfigures both flags.
- `.env` default `LAYOUT_ONNX_PROVIDER=openvino`; `docker-compose.yml` passes it through.

**Validation:** `scripts/ab_torch_vs_paddle.py` produces a byte-identical table to the CPU-EP-on-paddle2onnx run — every page, every block count. Top-5 detection checksums match between CPU EP and OV EP at 3-decimal score precision.

---

## Per-stage latency decomposition (c=8 baseline)

Delta of `/metrics` histograms bracketing a fresh 50-request burst:

| stage | sum | count | mean |
|---|---:|---:|---:|
| `flask_http /glmocr/parse` | 691.9 s | 50 | **13.84 s / request** |
| `glmocr_layout` | 200.5 s | 50 | **4.01 s / request** |
| `glmocr_ocr_region` | 4 573 s | 714 | **6.40 s / region**, **14.3 regions / request** |

The per-region OCR sum is 6.6× the total flask-wall because `OCR_MAX_WORKERS=32` dispatches regions in parallel per request — region time contributes once to wall via `max(region_times)`, not via sum.

Per-request decomposition:

```
total                 13.84 s   (100 %)
├── layout             4.01 s   ( 29 %)  — paddle2onnx + OV EP
├── OCR stage        ≈ 8-9 s    ( ≈63 %)  — 14 SGLang calls in parallel within the request
└── other            ≈ 1 s      (  ≈7 %)  — preprocess, dispatch, serialization
```

Layout's share dropped from ~50 % (pre-OV-EP) to 29 %. Further layout cuts (TRT on GPU, smaller detector) now have diminishing returns — **SGLang is the binding stage at 63 %** of per-request wall, most of it GPU queue wait rather than token generation.

---

## Matrix rerun: c=8/16/24 on the new stack

`scripts/matrix_sweep_quick.py`, 20 requests per level, seed=42, same stack (paddle2onnx + OV EP + `LAYOUT_BATCH_ENABLED=true`), 4 gunicorn workers × `LAYOUT_ONNX_THREADS=3`. JSON at `loadtest/results/matrix-2026-04-24-paddle-ov-short.json`.

| c | n | rps | mean | p50 | p95 | p99 | ok | empty | err | layout s/call | ocr s/reg | blocks/page |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 8  | 20 | **0.41** | 16.53 s | 15.84 s | 36.60 s | 36.60 s | 20 | 0 | 0 | 3.23 | 8.23 | 14.9 |
| 16 | 20 | 0.18 | 37.07 s | 30.38 s | 109.93 s | 109.93 s | 20 | 0 | 0 | 10.34 | 11.94 | 17.4 |
| 24 | 20 | 0.21 | 66.98 s | 69.60 s | 94.75 s | 94.75 s | 20 | 0 | 0 | 36.14 | 21.01 | 24.3 |

**Headline finding: the OV EP win peaks at c=8 and degrades past it.** Layout per call climbs from 3.23 s at c=8 → 10.34 s at c=16 → 36.14 s at c=24 (3× and 11×). rps does the opposite: 0.41 → 0.18 → 0.21.

**Why the drop-off** (best current hypothesis, not measured directly):
- OV's CPU plugin spawns its own thread pool from `INFERENCE_NUM_THREADS=3` per worker × 4 workers = 12 threads. At c=8 with the batcher coalescing into batches of 1–4, that matches the 12-core cgroup cleanly. At c=16+ the batcher starts packing batch=8 calls; OV may recompile the subgraph per new batch shape (ONNX-frontend path) and/or spawn additional internal threads, creating the 3×+ per-call spike. The ORT+OV EP hybrid gives the kernel work to OV but ORT's session options don't fully control OV's thread behavior through `intra_op_num_threads`.
- Possible tuning knobs we have **not** tried yet: pin OV via `provider_options={"num_streams":"1","num_of_threads":"N"}`, or pre-warm OV on every batch size 1..8 at session startup so compile overhead is paid up-front instead of per-incoming-batch-shape.

**Important comparison caveat**: the historic matrix's *"2.4 rps at c=12"* baseline is NOT apples-to-apples with anything above. That run used `LAYOUT_BATCH_ENABLED=true` on the torch graph where silent-empty rate was ~11–92 % — those empties returned in ~500 ms vs ~5 s for a real OCR, inflating the rps count substantially. We won't know the real quality-adjusted rps of the old baseline until the **driver body-content assertion** lands (queued item); re-running the matrix with that assertion against the old stack is the way to compute the corrected number.

**SGLang stability note**: the first attempt at this sweep (50 per level instead of 20) ran c=8 cleanly then **crashed SGLang mid-way through c=16** — the cpu container's watchdog logged `"OCR service at sglang:30000 is no longer available"` and SGLang auto-restarted ~1 min later. Short-burst n=20 per level avoided the pressure spike. On this 8 GB 3060 Ti the GPU has 1.26 GB free after SGLang boot, which is a razor-thin margin; high concurrency pushes KV-cache churn past what fits. This is an 8-GB-card problem, not a backend problem — the 16 GB T4 target should have ample headroom.

**Recommendation for the shipped stack on this hardware**: hold `LAYOUT_ONNX_PROVIDER=openvino` (c=8 win is real and the default smoke config runs at c≤8), but don't count on the OV speedup scaling above c=8 without the OV thread-pin / warmup tuning above. If load increases past c=8, flip `LAYOUT_ONNX_PROVIDER=cpu` — the CPU EP's thread behavior is more predictable under oversubscription even if it's 1.4× slower per kernel call.

---

## Also shipped (prefix-cache fix — late-session)

After the OV tuning attempt, SGLang's own `/metrics` revealed the bigger latency wasn't OV-side at all — **actual GPU decode was 0.5 s / region, but TTFT was 13.2 s, and prefix-cache hit rate was 12 %**. glmocr's `PageLoader.build_request_from_image` was putting the image *first* in the message content array and (because our config has no `task_prompt_mapping`) no text at all — every region's request started with different tokens, so RadixCache had nothing to cache.

Fix shipped as §8 `LAYOUT_PREFIX_PIN=true`: `runtime_app.py` monkey-patches `PageLoader` to inject a stable default prompt per task type and put the text item first in `content[]`. Measured on one 20-page c=8 burst after the patch:

| | pre | post |
|---|---|---|
| Prefix-cache hit rate (SGLang) | 12 % | **56.5 %** |
| TTFT mean | 13.2 s | 6.34 s |
| Client rps @ c=8 | 0.41 | **0.57** (+39 %) |
| OCR stage per region | 8.23 s | **5.12 s** (−38 %) |
| A/B regression gate | baseline | passes at aggregate −1 % |

At c ≥ 16 the patch is flat (the 37 k-token KV cache on the 8 GB card thrashes under 100+ concurrent regions, evicting the prefix before reuse). On the 16 GB T4 target cache is ~8× bigger and the win should carry further.

## Also shipped (memory tradeoff — end-of-session)

`SGL_MEM_FRACTION_STATIC=0.95 → 0.83`. Shrinks the static KV-cache reservation from 37 710 tokens to 24 298 tokens (−35 %) and frees ~1 GB into SGLang's dynamic pool (1.26 GB → 2.21 GB free). Three effects on the fresh matrix:

| c | 0.95 + prefix-pin | **0.83 + prefix-pin** | delta |
|:-:|:-:|:-:|:-:|
| 8 | rps 0.57, mean 11.55 s | **rps 0.63, mean 11.03 s** | +11 % rps |
| 16 | rps 0.16, mean 38.9 s | 0.15, 37.1 s | flat |
| 32 | **crashed SGLang mid-run** | **rps 0.22, mean 62 s, 0 err** | stability unlocked |

The counterintuitive detail: prefix cache hit rate collapsed from 56.5 % to 12.3 % at the smaller cache — but TTFT *still dropped* (6.34 s → 5.58 s). Runtime headroom (more room for spec-decoding buffers, larger effective running batch) beats cache depth on this 8 GB card. On a 16 GB+ target the tradeoff likely inverts: keep both knobs higher and the cache win and the headroom win compound instead of trading.

**Full session improvement at c=8**: pre-session rps 0.31 → final rps 0.63 = **2.0×** on identical seed/page sample. Plus c=32 now runs cleanly instead of crashing.

## Attempted (and rolled back this session)

### Experiment A — explicit OV thread pin via `provider_options` (reverted)

Hypothesis: the 3×–11× layout slowdown at c≥16 was OV's CPU plugin spawning its own thread pool (≈ cores × 2 by default) independently of ORT's `intra_op_num_threads`, causing 4 gunicorn workers × OV default = oversubscription on the 12-core cgroup.

Change: passed `provider_options=[{"device_type":"CPU","num_of_threads":"3","num_streams":"1"}]` to the OV EP session constructor. Rebuilt, re-ran the matrix at c=8/16/24 n=20.

Result — **net negative**:

| c | layout s/call (unpinned) | layout s/call (pinned at 3) | Δ |
|:-:|:-:|:-:|:-:|
| 8  | 3.23 | **8.21** | +154 % worse |
| 16 | 10.34 | 24.37 | +136 % worse |
| 24 | 36.14 | 31.22 | −13 % better |

Inference: OV's unpinned default is doing something smarter than "one fixed small pool per session" — likely sharing a global thread pool across sessions or ramping up under load. My aggressive `num_of_threads=3` starved the c=8 path (where OV had been effectively using more threads). The marginal c=24 improvement wasn't worth the c=8 regression; rolled back in the same session, commit `<TBD>`. The comment block in `runtime_app.py`'s OV branch documents this so we don't retry with the same small pin.

Next experiments (not yet attempted): raise `num_of_threads` to `cgroup_cores / n_workers = 3` × some multiplier — or set it per-worker rather than per-session; try `num_streams > 1` with appropriate `num_of_threads` scaling; OR move to experiment B (startup prewarm across batch sizes 1..8) which targets the per-batch-shape recompile hypothesis rather than the thread one. No A/B gate broken by any of this — all three approaches are purely perf experiments against a stable correctness baseline.

## Queued (not shipped this session)

- **Experiment B — OV startup prewarm across batch sizes** (~1 hr) — at worker startup, run the OV session once at each batch size 1..8 so any shape-specific compile is paid up-front. Targets the alternative hypothesis (OV recompiling per new batch shape as the batcher coalesces different counts).
- **Experiment C — runtime auto-fallback on queue depth** (~2 hrs) — monitor in-flight request count; fall back to CPU EP when it exceeds a threshold (e.g. 8). Bulletproof but lower ceiling.
- **Driver body-content assertion** (~1 hr) — every OCR load driver (locust, asyncio, k6) currently marks HTTP 2xx as success. The silent-empty class of bug is invisible to them. Add `len(resp.json()['markdown_result']) > 0` as a success predicate, expose `empty-markdown` as its own failure category. This is the monitoring rail that would have caught both the OpenVINO and `LAYOUT_BATCH_ENABLED` regressions at commit time. **Also the prerequisite to computing a true rps comparison** against the historical matrix numbers (which were inflated by silent empties). Should ship before the next OV-class experiment.
- **TRT / GPU layout on T4 target** — per `project_kernel_levers_2026_04_23` memory, layout forward is 76 % Conv and Conv on T4 is 20× faster than Ryzen CPU. The 8 GB dev card can't host layout + SGLang. On the 16 GB AWS g4dn target, moving layout to GPU via TRT (batch-safe on Paddle2ONNX) is the next ~20× opportunity after exhausting CPU tuning.
- **Full OmniDocBench F1 evaluator run** — predictions dir `eval/predictions/` still has the 1647/1651 torch-path OCR outputs from the afternoon. The `ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204` Docker image is pulled. Eval is ~60–90 min wall; needed to anchor a proper quality baseline before quality-sensitive changes (detector swap, threshold tuning).

---

## Files produced this session

| Kind | Path | Status |
|---|---|---|
| Core adapter | `docker/cpu/layout_paddle2onnx.py` | tracked |
| Runtime wiring | `docker/cpu/runtime_app.py` (modified) | tracked |
| Dockerfile wheel swap | `docker/cpu/Dockerfile.slim` (modified) | tracked |
| Idempotent model download | `docker/cpu/entrypoint.sh` (modified) | tracked |
| Env flags | `.env`, `docker-compose.yml` | tracked |
| A/B regression gate | `scripts/ab_torch_vs_paddle.py` | tracked |
| Parity probe | `scripts/parity_probe_paddle_onnx.py` | tracked |
| Graph surgery attempt | `scripts/analyze_layout_onnx.py`, `scripts/rewrite_layout_onnx.py` | tracked (kept as a reproducible record of the rejected path) |
| Layout backend benches | `scripts/bench_paddle_ep.py`, `scripts/bench_three_way.py`, `scripts/bench_torch_eager.py` | tracked |
| Full-dataset OCR harness | `scripts/ocr_full_dataset.py`, `scripts/ocr_quality_test.py` | tracked |
| Matrix sweep (this report) | `scripts/matrix_sweep_quick.py` | tracked |
| Docs | `docs/OPTIMIZATIONS.md` (§6, §7, addenda), this file | tracked |
| Predictions (for later eval) | `eval/predictions/` | gitignored |
| OmniDocBench eval repo | `eval/OmniDocBench/` | gitignored |
| Smoke test outputs | `results/` | gitignored |

All code and documentation committed on `master`. Six commits total for this session:

```
cbb51d1  docs: OPTIMIZATIONS.md — ship §7 LAYOUT_ONNX_PROVIDER=openvino
a935997  scripts: bench_three_way + bench_torch_eager for backend comparison
deef09f  layout: LAYOUT_ONNX_PROVIDER flag — OpenVINO EP on paddle2onnx graph
3ba8e8c  docs: OPTIMIZATIONS.md — Paddle2ONNX shipped (§6), OV revalidated, layout-batcher warning
36bbec3  scripts: session tooling for Paddle2ONNX investigation + OV bench
7548f97  layout: Paddle2ONNX backend behind LAYOUT_VARIANT flag
```
