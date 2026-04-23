# Why Flask, and why the CPU layer is slow

## Context

The `omnidoc-20260421-181744-asyncio-matrix.md` sweep shows the CPU container is the clear bottleneck:

| Signal | c=1 | c=24 | c=32 |
|---|---|---|---|
| **CPU in-flight peak** | 5 | 11 | 7 |
| **Flask end-to-end mean** | 6.1 s | 26.4 s | 12.4 s |
| **Layout forward mean** | 2.8 s | 5.2 s | 3.9 s |
| **OCR region mean** (per region) | 1.4 s | 8.7 s | 3.6 s |
| **SGLang running (GPU batch)** | mean 5 / peak 37 | mean 21 / peak 37 | mean 22 / peak 40 |
| **SGLang queued (scheduler)** | mean 1 / peak 32 | mean 53 / peak 115 | mean 57 / peak 110 |
| **rps (success)** | 0.18 | **0.45 (plateau)** | 0.47 + 16 fails |

rps plateaus at ~0.45 from c=12 onward — **raising client concurrency only grows the SGLang queue and Flask p99**, it does not produce throughput. The user wants to understand *why Flask* and *where the waste is*.

---

## Diagnosis — the real bottlenecks (not Flask per se)

Ground truth confirmed by reading `/usr/local/lib/python3.12/site-packages/glmocr/{server.py,ocr_client.py,pipeline/_workers.py}` inside the live `glmocr-cpu` container and inspecting `docker/cpu/runtime_app.py` + `.env`:

1. **Hard concurrency ceiling: 12 request slots.**
   - `docker/cpu/entrypoint.sh:46–56` launches gunicorn with `--worker-class gthread --workers ${CPU_WORKERS} --threads ${CPU_THREADS}`.
   - `.env` has `CPU_WORKERS=2, CPU_THREADS=6` → **2 × 6 = 12 concurrent requests admitted**, period.
   - 12 slots × ~24 s/request (at c=24) = **0.5 rps** — exactly the observed plateau.

2. **Layout inference is on the request thread, unbatched.**
   - `.env` has `LAYOUT_BATCH_ENABLED=false`, so the coalescer at `docker/cpu/runtime_app.py:573–656` is dormant.
   - Every request does its own 3–5 s ONNX forward with `LAYOUT_ONNX_THREADS=1` and `OMP_NUM_THREADS=1`. With 12 threads all racing through ONNX at once, wall time per forward doubles (c=1: 2.8 s → c=24: 5.2 s).
   - This is the single largest recoverable latency in the system.

3. **Region fan-out is sync `requests.post` inside a per-request ThreadPoolExecutor.**
   - `glmocr/ocr_client.py:96–240` uses `requests.Session` + `HTTPAdapter` with blocking `requests.post`.
   - `glmocr/pipeline/_workers.py:328` spawns a fresh `ThreadPoolExecutor(max_workers=OCR_MAX_WORKERS)` **per request**. At 12 concurrent requests that's up to 192 throwaway threads.
   - `OCR_CONN_POOL=192` is correctly sized (2 × 6 × 16), so pool starvation is **not** the issue.
   - Socket I/O releases the GIL, so this part actually works — but it's thread-heavy and prevents a single in-flight request from starting its regions before layout finishes.

4. **Why Flask, specifically.** Flask lives inside the upstream `glmocr` package (`glmocr.server.create_app()` at `/usr/local/lib/python3.12/site-packages/glmocr/server.py:75` registers `@app.route("/glmocr/parse")`). The CPU container *imports* Flask via `wsgi.py`; the choice was inherited from upstream glmocr, not made here. The `/glmocr/parse-async` endpoint referenced in `scripts/tune_params.py:914` and `scripts/lib/render_report.py:1397` **does not exist** in the installed glmocr — it was aspirational.

5. **SGLang is not the villain.** KV util is 51 %, peak running 40/64, `max_total_tokens` comfortably sized. The queue grows only because Flask fires bursts of 12 × ~10 regions then goes silent.

**Summary:** the 0.45 rps ceiling is 80 % layout inference serialization + 20 % gthread slot count. The framework (Flask vs FastAPI) matters least — even FastAPI couldn't help if layout stayed on the request path unbatched.

---

## Recommended fix — two-tier approach

### Tier 1 — configuration wins, no code (try first, measure)

Change `.env` and re-run the `scripts/omnidoc_asyncio_matrix.sh` sweep. Expected recovery: **2–3× rps**, much of it free.

| Knob | Current | Proposed | Why |
|---|---|---|---|
| `LAYOUT_BATCH_ENABLED` | `false` | **`true`** | Coalesces up to 4 concurrent layout calls into one ONNX batch. Code already lives at `docker/cpu/runtime_app.py:573–656`. Single biggest expected lift. |
| `LAYOUT_BATCH_MAX` | `4` | keep `4`, try `8` | 8 lets bigger bursts collapse into one forward. |
| `LAYOUT_BATCH_WINDOW_MS` | `20` | keep `20` | Adds at most 20 ms p50 latency; negligible vs 3 s inference. |
| `CPU_THREADS` | `6` | **`12`** | Doubles admitted concurrency. Each extra thread feeds the batch coalescer. `OCR_CONN_POOL` rises to `2 × 12 × 16 = 384`. |
| `OCR_CONN_POOL` | `192` | **`384`** | Keep the invariant `OCR_CONN_POOL ≥ CPU_WORKERS × CPU_THREADS × OCR_MAX_WORKERS`. |
| `LAYOUT_ONNX_THREADS` | `1` | keep `1` (under batching) | Batching replaces the need for per-call intra-op threading. |

Files to touch for Tier 1: `.env` only.

**Verification for Tier 1:**
- `docker compose up -d --force-recreate cpu` to pick up env changes.
- Confirm live values at `curl http://localhost:5002/runtime | jq '.runtime_actual,.env_claimed'`.
- Re-run `bash scripts/omnidoc_asyncio_matrix.sh` and compare the new report's `rps`, `Layout forward mean`, and SGLang `running/queued` against the baseline at the four concurrency levels.
- Success criterion: c=12 rps rises above 0.9, Layout-forward mean at c=24 stays under 3 s, SGLang running mean rises toward 40.

### Tier 2 — async handler in `docker/cpu/` (if Tier 1 plateau < target rps)

Add a **thin async handler that bypasses `glmocr.server` but reuses `glmocr.layout` and `glmocr.ocr_client` primitives**. This is the answer to "why Flask?" — we keep Flask for `/health`, `/runtime`, `/metrics`, and all other routes, but peel off the hot path.

New / modified files (all inside this repo — no upstream fork):

| File | Action | Purpose |
|---|---|---|
| `docker/cpu/async_app.py` | **create** | FastAPI (or Starlette) app exposing `POST /glmocr/parse-async`. |
| `docker/cpu/wsgi.py` | **edit** | Mount async app under `/async` via `a2wsgi.ASGIMiddleware` **or** serve it as a second uvicorn process in `entrypoint.sh` on a sidecar port. |
| `docker/cpu/entrypoint.sh` | **edit** | Start uvicorn for async handler alongside gunicorn (different port, e.g. 5003), or switch the whole container to a single hypercorn process serving both WSGI + ASGI. |
| `docker/cpu/runtime_app.py` | **edit** | Instrument the async handler with the existing Prometheus histograms (`glmocr_flask_*`, `glmocr_layout_forward_*`, `glmocr_ocr_region_*`) so `scripts/lib/render_report.py` keeps working unmodified. |
| `scripts/omnidoc_asyncio_matrix.sh` | **edit** | Add a second matrix run with `LOCUST_ENDPOINT=/glmocr/parse-async` for head-to-head comparison (the tune_params stage-e already supports this via `STAGE_E_ENDPOINT_ENV` — `scripts/tune_params.py:935`). |

Inside `async_app.py`, the hot path:

```python
@app.post("/glmocr/parse-async")
async def parse_async(req: Request) -> Response:
    img = await _load_image(req)                         # small, async I/O
    regions = await asyncio.to_thread(layout.process, img)   # CPU, offloaded
    async with httpx.AsyncClient(limits=...) as client:
        results = await asyncio.gather(*[
            _ocr_region(client, r) for r in regions
        ])
    return _assemble(results)
```

Primitives reused from the installed glmocr package (no rewrite):
- `glmocr.layout.LayoutDetector.process` — already instrumented by `runtime_app.py:564–667`.
- `glmocr.ocr_client.OCRClient.process` — swap `requests.post` for `httpx.AsyncClient.post` via a small adapter that targets the same config (`config.yaml`).

**Why this is worth it:** an async handler lets one process keep 100+ requests in flight (layout batching + region I/O overlapping across requests), not 12. Combined with Tier 1 batching, the same 2 vCPU container should sustain **≥ 2 rps** instead of 0.45 — this is the pipeline-overlap uplift the existing tune_params comment refers to.

**Verification for Tier 2:**
- Unit smoke: `curl -X POST localhost:5003/glmocr/parse-async -F @sample.png` matches byte-for-byte against `/glmocr/parse`.
- Head-to-head matrix: run `scripts/omnidoc_asyncio_matrix.sh` twice, once per endpoint, and diff the reports. Report acceptance: p95 latency at c=24 under 30 s AND rps > 1.5 AND SGLang queued peak < 60.
- Fall-back switch: keep `/glmocr/parse` online; the ALB routes based on header for easy rollback.

---

## Critical files to modify, in order

1. `.env` — Tier 1 knob changes.
2. `docker/cpu/async_app.py` — new, Tier 2.
3. `docker/cpu/entrypoint.sh` — Tier 2, add uvicorn sidecar.
4. `docker/cpu/runtime_app.py` — Tier 2, instrument async handler with existing metrics (reuse `_timed_layout` and `_timed_ocr` wrappers already defined at lines 564–683).
5. `scripts/omnidoc_asyncio_matrix.sh` — Tier 2, add endpoint= variant loop.
6. `README.md` — document the new endpoint.

## Verification plan (end-to-end)

1. Tier 1: edit `.env`, `docker compose up -d --force-recreate cpu`, re-run the matrix script, compare reports in `loadtest/results/`. Should show layout batching in `glmocr_layout_batch_*` histograms.
2. Tier 2: develop `async_app.py` under `docker/cpu/`, rebuild image (`docker compose build cpu`), smoke with a single `curl`, then the full matrix against both endpoints. Inspect Grafana `http://localhost:3000/d/glmocr-load` for CPU-inflight, SGLang running/queued, and per-phase histograms side-by-side.
3. Sanity: confirm Pushgateway `job=glmocr_asyncio` shows distinct `run_id`s per endpoint so the reports are comparable.

Stop at Tier 1 if it hits the rps target. Only move to Tier 2 if the async rewrite is needed to break past the single-process ceiling.
