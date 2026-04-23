---
name: DocLayout-YOLO is slower than PP-DocLayoutV3 on this CPU — 2026-04-23 probe
description: Feasibility probe invalidated the "YOLO is ~3× faster" assumption carried in project_kernel_levers_2026_04_23.md. At imgsz=1024 YOLO is 3× slower; at imgsz=640 it's 11% slower; only beats PP-DocLayoutV3 at imgsz=512 by 1.32×. Public benchmarks cited GPU numbers. Do not pursue the detector swap on Ryzen 5600X (Zen3, AVX2 only).
type: project
originSessionId: c8845dae-98e9-4b22-a7ae-02787ef2bf30
---
**Probe setup:** `docker/cpu/tests/probe_doclayout_yolo.py`, run inside the glmocr-cpu container. Loads `juliozhao/DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1024.pt` (30 MB, PyTorch eager), runs 3 warmup + 8 benchmark images from OmniDocBench at various imgsz. Package: `doclayout-yolo==0.0.4` (pip-installed in the running container).

**Measured CPU wall-time per forward pass:**

| imgsz | wall mean (ms) | p95 (ms) | vs PP-DocLayoutV3 @ 800 (~1,100 ms) |
|------:|---------------:|---------:|:---|
|  1024 |          3,285 |    4,140 | **3× SLOWER** |
|   640 |          1,239 |    1,433 | 11% slower |
|   512 |            836 |      ≈900 | 1.32× faster |

**Reality vs expectation:** the kernel-levers memory carried an assumption that YOLO is "~3× faster at comparable resolution per public benchmarks." That was GPU-only (RTX 3090-class benchmarks). On Ryzen 5600X CPU (Zen3, AVX2 but no AVX-512 / no VNNI), YOLOv10's backbone is not meaningfully faster than PP-DocLayoutV3's DETR at equivalent image sizes; at imgsz=1024 it's much slower because YOLOv10's Conv stack scales worse than DETR's transformer at high resolution without tensor-core acceleration.

**Even with optimistic ONNX conversion gains:** ~1.5-1.8× typical ORT speedup over torch eager would put imgsz=640 YOLO at ~700-800 ms — only marginally faster than current PP-DocLayoutV3 ONNX (~1,100 ms), and still worse than PP-DocLayoutV3 at imgsz=640 would be (ONNX scales similarly with resolution). The detector-architecture lever is not a structural win on this hardware.

**Detection quality at the "fast" sizes:** imgsz=512 produces detection counts similar to imgsz=1024 on OmniDocBench (13.6 vs 14.1 mean per image), but small-text classes (captions, footnotes, formulas) historically degrade faster at smaller input sizes. The same methodological concern that sank Phase 2a (LAYOUT_INPUT_SIZE=640) — OmniDocBench doesn't represent passport/ID/receipt distributions — applies to any DocLayout-YOLO config at imgsz<640.

**Integration cost if pursued anyway:**

Dropping what we'd lose + new code required:
- New inference path in `runtime_app.py` (~150 LoC)
- Rewrite of `layout_postprocess.py` — DETR query-head logic doesn't transfer to YOLO grid+anchor output (~150 LoC)
- Drop PP-DocLayoutV3 reading-order decoder (`order_logits` → `order_seq`); YOLO has no equivalent — either derive from box geometry or drop the `order` field (requires glmocr downstream audit)
- Drop mask-based polygon extraction (PP-DocLayoutV3 outputs per-detection masks; YOLO doesn't) — degrade `polygon_points` to rectangular `bbox_2d` everywhere
- Class taxonomy remap: PP-DocLayoutV3 ~17 classes vs DocLayout-YOLO 10 classes, no direct mapping (e.g. `aside_text`, `seal`, `chart`, `number`, `formula_number` missing from YOLO side; YOLO's `abandon` is a garbage-text class with no PP-DocLayoutV3 equivalent)
- Drop `LAYOUT_GRAPH=fused` code path (dead for YOLO)
- Rewrite parity tests (all current parity_* tests assume PP-DocLayoutV3 output shape)
- New heavy deps: ultralytics + albumentations + seaborn + matplotlib (~200 MB installed)

Total: 7-10 focused hours + open-ended corpus-parity work. Not worth it for the measured/projected speedup.

**Verdict:** do not pursue on this hardware. If the project moves to a different host (Intel Xeon with AVX-512 VNNI, or any GPU with enough VRAM spare for layout), re-run the probe — the finding is CPU-microarchitecture-specific, not a general YOLO rejection.

**Artifacts kept in tree:** `docker/cpu/tests/probe_doclayout_yolo.py` as the benchmark script. Re-run any time with `docker exec glmocr-cpu python /app/probe_doclayout_yolo.py` (with `PROBE_IMGSZ` env override).

**Cleanup:** the `doclayout-yolo` pip package is still installed in the running cpu container (ephemeral — gone on next image rebuild). Not added to Dockerfile.slim. No persistent footprint except the downloaded 30 MB weights in `~/.cache/huggingface/hub/models--juliozhao--DocLayout-YOLO-DocStructBench/`, harmless.
