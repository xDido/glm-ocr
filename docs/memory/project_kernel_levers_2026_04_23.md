---
name: Kernel-level layout optimization — 2026-04-23 outcomes
description: 4-phase experiment ended with no code changes shipped. Phase 2a (LAYOUT_INPUT_SIZE=640x640) measured +42-107% rps but was reverted on methodological grounds (parity validated on OmniDocBench only, unsafe to default for passport/ID/receipt distributions). Phase 2b (PTQ int8) rejected — PP-DocLayoutV3's DETR head too fragile for post-training static quantization. Phase 3 verified no-op. Only the Phase 3 probe script remains in the tree.
type: project
originSessionId: c8845dae-98e9-4b22-a7ae-02787ef2bf30
---
**Session end state (2026-04-23):** working tree clean vs HEAD (`968f73e`), only `docker/cpu/tests/probe_ort_fusion.py` is untracked. Baseline stack running at 800×800 fp32 (`pp_doclayout_v3.onnx`), SGL_SCHEDULE_POLICY=lpm.

**Why Phase 2a (LAYOUT_INPUT_SIZE=640×640) was reverted even though the matrix showed +42-107% rps:**

The parity gate was validated **only on OmniDocBench** (10 pages, seed=42 — academic papers, slides, reports). On that distribution: 1.8% count drop, mean IoU 0.934. But OmniDocBench is not representative of real production document distributions. Classes where 640×640 would degrade more than measured:

- Passport / national ID: MRZ zones and microtext — each character stroke covers fewer pixels, detector may miss the whole MRZ region → SGLang never sees the crop → **entire field drops from output silently**.
- Receipts (thermal print): low-contrast small text; the bench already shows text class losing 13% on 640×640 — likely worse here.
- Invoices with fine-print "terms" blocks at 7-8pt.
- Multilingual text with diacritics.

The failure mode is asymmetric and quiet: the detector runs at 640×640 but OCR reads full-resolution crops, so quality loss manifests as "whole regions missing from output" not "garbled text" — harder to notice in QA, more damaging to customers than a visible transcription error. This is structurally the same class of silent regression as the OpenVINO EP ship (HTTP 200 empty responses; see `project_openvino_ep.md`).

Measured rps deltas are recorded in `loadtest/results/comparison-20260423-kernel-levers.md` for reference if a future attempt runs a properly-scoped parity gate against a diverse corpus.

**Why Phase 2b (int8 static quantization) was rejected:**

Three PTQ variants on 10 OmniDocBench pages vs fp32 640×640:

1. QDQ + QInt8 + MinMax + Conv+MatMul: 99.1% detection drop (catastrophic — attention softmax breaks).
2. QDQ + QInt8 + MinMax + Conv-only: 98.7% drop (MinMax calibration too aggressive on outlier activations).
3. QDQ + QUInt8 + Percentile 99.999 + Conv-only: 16.7% drop, matched IoU 0.901, score drift 0.101 — closest but fails gate. `formula` -25%, `text` -13%.

Root cause: PP-DocLayoutV3 is DETR-style with a 300-query head. Small per-query score shifts (≤0.1 sigmoid-space) push many detections below the 0.3 threshold. Backbone Conv activations feed the scoring head with high gain — the regime where PTQ fails. Ryzen 5600X is Zen3 (AVX2 only, no VNNI), so even if V3's 16.7% drop were acceptable, the theoretical int8 Conv ceiling is only 1.5-1.8× — smaller than the measured 1.42× of Phase 2a and without the quality tradeoff. Not worth pursuing on this detector.

**Why Phase 3 (Conv+BN fusion) is a literal no-op:**

Probe at `docker/cpu/tests/probe_ort_fusion.py` — only script kept from this session. Runs `sess_options.optimized_model_filepath` on the live graph and counts ops. On this stack:
- Raw graph already has **0 BN nodes** (torch's `do_constant_folding=True` on an `.eval()` model fuses Conv+BN at export time).
- ORT optimizer further fuses Sigmoid (38→7), Relu (91→38), MatMul+Add → Gemm (0→130), SkipLayerNorm (0→13). Total nodes 1549 → 1330.
- Conv count unchanged at 130 — the kernel work is unchanged; everything around it is already as fused as ORT can make it.

If a future maintainer considers "fusing Conv+BN" as an optimization, point them at the probe first.

**What a future attempt would need to get right:**

For resolution reduction (Phase 2a retry):
1. Parity corpus must span passport/ID/receipt/invoice/handwritten/multilingual samples, not just OmniDocBench.
2. Gate on per-class F1 against the full detector output, not just count/IoU/score deltas.
3. Treat it as a per-deployment opt-in knob, not a default.

For kernel-level wins in general, the structural answer on this stack is:
- ~~**DocLayout-YOLO detector swap (Phase 1 in the plan).**~~ **Falsified 2026-04-23 by direct probe** — see `project_doclayout_yolo_probe_2026_04_23.md`. The "~3× faster" claim in public benchmarks is GPU-only; on Ryzen 5600X CPU the DocLayout-YOLO/YOLOv10 backbone is 3× **slower** than PP-DocLayoutV3 at imgsz=1024, and 11% slower at imgsz=640. Only beats PP-DocLayoutV3 at imgsz=512 (1.32×), where small-text quality is already compromised. Do not re-pursue on this CPU hardware.

On this specific hardware (Ryzen 5600X, Zen3, AVX2 only, no VNNI), there is no known kernel-level lever that raises layout throughput meaningfully without a quality or stability regression. Meaningful further gains need infra-level changes: GPU with more VRAM for layout-on-CUDA, more CPU cores for more workers, or horizontal scale.
