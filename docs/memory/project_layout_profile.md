---
name: Layout ONNX forward is Conv-dominated
description: Per-op profile of pp_doclayout_v3.onnx (CPUExecutionProvider, INTRA_OP=3) — Conv is 76% of wall time, pointing the next optimization at DnnlEP / OpenVINOEP / int8-static-with-exclusions rather than more graph surgery.
type: project
originSessionId: c8845dae-98e9-4b22-a7ae-02787ef2bf30
---
Profile from `scripts/profile_layout.py` inside glmocr-cpu on 2026-04-22 (RTX 3060 Ti host, 12-core cgroup, ORT 1.24.4 with only `['AzureExecutionProvider', 'CPUExecutionProvider']` available):

- Wall time: mean 1,134 ms per forward (warm, solo call, 800×800 input).
- **Conv: 859 ms per call = 76% of wall.** 2,990 Conv invocations per forward.
- Gemm: 113 ms (10%). Concat: 77 ms (7%). GridSample: 61 ms (5%). MatMul: 46 ms. Where: 42 ms. Cast: 38 ms.

**Why:** We already shipped ONNX backend + numpy post-proc + the cross-request coalescer + LAYOUT_ONNX_THREADS=3. The remaining per-call cost is overwhelmingly native Conv kernels, so:
- More intra-op threads don't help — we're already on the 12-core cgroup ceiling.
- Graph-surgery experiments like the fused-v2 retry won't help the Conv share (E1 failed the gate, 2026-04-22).
- Relieving SGLang doesn't help — matrix rps is capped by this 4.5 s layout forward (E2 failed the gate, 2026-04-22).

**How to apply:** The next layout optimization should target the Conv kernel, not the graph shape. Highest-ROI candidates:
1. Swap providers to Dnnl/oneDNN or OpenVINO EP (ships in `onnxruntime-openvino`, ~200 MB wheel; historical 20-30% on CNN-heavy). Requires replacing `onnxruntime` in `docker/cpu/Dockerfile.slim`.
2. Static per-layer int8 calibration with exclusions on the bbox-regression head (dynamic int8 was 3x slower and broke pred_boxes — OPTIMIZATIONS.md). Half-day effort, needs calibration dataset.
3. Reducing the layout model's input resolution (processor default 800×800 → 600×600 halves Conv FLOPs). Quality tradeoff must be measured on OmniDocBench.

Don't re-profile without reason — this profile is stable. Re-run `docker exec glmocr-cpu python /tmp/profile_layout.py` after copying `scripts/profile_layout.py` with `MSYS_NO_PATHCONV=1 docker cp ...`.
