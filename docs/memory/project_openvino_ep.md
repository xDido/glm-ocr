---
name: OpenVINOExecutionProvider was rejected for PP-DocLayoutV3 layout ONNX
description: OpenVINO EP was tried for the layout stage (2026-04-22→23) and reverted in commits 102f9c1 + 1350454. The apparent "3x RPS" win was 15% silent empty-response inflation; the root cause is PP-DocLayoutV3's batch-1-baked shape ops, unfixable without a different detector.
type: project
originSessionId: c8845dae-98e9-4b22-a7ae-02787ef2bf30
---
**REVERTED.** Commits `1350454` + `102f9c1` unshipped the OpenVINO EP attempt and the `OPTIMIZATIONS.md` postmortem lives in the Rejected section (commit `968f73e`). Do not re-ship without reading the postmortem.

**Short version of the failure chain:**

1. E4 profile showed layout Conv = 76% of wall time. Swapping MLAS CPU EP → OpenVINOExecutionProvider on the same ONNX graph measured −31% per solo warm forward in `scripts/ov_bench.py`, 3–4× rps gain on the 3 × N=200 matrix. Shipped.
2. Later spot-check: 3 / 20 sequential requests returned empty detections as HTTP 200. Log inspection: 14,594 `node_view_320` Reshape errors in 2 h. glmocr's pipeline catches the ORT error, returns empty detections, bench counts HTTP 200 as success — the "3× win" was ~15% inflated.
3. Root cause is in the ONNX graph: 52 / 164 Reshape ops ship with frozen `[1, ...]` shape constants because PP-DocLayoutV3's torch source uses literal `.view(1, 300, ...)` in attention/decoder paths. MLAS silently accepts the input/output-product mismatch when batch > 1; OpenVINO correctly rejects it.
4. Every workaround dead-ended:
   - `disable_dynamic_shapes: False` OV provider option — no effect.
   - `torch.onnx.export(dynamo=True)` — worse: 106 frozen-bug Reshapes (vs 52 in the legacy exporter).
   - `openvino.convert_model(torch_model, ...)` — aborts conversion entirely because `aten::mul` inside attention has shape mismatch `f32[?,300,4] × f32[?,300,200,1]`. Model forward isn't batch-safe at the source.
   - Batcher OFF (so every call is batch=1, no shape variation): 0 empties but 0.38 rps at c=12. OV's per-inference fixed cost at batch=1 without coalescer amortization is catastrophic on this Ryzen 5600X host.

**What this tells us about future work:**

- **The right fix for layout throughput is a different detector, not a different ORT EP.** DocLayout-YOLO (YOLOv10-based, no `.view(N, ...)` patterns, ~200–400 ms/forward) is the obvious candidate. It's batch-dynamic-safe and unlocks OV naturally as a bonus — but OV is not actually needed since YOLO is already ~3× faster than PP-DocLayoutV3 on any EP.
- **The E4 profile finding is still load-bearing.** Conv dominates layout; when you swap detectors, measure Conv share to confirm where time goes.
- The 3060 Ti host has no Intel iGPU/NPU, so OV would be CPU-only regardless. On Intel hardware (Arc, Xeon w/ iGPU) the per-call overhead might be different and the empty-response trap might still bite; either way the frozen-shape bug is a model-level issue.
- If a future maintainer insists on retrying OV, the only non-model-swap option is `onnx-graphsurgeon` patching of all 52 Reshape nodes PLUS the attention `aten::mul` shape mismatch, which is real graph surgery (hours of work + parity validation).

**Rollback verified:** post-revert config is `LAYOUT_BACKEND=onnx`, `LAYOUT_POSTPROC=numpy`, `LAYOUT_GRAPH=raw`, `LAYOUT_ONNX_THREADS=3`, `LAYOUT_BATCH_ENABLED=true`, `LAYOUT_BATCH_MAX=8`. ORT 1.24.4, providers = CPUExecutionProvider. Layout-forward per-call wall time back to ~1,100 ms (Conv-bound). Throughput ceiling ~3.58 rps at c=12 per `loadtest/results/omnidoc-20260422-135324-asyncio-matrix.md`.
