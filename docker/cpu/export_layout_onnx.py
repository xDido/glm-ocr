"""Export PP-DocLayoutV3 to ONNX for CPU inference.

Two graphs are produced, both inside the HF cache volume so they survive
container recreation alongside the model weights:

    ${HF_HOME:-/root/.cache/huggingface}/glmocr-layout-onnx/
        pp_doclayout_v3.onnx            raw graph (forward pass only)
        pp_doclayout_v3.onnx.data       weights sidecar
        pp_doclayout_v3_fused.onnx      fused graph (forward + post-proc)
        pp_doclayout_v3_fused.onnx.data weights sidecar

Raw graph outputs:
    (logits, pred_boxes, order_logits, out_masks, last_hidden_state)
Consumed by glmocr's upstream torch post-proc OR our numpy port.

Fused graph outputs (Phase 2):
    (scores_topk, labels_topk, boxes_topk_xyxy, order_seq_topk,
     masks_topk_logits, last_hidden_state)
The deterministic-arithmetic subset of post-processing is baked in:
sigmoid, top-K over (N·C), cxcywh→xyxy, target-size rescale, and the
pairwise-vote reading-order decoder. Mask thresholding, per-image score
filter, sort-by-order, polygon extraction, and downstream
per-class/NMS/unclip steps remain in numpy — they have data-dependent
shapes or cv2 dependencies that aren't worth forcing into ONNX.

Both exports are idempotent: skip if the target file exists and is
non-empty.

We avoid `optimum.exporters.onnx` because optimum 1.x pins
transformers<5 while glmocr pulls transformers 5.5.4 for the
`pp_doclayout_v3` model type — a hard conflict. Raw `torch.onnx.export`
on a thin wrapper module is sufficient for a transformer-detector with
a single static-shape input.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _cache_dir() -> Path:
    hf = os.environ.get("HF_HOME") or "/root/.cache/huggingface"
    return Path(hf) / "glmocr-layout-onnx"


def _export_raw(model, proc, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[export] raw already present: {out_path}", flush=True)
        return
    import torch
    from PIL import Image

    print(f"[export] raw -> {out_path}", flush=True)
    dummy = proc(
        images=[Image.new("RGB", (1024, 768), "white")],
        return_tensors="pt",
    )["pixel_values"]

    class WrappedDetector(torch.nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, pixel_values: "torch.Tensor") -> tuple:
            out = self.m(pixel_values=pixel_values)
            return (
                out.logits,
                out.pred_boxes,
                out.order_logits,
                out.out_masks,
                out.last_hidden_state,
            )

    wrapped = WrappedDetector(model).eval()
    t0 = time.perf_counter()
    torch.onnx.export(
        wrapped,
        (dummy,),
        str(out_path),
        input_names=["pixel_values"],
        output_names=[
            "logits",
            "pred_boxes",
            "order_logits",
            "out_masks",
            "last_hidden_state",
        ],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
            "order_logits": {0: "batch"},
            "out_masks": {0: "batch"},
            "last_hidden_state": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    elapsed = time.perf_counter() - t0
    size_mb = out_path.stat().st_size / 1e6
    data_path = out_path.with_suffix(out_path.suffix + ".data")
    data_mb = data_path.stat().st_size / 1e6 if data_path.exists() else 0.0
    print(
        f"[export] raw OK in {elapsed:.1f}s  graph={size_mb:.1f} MB  "
        f"weights={data_mb:.1f} MB",
        flush=True,
    )


def _export_fused(model, proc, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[export] fused already present: {out_path}", flush=True)
        return
    import torch
    from PIL import Image

    print(f"[export] fused -> {out_path}", flush=True)
    dummy_inputs = proc(
        images=[Image.new("RGB", (1024, 768), "white")],
        return_tensors="pt",
    )
    dummy_pixel_values = dummy_inputs["pixel_values"]
    # target_sizes is (B, 2) in [H, W] order — same convention upstream
    # `post_process_object_detection` uses for its `target_sizes` arg.
    dummy_target_sizes = torch.tensor([[768, 1024]], dtype=torch.int64)

    class WrappedDetectorFused(torch.nn.Module):
        """Forward pass + deterministic-arithmetic post-proc, all in torch
        so torch.onnx.export captures it. Mirrors the numpy port in
        layout_postprocess.py::np_post_process_object_detection up to
        (but not including) the data-dependent threshold filter."""

        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(
            self, pixel_values: "torch.Tensor", target_sizes: "torch.Tensor"
        ) -> tuple:
            out = self.m(pixel_values=pixel_values)
            logits = out.logits  # (B, N, C)
            pred_boxes = out.pred_boxes  # (B, N, 4) cxcywh normalized
            order_logits = out.order_logits  # (B, N, N)
            out_masks = out.out_masks  # (B, N, Hm, Wm)

            B, N, C = logits.shape

            # Reading-order decoder: vote-rank via pairwise sigmoid scores.
            order_scores = torch.sigmoid(order_logits)
            triu_part = order_scores.triu(diagonal=1).sum(dim=1)
            tril_part = (
                (1.0 - order_scores.transpose(1, 2)).tril(diagonal=-1).sum(dim=1)
            )
            votes = triu_part + tril_part  # (B, N)
            pointers = torch.argsort(votes, dim=1)  # (B, N)
            ranks = (
                torch.arange(N, dtype=pointers.dtype, device=pointers.device)
                .unsqueeze(0)
                .expand(B, -1)
                .contiguous()
            )
            # Functional scatter to keep the graph traceable.
            order_seq = torch.zeros_like(pointers).scatter(1, pointers, ranks)  # (B, N)

            # cxcywh → xyxy, then rescale to image coords.
            centers = pred_boxes[..., :2]
            dims = pred_boxes[..., 2:]
            boxes_xyxy = torch.cat(
                [centers - 0.5 * dims, centers + 0.5 * dims], dim=-1
            )  # (B, N, 4)
            img_h = target_sizes[:, 0].to(boxes_xyxy.dtype)
            img_w = target_sizes[:, 1].to(boxes_xyxy.dtype)
            scale = torch.stack([img_w, img_h, img_w, img_h], dim=1).unsqueeze(
                1
            )  # (B, 1, 4)
            boxes_xyxy = boxes_xyxy * scale

            # Sigmoid + TopK over flattened (N·C).
            scores_all = torch.sigmoid(logits)  # (B, N, C)
            flat = scores_all.flatten(1)  # (B, N*C)
            scores_topk, indices = torch.topk(flat, N, dim=-1)
            labels_topk = (indices % C).to(torch.int64)  # (B, N)
            query_idx = (indices // C).to(torch.int64)  # (B, N)

            # Gather boxes / masks / order_seqs by query_idx.
            boxes_topk = boxes_xyxy.gather(
                1, query_idx.unsqueeze(-1).expand(-1, -1, 4)
            )  # (B, N, 4)
            Hm = out_masks.shape[-2]
            Wm = out_masks.shape[-1]
            masks_topk = out_masks.gather(
                1, query_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, Hm, Wm)
            )  # (B, N, Hm, Wm)
            order_seq_topk = order_seq.gather(1, query_idx)  # (B, N)

            return (
                scores_topk,  # (B, N) fp32
                labels_topk,  # (B, N) int64
                boxes_topk,  # (B, N, 4) fp32 in image coords
                order_seq_topk,  # (B, N) int64
                masks_topk,  # (B, N, Hm, Wm) fp32 — raw mask logits
                out.last_hidden_state,  # (B, N, D) fp32
            )

    wrapped = WrappedDetectorFused(model).eval()

    t0 = time.perf_counter()
    torch.onnx.export(
        wrapped,
        (dummy_pixel_values, dummy_target_sizes),
        str(out_path),
        input_names=["pixel_values", "target_sizes"],
        output_names=[
            "scores_topk",
            "labels_topk",
            "boxes_topk",
            "order_seq_topk",
            "masks_topk",
            "last_hidden_state",
        ],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "target_sizes": {0: "batch"},
            "scores_topk": {0: "batch"},
            "labels_topk": {0: "batch"},
            "boxes_topk": {0: "batch"},
            "order_seq_topk": {0: "batch"},
            "masks_topk": {0: "batch"},
            "last_hidden_state": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    elapsed = time.perf_counter() - t0
    size_mb = out_path.stat().st_size / 1e6
    data_path = out_path.with_suffix(out_path.suffix + ".data")
    data_mb = data_path.stat().st_size / 1e6 if data_path.exists() else 0.0
    print(
        f"[export] fused OK in {elapsed:.1f}s  graph={size_mb:.1f} MB  "
        f"weights={data_mb:.1f} MB",
        flush=True,
    )


def main() -> int:
    model_dir = os.environ.get(
        "LAYOUT_MODEL_DIR", "PaddlePaddle/PP-DocLayoutV3_safetensors"
    )
    out_dir = _cache_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "pp_doclayout_v3.onnx"
    fused_path = out_dir / "pp_doclayout_v3_fused.onnx"

    need_raw = not (raw_path.exists() and raw_path.stat().st_size > 0)
    need_fused = not (fused_path.exists() and fused_path.stat().st_size > 0)
    if not (need_raw or need_fused):
        print(f"[export] both graphs already present in {out_dir}", flush=True)
        return 0

    print(f"[export] loading {model_dir}", flush=True)
    from transformers import (
        PPDocLayoutV3ForObjectDetection,
        PPDocLayoutV3ImageProcessor,
    )

    model = PPDocLayoutV3ForObjectDetection.from_pretrained(model_dir).eval()
    proc = PPDocLayoutV3ImageProcessor.from_pretrained(model_dir)

    if need_raw:
        _export_raw(model, proc, raw_path)
    else:
        print(f"[export] raw already present: {raw_path}", flush=True)
    if need_fused:
        _export_fused(model, proc, fused_path)
    else:
        print(f"[export] fused already present: {fused_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
