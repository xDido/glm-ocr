"""Export the PP-DocLayoutV3 HF model to ONNX for CPU inference.

Runs once per container (or whenever the cache dir is empty). Called
from the container entrypoint when LAYOUT_BACKEND=onnx. Idempotent:
exits fast if the target file already exists.

The output is placed inside the HF cache volume so it survives container
recreation alongside the model weights:

    ${HF_HOME:-/root/.cache/huggingface}/glmocr-layout-onnx/
        pp_doclayout_v3.onnx            (graph, ~4 MB)
        pp_doclayout_v3.onnx.data       (weights sidecar, ~130 MB)

We export three tensors — logits, pred_boxes, last_hidden_state —
because glmocr's post-processor reads from `.logits` and `.pred_boxes`
on the HF ForObjectDetection output, and we expose last_hidden_state
for parity with the torch path.

We avoid `optimum.exporters.onnx` because optimum 1.x pins
transformers<5 while glmocr pulls transformers 5.5.4 for the
`pp_doclayout_v3` model type — a hard conflict. Raw torch.onnx.export
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


def main() -> int:
    model_dir = os.environ.get(
        "LAYOUT_MODEL_DIR", "PaddlePaddle/PP-DocLayoutV3_safetensors"
    )
    out_dir = _cache_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pp_doclayout_v3.onnx"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[export] already present: {out_path}", flush=True)
        return 0

    print(f"[export] exporting {model_dir} -> {out_path}", flush=True)
    import torch
    from transformers import (
        PPDocLayoutV3ForObjectDetection,
        PPDocLayoutV3ImageProcessor,
    )
    from PIL import Image

    model = PPDocLayoutV3ForObjectDetection.from_pretrained(model_dir).eval()
    proc = PPDocLayoutV3ImageProcessor.from_pretrained(model_dir)

    # The processor resizes every input to 800x800, so the ONNX graph
    # uses that as the static H,W. Batch dim stays dynamic.
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
            # post_process_object_detection reads all four tensors.
            return (
                out.logits, out.pred_boxes,
                out.order_logits, out.out_masks,
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
            "logits", "pred_boxes",
            "order_logits", "out_masks", "last_hidden_state",
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
        f"[export] OK in {elapsed:.1f}s  graph={size_mb:.1f} MB  "
        f"weights={data_mb:.1f} MB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
