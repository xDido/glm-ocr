"""Parity check: fused ONNX graph vs raw graph + numpy post-proc.

Loads both graphs via onnxruntime, runs them on sample pages, then
compares the intermediate tensors that the numpy path would produce
against what the fused graph emits directly.

Comparison point: the 5 post-proc outputs just before the data-dependent
per-image score filter. Specifically:
    scores_topk, labels_topk, boxes_topk (xyxy image-coords),
    order_seq_topk, masks_topk_bool (v2: sigmoid + threshold inside the
    graph, emitted as bool).

Usage inside container:
    docker exec glmocr-cpu python /app/parity_fused_vs_raw_in_container.py [N]

N = number of pages to test (default 10). Images are pulled from
/tmp/bench_images (docker cp the datasets/OmniDocBench/images in first).

Exit 0 on pass, non-zero on mismatch beyond tolerance.
"""
from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import PPDocLayoutV3ImageProcessor


MODEL_DIR = os.environ.get(
    "LAYOUT_MODEL_DIR", "PaddlePaddle/PP-DocLayoutV3_safetensors"
)
HF_HOME = os.environ.get("HF_HOME", "/root/.cache/huggingface")
ONNX_DIR = Path(HF_HOME) / "glmocr-layout-onnx"
RAW_PATH = ONNX_DIR / "pp_doclayout_v3.onnx"
FUSED_PATH = ONNX_DIR / "pp_doclayout_v3_fused_v2.onnx"

IMAGES_DIR = os.environ.get("PARITY_IMAGES_DIR", "/tmp/bench_images")
N_PAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SCORE_TOL = float(os.environ.get("PARITY_SCORE_TOL", "1e-4"))
BOX_TOL_PX = float(os.environ.get("PARITY_BOX_TOL", "0.5"))
# Threshold used by the fused graph's in-graph sigmoid + threshold on
# per-query masks. Must match the value passed at Run time below.
MASK_THRESHOLD = float(os.environ.get("PARITY_MASK_THRESHOLD", "0.3"))

sys.path.insert(0, "/app")
from layout_postprocess import _np_get_order_seqs, _sigmoid  # noqa: E402


def main() -> int:
    if not RAW_PATH.exists() or not FUSED_PATH.exists():
        print(f"[parity] missing graph(s): raw={RAW_PATH.exists()} "
              f"fused={FUSED_PATH.exists()}", file=sys.stderr)
        return 2

    paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))[:N_PAGES]
    if not paths:
        # Fall back to the single smoke test page if bench images aren't present.
        paths = ["/app/smoke_test.png"]
    print(f"[parity] testing {len(paths)} page(s)")

    processor = PPDocLayoutV3ImageProcessor.from_pretrained(MODEL_DIR)

    # intra_op=1 to match production; disable ORT optimizations for reproducibility
    # across both sessions.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    raw_sess = ort.InferenceSession(str(RAW_PATH), opts, providers=["CPUExecutionProvider"])
    fused_sess = ort.InferenceSession(str(FUSED_PATH), opts, providers=["CPUExecutionProvider"])

    worst = {
        "score": 0.0,
        "box": 0.0,
        "mask_mismatches": 0,
        "order_mismatch_pages": 0,
        "label_mismatch_pages": 0,
    }
    n_ok = 0
    t0 = time.perf_counter()

    for idx, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        pv = processor(images=[img], return_tensors="np")["pixel_values"]
        target_sizes = np.array([img.size[::-1]], dtype=np.int64)  # (H, W)

        # Raw graph: get the 5 pre-post-proc tensors.
        logits, pred_boxes, order_logits, out_masks, _last_hs = raw_sess.run(
            None, {"pixel_values": pv}
        )

        # Replicate what the fused graph does, in numpy, on the raw outputs.
        B, N, C = logits.shape
        # Order decoder (matches the baked torch version).
        order_seq_ref = _np_get_order_seqs(order_logits)  # (B, N)

        # Box decode + rescale.
        centers = pred_boxes[..., :2]
        dims = pred_boxes[..., 2:]
        boxes_xyxy = np.concatenate(
            [centers - 0.5 * dims, centers + 0.5 * dims], axis=-1
        )
        img_h = target_sizes[:, 0].astype(boxes_xyxy.dtype)
        img_w = target_sizes[:, 1].astype(boxes_xyxy.dtype)
        scale = np.stack([img_w, img_h, img_w, img_h], axis=1)[:, None, :]
        boxes_xyxy = boxes_xyxy * scale

        scores_all = _sigmoid(logits)
        flat = scores_all.reshape(B, -1)
        # argsort-based top-N mirror of torch.topk (stable for determinism).
        top_idx = np.argsort(-flat, axis=-1, kind="stable")[:, :N]
        row = np.arange(B)[:, None]
        scores_ref = flat[row, top_idx]
        labels_ref = (top_idx % C).astype(np.int64)
        query_idx = (top_idx // C).astype(np.int64)
        boxes_ref = np.take_along_axis(
            boxes_xyxy, np.broadcast_to(query_idx[..., None], (B, N, 4)),
            axis=1,
        )
        Hm, Wm = out_masks.shape[-2:]
        masks_logits_ref = np.take_along_axis(
            out_masks,
            np.broadcast_to(query_idx[..., None, None], (B, N, Hm, Wm)),
            axis=1,
        )
        # v2 fused graph does sigmoid + threshold inside ONNX — mirror that
        # here so we can compare the bool output bit-exactly.
        masks_ref_bool = _sigmoid(masks_logits_ref) > MASK_THRESHOLD
        order_ref = np.take_along_axis(order_seq_ref, query_idx, axis=1)

        # Fused v2: pass the threshold as a graph input. Output mask is bool.
        fused_out = fused_sess.run(
            None,
            {
                "pixel_values": pv,
                "target_sizes": target_sizes,
                # ORT rejects bare np.float32 scalars; wrap in a 0-d array.
                "threshold": np.asarray(MASK_THRESHOLD, dtype=np.float32),
            },
        )
        scores_f, labels_f, boxes_f, order_f, masks_f, _last_hs_f = fused_out

        # Compare.
        score_delta = float(np.abs(scores_ref - scores_f).max())
        box_delta = float(np.abs(boxes_ref - boxes_f).max())
        # Mask is bool on both sides — count per-element disagreement.
        mask_disagreement = int(
            np.count_nonzero(masks_ref_bool != masks_f.astype(bool))
        )
        label_ok = bool((labels_ref == labels_f).all())
        order_ok = bool((order_ref == order_f).all())

        worst["score"] = max(worst["score"], score_delta)
        worst["box"] = max(worst["box"], box_delta)
        worst["mask_mismatches"] = max(
            worst["mask_mismatches"], mask_disagreement
        )
        if not label_ok:
            worst["label_mismatch_pages"] += 1
        if not order_ok:
            worst["order_mismatch_pages"] += 1

        # Mask parity is bit-exact: no tolerance. Upstream sigmoid is the
        # same math as torch's, and > is a boolean op. Any disagreement is
        # a real divergence and should fail the test.
        page_ok = (
            score_delta <= SCORE_TOL
            and box_delta <= BOX_TOL_PX
            and mask_disagreement == 0
            and label_ok
            and order_ok
        )
        if page_ok:
            n_ok += 1
        else:
            print(f"[parity] {Path(p).name}: "
                  f"score Δ={score_delta:.2e}  box Δ={box_delta:.2f}  "
                  f"mask_mismatches={mask_disagreement}  "
                  f"labels={label_ok} order={order_ok}")

        if (idx + 1) % 5 == 0:
            print(f"[parity] progress {idx+1}/{len(paths)}  ok={n_ok}  "
                  f"elapsed={time.perf_counter()-t0:.1f}s")

    print()
    print("[parity] summary:")
    print(f"  pages:                       {len(paths)}")
    print(f"  pages ok:                    {n_ok}")
    print(f"  worst score Δ:               {worst['score']:.2e}")
    print(f"  worst box Δ:                 {worst['box']:.4f} px")
    print(f"  worst mask bit mismatches:   {worst['mask_mismatches']}")
    print(f"  pages with label diff:       {worst['label_mismatch_pages']}")
    print(f"  pages with order diff:       {worst['order_mismatch_pages']}")
    print(f"  elapsed:                     {time.perf_counter()-t0:.1f}s")

    return 0 if n_ok == len(paths) else 1


if __name__ == "__main__":
    sys.exit(main())
