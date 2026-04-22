"""End-to-end parity: fused graph vs raw graph+numpy, comparing the
final detection lists (after threshold filter + sort-by-order). This is
the output the downstream OCR pipeline consumes, so this is the only
parity that truly matters for production equivalence.

Tolerates tie-break divergence in top-K (torch.topk in the fused graph
vs np.argsort(kind='stable') in the raw+numpy path), as long as the
post-threshold detection set is equivalent. Note the fused graph's
top-K semantics actually match the ORIGINAL upstream torch post-proc
more closely than the Phase-1 numpy path does.

Pairs detections by IoU (Hungarian-ish greedy) rather than by index
since tie-breaking can rotate equal-score items.
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
FUSED_PATH = ONNX_DIR / "pp_doclayout_v3_fused.onnx"

IMAGES_DIR = os.environ.get("PARITY_IMAGES_DIR", "/tmp/bench_images")
N_PAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 20
THRESHOLD = float(os.environ.get("PARITY_THRESHOLD", "0.3"))
IOU_BAR = float(os.environ.get("PARITY_IOU_BAR", "0.98"))
SCORE_TOL = float(os.environ.get("PARITY_SCORE_TOL", "1e-4"))

sys.path.insert(0, "/app")
from layout_postprocess import (  # noqa: E402
    _post_process_from_fused,
    np_post_process_object_detection,
)


def _iou(b1, b2):
    x1, y1, x2, y2 = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    return inter / max(1e-9, a1 + a2 - inter)


def _match(a_boxes, a_scores, a_labels, b_boxes, b_scores, b_labels):
    """Greedy match by IoU. Returns list of (a_idx, b_idx, iou)."""
    n = len(a_boxes)
    m = len(b_boxes)
    used_b = set()
    pairs = []
    for i in range(n):
        best_j, best_iou = -1, -1.0
        for j in range(m):
            if j in used_b:
                continue
            iu = _iou(a_boxes[i], b_boxes[j])
            if iu > best_iou:
                best_iou = iu
                best_j = j
        if best_j >= 0 and best_iou >= 0:
            pairs.append((i, best_j, best_iou))
            used_b.add(best_j)
    return pairs


def main() -> int:
    paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))[:N_PAGES]
    if not paths:
        paths = ["/app/smoke_test.png"]
    print(f"[parity-e2e] testing {len(paths)} page(s), threshold={THRESHOLD}")

    processor = PPDocLayoutV3ImageProcessor.from_pretrained(MODEL_DIR)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    raw_sess = ort.InferenceSession(str(RAW_PATH), opts, providers=["CPUExecutionProvider"])
    fused_sess = ort.InferenceSession(str(FUSED_PATH), opts, providers=["CPUExecutionProvider"])

    n_ok = 0
    n_count_diff = 0
    n_unmatched = 0
    n_label_diff = 0
    worst_score_delta = 0.0
    worst_iou = 1.0
    t0 = time.perf_counter()

    for idx, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        pv = processor(images=[img], return_tensors="np")["pixel_values"]
        target_sizes = np.array([img.size[::-1]], dtype=np.int64)

        # Raw + numpy path (Phase 1 reference).
        logits, pred_boxes, order_logits, out_masks, _ = raw_sess.run(
            None, {"pixel_values": pv}
        )
        raw_np = np_post_process_object_detection(
            logits, pred_boxes, order_logits, out_masks,
            target_sizes=target_sizes,
            threshold=THRESHOLD,
            processor_size=dict(processor.size),
        )[0]

        # Fused + numpy tail.
        scores_topk, labels_topk, boxes_topk, order_seq_topk, masks_logits_topk, _ = \
            fused_sess.run(
                None, {"pixel_values": pv, "target_sizes": target_sizes}
            )
        fused_np = _post_process_from_fused(
            scores_topk, labels_topk, boxes_topk, order_seq_topk, masks_logits_topk,
            target_sizes=target_sizes,
            threshold=THRESHOLD,
            processor_size=dict(processor.size),
        )[0]

        a_scores = raw_np["scores"]; a_labels = raw_np["labels"]; a_boxes = raw_np["boxes"]
        b_scores = fused_np["scores"]; b_labels = fused_np["labels"]; b_boxes = fused_np["boxes"]

        if len(a_scores) != len(b_scores):
            n_count_diff += 1
            print(f"[parity-e2e] {Path(p).name}: count {len(a_scores)} vs {len(b_scores)}")
            continue

        if len(a_scores) == 0:
            n_ok += 1
            continue

        pairs = _match(a_boxes, a_scores, a_labels, b_boxes, b_scores, b_labels)
        if len(pairs) != len(a_scores):
            n_unmatched += 1
            print(f"[parity-e2e] {Path(p).name}: unmatched detections")
            continue

        page_ok = True
        for i, j, iu in pairs:
            worst_iou = min(worst_iou, iu)
            s_delta = abs(float(a_scores[i]) - float(b_scores[j]))
            worst_score_delta = max(worst_score_delta, s_delta)
            if iu < IOU_BAR or s_delta > SCORE_TOL or a_labels[i] != b_labels[j]:
                if a_labels[i] != b_labels[j]:
                    n_label_diff += 1
                page_ok = False
        if page_ok:
            n_ok += 1
        else:
            print(f"[parity-e2e] {Path(p).name}: page mismatch")

        if (idx + 1) % 5 == 0:
            print(f"[parity-e2e] progress {idx+1}/{len(paths)}  ok={n_ok}  "
                  f"elapsed={time.perf_counter()-t0:.1f}s")

    print()
    print("[parity-e2e] summary:")
    print(f"  pages:                  {len(paths)}")
    print(f"  pages ok:               {n_ok}")
    print(f"  count mismatch:         {n_count_diff}")
    print(f"  unmatched:              {n_unmatched}")
    print(f"  label mismatches:       {n_label_diff}")
    print(f"  worst score Δ (paired): {worst_score_delta:.2e}")
    print(f"  worst IoU (paired):     {worst_iou:.4f}")
    print(f"  elapsed:                {time.perf_counter()-t0:.1f}s")
    pass_rate = n_ok / max(1, len(paths))
    print(f"  pass rate:              {pass_rate:.2%}")
    return 0 if pass_rate >= 0.99 else 1


if __name__ == "__main__":
    sys.exit(main())
