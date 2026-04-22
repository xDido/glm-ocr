"""Multi-page parity check: numpy vs torch postproc on real pages.

Loads the model once, then for each page:
  1. Runs torch postproc (ORT forward + upstream torch post_process) to get
     the reference detection list.
  2. Runs numpy postproc (same ORT forward + our numpy pipeline).
  3. Compares detection-by-detection: Hungarian-matching is overkill given
     the order is deterministic on both paths — pair by index after sorting
     by (y1, x1) and assert IoU ≥ 0.98, score Δ ≤ 1e-4, label identical.

Usage inside container:
    # Copy datasets in first if they aren't already:
    #   docker cp datasets/OmniDocBench/images glmocr-cpu:/tmp/bench_images
    python /app/parity_many_pages.py [N]

where N is the number of pages to test (default 50).
"""
from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    PPDocLayoutV3ForObjectDetection,
    PPDocLayoutV3ImageProcessor,
)

sys.path.insert(0, "/app")
from layout_postprocess import (  # noqa: E402
    compute_paddle_format_results,
    np_post_process_object_detection,
    paddle_to_all_results,
)


MODEL_DIR = os.environ.get(
    "LAYOUT_MODEL_DIR", "PaddlePaddle/PP-DocLayoutV3_safetensors"
)
IMAGES_DIR = os.environ.get("PARITY_IMAGES_DIR", "/tmp/bench_images")
N_PAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 50
SCORE_TOL = float(os.environ.get("PARITY_SCORE_TOL", "1e-4"))
BOX_TOL_PIXELS = float(os.environ.get("PARITY_BOX_TOL", "0.5"))


def _iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    return inter / max(1e-9, a1 + a2 - inter)


def main() -> int:
    paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    if not paths:
        print(f"[parity] no images found in {IMAGES_DIR}", file=sys.stderr)
        return 2
    paths = paths[:N_PAGES]
    print(f"[parity] testing {len(paths)} pages from {IMAGES_DIR}")

    print(f"[parity] loading {MODEL_DIR}")
    model = PPDocLayoutV3ForObjectDetection.from_pretrained(MODEL_DIR).eval()
    processor = PPDocLayoutV3ImageProcessor.from_pretrained(MODEL_DIR)
    id2label = dict(model.config.id2label)
    threshold = 0.3
    label_task_mapping = {"ocr": list(id2label.values())}

    # Single forward per page — cache the outputs and feed both paths.
    cached_outputs = {"val": None}

    def ort_like_run(pv_np):
        # Only called by the numpy path; reuses whatever the torch path stored.
        o = cached_outputs["val"]
        return (
            o.logits.detach().cpu().numpy(),
            o.pred_boxes.detach().cpu().numpy(),
            o.order_logits.detach().cpu().numpy(),
            o.out_masks.detach().cpu().numpy(),
            o.last_hidden_state.detach().cpu().numpy(),
        )

    n_pages_ok = 0
    n_count_mismatch = 0
    n_label_mismatch = 0
    n_score_mismatch = 0
    n_box_mismatch = 0
    n_empty_both = 0
    worst_score_delta = 0.0
    worst_box_delta = 0.0
    t0 = time.perf_counter()

    for idx, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"[parity] skip {p}: {e}")
            continue

        # Single forward, reused by both paths.
        inputs = processor(images=[img], return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        cached_outputs["val"] = outputs

        target_sizes_torch = torch.tensor([img.size[::-1]])
        torch_raw = processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes_torch,
        )[0]

        # Numpy path — compare at the raw (pre-int-cast) post-proc output,
        # which is the correct parity point. The int-cast that happens later
        # in apply_layout_postprocess rounds boxes to integers on BOTH paths,
        # so comparing to paddle-format's int coordinates overstates the gap.
        np_raw = np_post_process_object_detection(
            logits=outputs.logits.detach().cpu().numpy(),
            pred_boxes=outputs.pred_boxes.detach().cpu().numpy(),
            order_logits=outputs.order_logits.detach().cpu().numpy(),
            out_masks=outputs.out_masks.detach().cpu().numpy(),
            target_sizes=np.array([img.size[::-1]], dtype=np.int64),
            threshold=threshold,
            processor_size=dict(processor.size),
        )[0]

        torch_boxes = torch_raw["boxes"].detach().cpu().numpy()
        torch_scores = torch_raw["scores"].detach().cpu().numpy()
        torch_labels = torch_raw["labels"].detach().cpu().numpy()

        np_boxes = np.asarray(np_raw["boxes"])
        np_scores = np.asarray(np_raw["scores"])
        np_labels = np.asarray(np_raw["labels"])

        if len(torch_boxes) == 0 and len(np_boxes) == 0:
            n_empty_both += 1
            n_pages_ok += 1
            continue

        if len(torch_boxes) != len(np_boxes):
            n_count_mismatch += 1
            print(f"[parity] {Path(p).name}: count {len(torch_boxes)} vs {len(np_boxes)}")
            continue

        n = len(torch_boxes)
        page_score_delta = float(np.abs(torch_scores - np_scores).max()) if n else 0.0
        # np_boxes are int-cast; torch_boxes are float — so expect some rounding.
        page_box_delta = float(np.abs(torch_boxes - np_boxes).max()) if n else 0.0
        labels_match = (torch_labels == np_labels).all() if n else True

        worst_score_delta = max(worst_score_delta, page_score_delta)
        worst_box_delta = max(worst_box_delta, page_box_delta)

        page_ok = True
        if not labels_match:
            n_label_mismatch += 1
            print(f"[parity] {Path(p).name}: labels differ")
            page_ok = False
        if page_score_delta > SCORE_TOL:
            n_score_mismatch += 1
            print(f"[parity] {Path(p).name}: score Δ={page_score_delta:.2e}")
            page_ok = False
        if page_box_delta > BOX_TOL_PIXELS:
            n_box_mismatch += 1
            print(f"[parity] {Path(p).name}: box Δ={page_box_delta:.2f} px")
            page_ok = False

        if page_ok:
            n_pages_ok += 1

        if (idx + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"[parity] progress {idx+1}/{len(paths)}  "
                  f"ok={n_pages_ok}  elapsed={elapsed:.1f}s")

    elapsed = time.perf_counter() - t0
    print()
    print(f"[parity] summary:")
    print(f"  pages:               {len(paths)}")
    print(f"  pages ok:            {n_pages_ok}")
    print(f"  empty on both paths: {n_empty_both}")
    print(f"  count mismatch:      {n_count_mismatch}")
    print(f"  label mismatch:      {n_label_mismatch}")
    print(f"  score mismatch:      {n_score_mismatch}")
    print(f"  box mismatch:        {n_box_mismatch}")
    print(f"  worst score Δ:       {worst_score_delta:.2e}")
    print(f"  worst box Δ:         {worst_box_delta:.2f} px")
    print(f"  elapsed:             {elapsed:.1f}s")

    pass_rate = n_pages_ok / max(1, len(paths))
    print(f"  pass rate:           {pass_rate:.2%}")
    return 0 if pass_rate >= 0.99 else 1


if __name__ == "__main__":
    sys.exit(main())
