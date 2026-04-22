"""Parity check: numpy port vs. torch upstream, run inside the container.

Loads the real PP-DocLayoutV3 model + processor, pushes a sample image
through both paths, and asserts the outputs agree to within a tight
tolerance. Runs torch eager (ignoring the ORT backend) because this
test validates the post-proc math, not the forward-pass backend.

Usage (from host):
    docker cp docker/cpu/layout_postprocess.py glmocr-cpu:/app/
    docker cp docker/cpu/tests/parity_in_container.py glmocr-cpu:/app/
    docker exec glmocr-cpu python /app/parity_in_container.py

Exit code 0 if parity passes, non-zero otherwise.
"""
from __future__ import annotations

import os
import sys
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
    np_post_process_object_detection,
    run_numpy_layout_pipeline,
)


MODEL_DIR = os.environ.get(
    "LAYOUT_MODEL_DIR", "PaddlePaddle/PP-DocLayoutV3_safetensors"
)
SAMPLE_PATH = os.environ.get("PARITY_SAMPLE", "/app/smoke_test.png")
THRESHOLD = float(os.environ.get("PARITY_THRESHOLD", "0.3"))


def main() -> int:
    if not Path(SAMPLE_PATH).exists():
        print(f"[parity] sample not found: {SAMPLE_PATH}", file=sys.stderr)
        return 2

    print(f"[parity] loading {MODEL_DIR}", flush=True)
    model = PPDocLayoutV3ForObjectDetection.from_pretrained(MODEL_DIR).eval()
    processor = PPDocLayoutV3ImageProcessor.from_pretrained(MODEL_DIR)

    img = Image.open(SAMPLE_PATH).convert("RGB")
    print(f"[parity] sample: {SAMPLE_PATH} size={img.size}", flush=True)

    inputs = processor(images=[img], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes_torch = torch.tensor([img.size[::-1]])
    torch_results = processor.post_process_object_detection(
        outputs, threshold=THRESHOLD, target_sizes=target_sizes_torch,
    )[0]

    # Numpy path — pull same tensors out as numpy and call the port.
    logits_np = outputs.logits.detach().cpu().numpy()
    pred_boxes_np = outputs.pred_boxes.detach().cpu().numpy()
    order_logits_np = outputs.order_logits.detach().cpu().numpy()
    out_masks_np = outputs.out_masks.detach().cpu().numpy()
    target_sizes_np = np.array([img.size[::-1]], dtype=np.int64)

    np_results = np_post_process_object_detection(
        logits_np, pred_boxes_np, order_logits_np, out_masks_np,
        target_sizes=target_sizes_np,
        threshold=THRESHOLD,
        processor_size=processor.size,
    )[0]

    torch_scores = torch_results["scores"].detach().cpu().numpy()
    torch_labels = torch_results["labels"].detach().cpu().numpy()
    torch_boxes = torch_results["boxes"].detach().cpu().numpy()
    torch_order = torch_results["order_seq"].detach().cpu().numpy()

    np_scores = np.asarray(np_results["scores"])
    np_labels = np.asarray(np_results["labels"])
    np_boxes = np.asarray(np_results["boxes"])
    np_order = np.asarray(np_results["order_seq"])

    print(f"[parity] torch: {len(torch_scores)} detections")
    print(f"[parity] numpy: {len(np_scores)} detections")

    failed = False

    if len(torch_scores) != len(np_scores):
        print(f"[FAIL] detection count mismatch: torch={len(torch_scores)} np={len(np_scores)}")
        failed = True

    n = min(len(torch_scores), len(np_scores))
    # Elements are already sorted by order_seq (ascending) in both paths; compare pairwise.
    if n > 0:
        score_diff = np.abs(torch_scores[:n] - np_scores[:n]).max()
        label_match = (torch_labels[:n] == np_labels[:n]).all()
        box_diff = np.abs(torch_boxes[:n] - np_boxes[:n]).max()
        order_match = (torch_order[:n] == np_order[:n]).all()

        print(f"[parity] max |Δscore|     = {score_diff:.2e}")
        print(f"[parity] labels identical = {label_match}")
        print(f"[parity] max |Δbox|       = {box_diff:.4f} (image-space pixels)")
        print(f"[parity] order identical  = {order_match}")

        # Tolerances: scores are float32 sigmoid outputs; small numeric drift allowed.
        if score_diff > 1e-4:
            print(f"[FAIL] score delta {score_diff:.2e} exceeds 1e-4")
            failed = True
        if not label_match:
            print(f"[FAIL] labels differ at some positions")
            failed = True
        if box_diff > 1e-2:
            print(f"[FAIL] box delta {box_diff:.4f} exceeds 0.01 pixels")
            failed = True
        if not order_match:
            print(f"[FAIL] order_seq differs")
            failed = True

    if failed:
        print("[parity] FAILED on post_process_object_detection")
        return 1

    # --- Full pipeline parity: run glmocr's detector vs our numpy pipeline.
    print()
    print("[parity] --- full-pipeline comparison ---")
    try:
        from glmocr.layout import PPDocLayoutDetector
        from glmocr.config import LayoutConfig  # type: ignore
    except ImportError as e:
        print(f"[parity] skipping full-pipeline (import failed: {e})")
        print("[parity] PASSED (core post-proc only)")
        return 0

    try:
        cfg = LayoutConfig(
            model_dir=MODEL_DIR,
            device="cpu",
            threshold=THRESHOLD,
            batch_size=1,
            threshold_by_class={},
            layout_nms=True,
            layout_unclip_ratio=1.0,
            layout_merge_bboxes_mode="union",
            label_task_mapping={"ocr": list(model.config.id2label.values())},
        )
    except Exception as e:
        print(f"[parity] could not build LayoutConfig ({e}); skipping full pipeline")
        print("[parity] PASSED (core post-proc only)")
        return 0

    detector = PPDocLayoutDetector(cfg)
    detector._model = model
    detector._image_processor = processor
    detector._device = "cpu"
    # LayoutConfig doesn't auto-populate id2label; the live pipeline's
    # start() does that. Plug it in from the model config for the test.
    detector.id2label = model.config.id2label

    all_results_torch, _ = detector.process(
        [img], save_visualization=False, global_start_idx=0, use_polygon=False,
    )

    img_sizes_wh = [img.size]  # (width, height)
    pixel_values = processor(images=[img], return_tensors="np")["pixel_values"]

    def ort_like_run(pv):
        pv_t = torch.from_numpy(pv)
        with torch.no_grad():
            o = model(pixel_values=pv_t)
        return (
            o.logits.detach().cpu().numpy(),
            o.pred_boxes.detach().cpu().numpy(),
            o.order_logits.detach().cpu().numpy(),
            o.out_masks.detach().cpu().numpy(),
            o.last_hidden_state.detach().cpu().numpy(),
        )

    all_results_np = run_numpy_layout_pipeline(
        pixel_values=pixel_values,
        ort_run=ort_like_run,
        img_sizes_wh=img_sizes_wh,
        id2label=model.config.id2label,
        label_task_mapping=cfg.label_task_mapping,
        threshold=cfg.threshold,
        threshold_by_class=cfg.threshold_by_class,
        layout_nms=cfg.layout_nms,
        layout_unclip_ratio=cfg.layout_unclip_ratio,
        layout_merge_bboxes_mode=cfg.layout_merge_bboxes_mode,
        processor_size=processor.size,
    )

    t_detections = all_results_torch[0]
    n_detections = all_results_np[0]
    print(f"[parity] torch full: {len(t_detections)} detections")
    print(f"[parity] numpy full: {len(n_detections)} detections")

    if len(t_detections) != len(n_detections):
        print("[FAIL] full-pipeline detection count mismatch")
        return 1

    full_failed = False
    for i, (t, n) in enumerate(zip(t_detections, n_detections)):
        if t["label"] != n["label"]:
            print(f"[FAIL] det {i}: label {t['label']} vs {n['label']}"); full_failed = True
        if abs(t["score"] - n["score"]) > 1e-4:
            print(f"[FAIL] det {i}: score Δ {abs(t['score']-n['score']):.2e}"); full_failed = True
        if t["bbox_2d"] != n["bbox_2d"]:
            print(f"[FAIL] det {i}: bbox_2d {t['bbox_2d']} vs {n['bbox_2d']}"); full_failed = True
        if t["task_type"] != n["task_type"]:
            print(f"[FAIL] det {i}: task_type {t['task_type']} vs {n['task_type']}"); full_failed = True

    if full_failed:
        return 1
    print("[parity] full-pipeline PASSED")
    print("[parity] ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
