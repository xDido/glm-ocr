"""Host-runnable unit tests for the numpy layout post-processing port.

These tests are torch-free — they use synthetic tensors with hand-chosen
values so outputs are known in advance. Parity against the live torch
pipeline runs inside the container (see test_layout_parity_in_container.py).

Run from the repo root with:

    python -m pytest docker/cpu/tests/test_layout_postprocess.py -q

Requires numpy + opencv-python + pytest. No torch, no transformers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from layout_postprocess import (  # noqa: E402
    _iou,
    _is_contained,
    _nms,
    _np_get_order_seqs,
    _sigmoid,
    _unclip_boxes,
    np_apply_per_class_threshold,
    np_post_process_object_detection,
    run_numpy_layout_pipeline,
)


# ---------------------------------------------------------------------------
# Low-level math
# ---------------------------------------------------------------------------


def test_sigmoid_matches_closed_form():
    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32)
    expected = 1.0 / (1.0 + np.exp(-x))
    np.testing.assert_allclose(_sigmoid(x), expected, atol=1e-6)


def test_sigmoid_handles_extreme_values_without_overflow():
    x = np.array([-1000.0, 1000.0], dtype=np.float32)
    out = _sigmoid(x)
    assert out[0] == pytest.approx(0.0, abs=1e-30)
    assert out[1] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Reading-order decoder
# ---------------------------------------------------------------------------


def test_order_seqs_identity_when_all_zeros():
    # All-zero logits → sigmoid = 0.5 everywhere → symmetric votes → tied →
    # stable argsort returns original order (0, 1, 2) so rank at each
    # pointer is the identity permutation.
    logits = np.zeros((1, 4, 4), dtype=np.float32)
    out = _np_get_order_seqs(logits)
    np.testing.assert_array_equal(out[0], np.array([0, 1, 2, 3]))


def test_order_seqs_clear_preference():
    # Construct a clear ordering: query 0 should come first, then 1, then 2.
    # Pairwise score[i, j] = P(i before j). If i < j, high score; i > j, low.
    n = 3
    score = np.full((n, n), 0.5, dtype=np.float32)
    # Huge positive → sigmoid ~1; huge negative → sigmoid ~0
    score[0, 1] = 20.0
    score[0, 2] = 20.0
    score[1, 2] = 20.0
    score[1, 0] = -20.0
    score[2, 0] = -20.0
    score[2, 1] = -20.0
    out = _np_get_order_seqs(score[None, :, :])
    # Expected: order_seq[0]=0 (first), order_seq[1]=1, order_seq[2]=2
    np.testing.assert_array_equal(out[0], np.array([0, 1, 2]))


def test_order_seqs_reversed_preference():
    n = 3
    score = np.full((n, n), 0.5, dtype=np.float32)
    score[0, 1] = -20.0
    score[0, 2] = -20.0
    score[1, 2] = -20.0
    score[1, 0] = 20.0
    score[2, 0] = 20.0
    score[2, 1] = 20.0
    out = _np_get_order_seqs(score[None, :, :])
    # Expected reverse: order_seq = [2, 1, 0]
    np.testing.assert_array_equal(out[0], np.array([2, 1, 0]))


# ---------------------------------------------------------------------------
# Top-K + box decode + rescale
# ---------------------------------------------------------------------------


def _build_synthetic_detector_output(
    batch=1, num_queries=4, num_classes=3, mask_h=200, mask_w=200
):
    """Deterministic synthetic outputs with known top-1 detection at query 0."""
    # logits: make query 0, class 1 the clear winner. Others very negative.
    logits = np.full((batch, num_queries, num_classes), -20.0, dtype=np.float32)
    logits[0, 0, 1] = 10.0
    # pred_boxes: cxcywh in [0, 1]. Query 0 box centered at (0.5, 0.5), 0.4×0.4.
    pred_boxes = np.zeros((batch, num_queries, 4), dtype=np.float32)
    pred_boxes[0, 0] = [0.5, 0.5, 0.4, 0.4]
    # order_logits: identity order.
    order_logits = np.zeros((batch, num_queries, num_queries), dtype=np.float32)
    # out_masks: full-positive mask for query 0 class, zero elsewhere.
    out_masks = np.full((batch, num_queries, mask_h, mask_w), -10.0, dtype=np.float32)
    out_masks[0, 0] = 10.0
    return logits, pred_boxes, order_logits, out_masks


def test_post_process_picks_top_detection():
    logits, pred_boxes, order_logits, out_masks = _build_synthetic_detector_output()
    # Target image is 100 × 200 (H × W).
    target_sizes = np.array([[100, 200]], dtype=np.int64)
    out = np_post_process_object_detection(
        logits,
        pred_boxes,
        order_logits,
        out_masks,
        target_sizes=target_sizes,
        threshold=0.5,
        processor_size={"width": 800, "height": 800},
    )
    assert len(out) == 1
    r = out[0]
    # Only query 0 passes the 0.5 threshold (its sigmoid(10) ≈ 1).
    assert len(r["scores"]) == 1
    assert r["labels"][0] == 1
    # Box: cxcywh (0.5, 0.5, 0.4, 0.4) → xyxy (0.3, 0.3, 0.7, 0.7) →
    # scale by (W=200, H=100): (60, 30, 140, 70).
    np.testing.assert_allclose(r["boxes"][0], [60.0, 30.0, 140.0, 70.0], atol=1e-4)
    assert r["scores"][0] > 0.99


def test_post_process_threshold_drops_everything():
    logits, pred_boxes, order_logits, out_masks = _build_synthetic_detector_output()
    target_sizes = np.array([[100, 200]], dtype=np.int64)
    out = np_post_process_object_detection(
        logits,
        pred_boxes,
        order_logits,
        out_masks,
        target_sizes=target_sizes,
        threshold=0.999999,  # above even sigmoid(10) after rounding — well above
        processor_size={"width": 800, "height": 800},
    )
    # sigmoid(10) ≈ 0.99995 < 0.999999, so nothing passes.
    assert len(out[0]["scores"]) == 0


# ---------------------------------------------------------------------------
# Per-class threshold
# ---------------------------------------------------------------------------


def test_per_class_threshold_filters_correctly():
    # Two detections: label 0 @ 0.6, label 1 @ 0.7. Global threshold 0.5.
    # Per-class: label 0 → 0.8 (filter it out); label 1 → 0.5 (keep it).
    raw = [
        {
            "scores": np.array([0.6, 0.7], dtype=np.float32),
            "labels": np.array([0, 1], dtype=np.int64),
            "boxes": np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32),
            "order_seq": np.array([0, 1], dtype=np.int64),
            "polygon_points": [np.array([[0, 0]]), np.array([[20, 20]])],
        }
    ]
    id2label = {0: "text", 1: "title"}
    out = np_apply_per_class_threshold(
        raw,
        threshold=0.5,
        threshold_by_class={"text": 0.8, "title": 0.5},
        id2label=id2label,
    )
    assert len(out[0]["scores"]) == 1
    assert out[0]["labels"][0] == 1
    assert out[0]["scores"][0] == pytest.approx(0.7)
    assert len(out[0]["polygon_points"]) == 1


def test_per_class_threshold_unknown_class_ignored():
    raw = [
        {
            "scores": np.array([0.6], dtype=np.float32),
            "labels": np.array([0], dtype=np.int64),
            "boxes": np.array([[0, 0, 10, 10]], dtype=np.float32),
            "order_seq": np.array([0], dtype=np.int64),
            "polygon_points": [],
        }
    ]
    id2label = {0: "text"}
    out = np_apply_per_class_threshold(
        raw,
        threshold=0.5,
        threshold_by_class={"not_a_real_class": 0.99},
        id2label=id2label,
    )
    # Unknown class silently ignored → fallback to global 0.5 → kept.
    assert len(out[0]["scores"]) == 1


# ---------------------------------------------------------------------------
# NMS / IoU / containment / unclip
# ---------------------------------------------------------------------------


def test_iou_identical_boxes():
    assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_iou_disjoint():
    assert _iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_is_contained_nested():
    outer = [0, 0.9, 0, 0, 100, 100]
    inner = [0, 0.9, 10, 10, 20, 20]
    assert _is_contained(inner, outer)
    assert not _is_contained(outer, inner)


def test_nms_removes_duplicate_same_class():
    # Two near-identical same-class boxes (iou > 0.6) → lower score dropped.
    # boxes format: [cls_id, score, x1, y1, x2, y2]
    boxes = np.array(
        [
            [0, 0.9, 0, 0, 10, 10],
            [0, 0.5, 0, 0, 10, 10],  # duplicate, should drop
            [1, 0.8, 100, 100, 110, 110],  # different class, different place
        ],
        dtype=np.float32,
    )
    kept = _nms(boxes, iou_same=0.6, iou_diff=0.95)
    assert sorted(kept) == [0, 2]


def test_nms_keeps_different_classes_at_same_location():
    # iou_diff=0.95; two classes at identical box → iou=1.0 ≥ 0.95 → loser drops.
    boxes = np.array(
        [
            [0, 0.9, 0, 0, 10, 10],
            [1, 0.5, 0, 0, 10, 10],
        ],
        dtype=np.float32,
    )
    kept = _nms(boxes, iou_same=0.6, iou_diff=0.95)
    assert kept == [0]


def test_unclip_scalar_ratio_expands_symmetrically():
    boxes = np.array([[0, 0.9, 10.0, 10.0, 20.0, 20.0]], dtype=np.float32)
    # 1.5× ratio → width 10→15, height 10→15, centered at (15, 15)
    # new coords: (7.5, 7.5, 22.5, 22.5)
    out = _unclip_boxes(boxes, unclip_ratio=(1.5, 1.5))
    np.testing.assert_allclose(out[0, 2:6], [7.5, 7.5, 22.5, 22.5], atol=1e-4)


def test_unclip_no_ratio_is_identity():
    boxes = np.array([[0, 0.9, 1, 2, 3, 4]], dtype=np.float32)
    out = _unclip_boxes(boxes, unclip_ratio=None)
    np.testing.assert_array_equal(out, boxes)


# ---------------------------------------------------------------------------
# End-to-end orchestrator
# ---------------------------------------------------------------------------


def test_run_numpy_layout_pipeline_end_to_end():
    logits, pred_boxes, order_logits, out_masks = _build_synthetic_detector_output()
    last_hs = np.zeros((1, 4, 16), dtype=np.float32)

    def fake_ort_run(pixel_values):
        return logits, pred_boxes, order_logits, out_masks, last_hs

    pixel_values = np.zeros((1, 3, 800, 800), dtype=np.float32)
    img_sizes_wh = [(200, 100)]  # (width, height)
    id2label = {0: "abandon_label", 1: "text", 2: "unused"}
    label_task_mapping = {"ocr": ["text"]}

    out = run_numpy_layout_pipeline(
        pixel_values=pixel_values,
        ort_run=fake_ort_run,
        img_sizes_wh=img_sizes_wh,
        id2label=id2label,
        label_task_mapping=label_task_mapping,
        threshold=0.5,
        threshold_by_class={},
        layout_nms=True,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        processor_size={"width": 800, "height": 800},
    )
    assert len(out) == 1
    results = out[0]
    assert len(results) == 1
    item = results[0]
    assert item["label"] == "text"
    assert item["task_type"] == "ocr"
    assert item["index"] == 0
    # bbox_2d is normalized to 0–1000 over image size (200, 100).
    # Box in image coords: (60, 30, 140, 70) → normalized:
    #   x: 60/200*1000=300, 140/200*1000=700
    #   y: 30/100*1000=300, 70/100*1000=700
    assert item["bbox_2d"] == [300, 300, 700, 700]


def test_run_numpy_layout_pipeline_filters_abandon_task():
    # Only "abandon" task_type → every detection filtered out.
    logits, pred_boxes, order_logits, out_masks = _build_synthetic_detector_output()
    last_hs = np.zeros((1, 4, 16), dtype=np.float32)

    def fake_ort_run(pixel_values):
        return logits, pred_boxes, order_logits, out_masks, last_hs

    pixel_values = np.zeros((1, 3, 800, 800), dtype=np.float32)
    id2label = {0: "junk", 1: "footer", 2: "unused"}
    label_task_mapping = {"abandon": ["footer"]}

    out = run_numpy_layout_pipeline(
        pixel_values=pixel_values,
        ort_run=fake_ort_run,
        img_sizes_wh=[(200, 100)],
        id2label=id2label,
        label_task_mapping=label_task_mapping,
        threshold=0.5,
        threshold_by_class={},
        layout_nms=True,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        processor_size={"width": 800, "height": 800},
    )
    assert out == [[]]
