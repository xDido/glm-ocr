"""Paddle2ONNX backend for PP-DocLayoutV3 layout detection.

Drop-in alternative to the torch-exported graph in
``docker/cpu/layout_postprocess.py`` when ``LAYOUT_VARIANT=paddle2onnx``.

Why: the torch export from HF transformers bakes ``batch=1`` into several
Reshape initializers (see docs/OPTIMIZATIONS.md → "Pos-0 literal-1 → 0
global Reshape rewrite" and the OpenVINO rejected entry). The upstream
Paddle2ONNX export (``alex-dinh/PP-DocLayoutV3-ONNX`` on HF) has no baked
batch dim anywhere and runs cleanly at batch=1..8+ — same weights, clean
graph.

Contract:
    ``run_paddle_layout_pipeline(pil_images, sess, ...) -> all_results``
returns the same ``list[list[dict]]`` shape the numpy torch path produces
(``paddle_to_all_results`` output: 0-1000 normalized ``bbox_2d``, plus
``label``, ``score``, ``polygon``, ``task_type``, ``index``).

Pipeline (mirrors ``run_numpy_layout_pipeline`` structurally):

  PIL images
    ↓ resize 800×800 + RGB/255 + HWC→CHW          (Paddle preproc, from config.json)
  pixel tensor (B, 3, 800, 800)
    ↓ ORT run with 3 inputs: image, im_shape, scale_factor
  fetch_name_0 (N_det, 7) = [cls, score, x1, y1, x2, y2, ?]  (ragged across batch)
  fetch_name_1 (B,)       = per-image detection counts
    ↓ group by batch, rescale boxes from 800² to original image space
  raw_results: list of dicts with numpy {scores, labels, boxes, order_seq, polygon_points}
    ↓ np_apply_layout_postprocess  (NMS / merge / unclip — shared with torch path)
  paddle-format intermediate
    ↓ paddle_to_all_results  (task filter + 0-1000 normalization — shared with torch path)
  all_results

Preprocessing gotcha: ``config.json`` for this export lists
``mean=0, std=1, norm_type=none`` which is misleading — Paddle's C++ image
loader divides by 255 before ``NormalizeImage`` sees the tensor. Without
the /255, max score is ~0.01 (nothing passes threshold). Confirmed
empirically 2026-04-23.

Scale-factor gotcha: the ``scale_factor`` input to this export is
effectively a no-op — ``sf=(1,1)``, ``sf=(h/800, w/800)``, and
``sf=(800/h, 800/w)`` each give different outputs but none produce boxes
in original-image coords. The adapter rescales manually instead.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import numpy as np
from PIL import Image

from layout_postprocess import np_apply_layout_postprocess, paddle_to_all_results

# 25 classes, in the order Paddle's fetch_name_0[:, 0] uses.
# Source: https://huggingface.co/alex-dinh/PP-DocLayoutV3-ONNX/blob/main/config.json
PADDLE_LABELS: Tuple[str, ...] = (
    "abstract", "algorithm", "aside_text", "chart", "content",
    "display_formula", "doc_title", "figure_title", "footer",
    "footer_image", "footnote", "formula_number", "header",
    "header_image", "image", "inline_formula", "number",
    "paragraph_title", "reference", "reference_content", "seal",
    "table", "text", "vertical_text", "vision_footnote",
)

PADDLE_ID2LABEL: Dict[int, str] = {i: lab for i, lab in enumerate(PADDLE_LABELS)}
PADDLE_LABEL2ID: Dict[str, int] = {lab: i for i, lab in enumerate(PADDLE_LABELS)}

# Paddle's native input size per config.json Preprocess[0].target_size.
PADDLE_INPUT_H = 800
PADDLE_INPUT_W = 800


def _preprocess_batch(pil_images: List[Image.Image]) -> np.ndarray:
    """PIL → (B, 3, 800, 800) float32 RGB/255. Matches Paddle2ONNX
    expectations; see module docstring for the /255 gotcha."""
    batch: List[np.ndarray] = []
    for im in pil_images:
        if im.mode != "RGB":
            im = im.convert("RGB")
        resized = im.resize((PADDLE_INPUT_W, PADDLE_INPUT_H), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0  # HWC, [0,1]
        batch.append(arr.transpose(2, 0, 1))                 # CHW
    return np.stack(batch, axis=0)


def _build_session_inputs(
    pixel_tensor: np.ndarray,
    orig_sizes_wh: List[Tuple[int, int]],
) -> Dict[str, np.ndarray]:
    """Construct the 3-input feed. Names come from the Paddle2ONNX graph."""
    batch_size = pixel_tensor.shape[0]
    assert batch_size == len(orig_sizes_wh), "orig_sizes must match batch"

    # im_shape: original H x W per image (float32).
    im_shape = np.array(
        [[h, w] for (w, h) in orig_sizes_wh],
        dtype=np.float32,
    )
    # scale_factor: nominally (h_orig/h_in, w_orig/w_in); the export ignores
    # this numerically (verified empirically — sf=(1,1) and sf=(h/800,w/800)
    # produce identical detections). Pass the "right" values anyway so any
    # downstream use is semantically correct.
    scale_factor = np.array(
        [[h / PADDLE_INPUT_H, w / PADDLE_INPUT_W] for (w, h) in orig_sizes_wh],
        dtype=np.float32,
    )
    return {
        "image": pixel_tensor,
        "im_shape": im_shape,
        "scale_factor": scale_factor,
    }


def _ungroup_detections(
    fetch_0: np.ndarray,
    fetch_1: np.ndarray,
    batch_size: int,
    score_threshold: float,
) -> List[np.ndarray]:
    """Paddle returns all detections across the batch concatenated in
    ``fetch_0`` (shape ``(N_total, 7)``) with ``fetch_1`` giving the
    per-image counts (shape ``(B,)``).

    Returns a list of per-image detection arrays, each ``(N_img, 7)``,
    pre-filtered by score_threshold. Empty images get zero-row arrays.
    """
    counts = [int(c) for c in fetch_1.tolist()]
    if len(counts) != batch_size:
        # Some exports emit fetch_1 as [total] scalar — fall back to
        # treating everything as one image's worth if shapes disagree.
        counts = [int(fetch_0.shape[0])] + [0] * (batch_size - 1)

    out: List[np.ndarray] = []
    cursor = 0
    for n in counts:
        chunk = fetch_0[cursor : cursor + n]
        cursor += n
        if score_threshold > 0 and chunk.size:
            chunk = chunk[chunk[:, 1] >= score_threshold]
        out.append(chunk)
    return out


def _rescale_and_build_raw(
    per_image_dets: List[np.ndarray],
    orig_sizes_wh: List[Tuple[int, int]],
) -> List[Dict[str, np.ndarray]]:
    """Convert per-image (N, 7) ``[cls, score, x1, y1, x2, y2, ?]`` arrays
    in 800² space to the raw_results shape ``np_apply_layout_postprocess``
    expects (keys ``scores``, ``labels``, ``boxes``, ``order_seq``,
    ``polygon_points``), rescaling coords to original image space."""
    raw_results: List[Dict[str, np.ndarray]] = []
    for i, dets in enumerate(per_image_dets):
        w_orig, h_orig = orig_sizes_wh[i]
        sx = w_orig / PADDLE_INPUT_W
        sy = h_orig / PADDLE_INPUT_H

        if dets.size == 0:
            raw_results.append({
                "scores": np.zeros((0,), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64),
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "order_seq": np.zeros((0,), dtype=np.int64),
                "polygon_points": [],
            })
            continue

        labels = dets[:, 0].astype(np.int64)
        scores = dets[:, 1].astype(np.float32)
        x1 = dets[:, 2] * sx
        y1 = dets[:, 3] * sy
        x2 = dets[:, 4] * sx
        y2 = dets[:, 5] * sy
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        # Paddle2ONNX doesn't emit a reading-order head. Use detection
        # index as a stable sentinel — downstream layout_postprocess only
        # uses order_seq for sort stability after NMS, not for semantic
        # reading order. Any correctness-sensitive reading-order work
        # would need a separate pass.
        order_seq = np.arange(len(scores), dtype=np.int64)

        # Rectangular polygon per detection, clockwise from top-left.
        # paddle_to_all_results reads this for the "polygon" field in
        # the final json_result blocks.
        polygon_points = [
            [[float(x1[k]), float(y1[k])],
             [float(x2[k]), float(y1[k])],
             [float(x2[k]), float(y2[k])],
             [float(x1[k]), float(y2[k])]]
            for k in range(len(scores))
        ]

        raw_results.append({
            "scores": scores,
            "labels": labels,
            "boxes": boxes,
            "order_seq": order_seq,
            "polygon_points": polygon_points,
        })
    return raw_results


def run_paddle_layout_pipeline(
    pil_images: List[Image.Image],
    sess,
    *,
    label_task_mapping: Dict[str, Any],
    id2label: Dict[int, str] = None,
    threshold: float = 0.5,
    threshold_by_class: Dict[Union[str, int], float] = None,
    layout_nms: bool = True,
    layout_unclip_ratio=None,
    layout_merge_bboxes_mode=None,
    batch_size: int = 8,
) -> List[List[Dict[str, Any]]]:
    """Top-level entry point. Matches ``run_numpy_layout_pipeline`` signature
    shape so runtime_app.py can select between them via a flag.

    ``sess`` is a pre-loaded ``onnxruntime.InferenceSession`` for the
    ``alex-dinh/PP-DocLayoutV3-ONNX`` graph. Caller owns session lifetime.

    ``id2label``: caller should pass glmocr's native label dict
    (``ld._model.config.id2label``). Class IDs are identical between the
    torch and Paddle exports — same model weights — but the Paddle
    config.json uses more granular label names (``display_formula``,
    ``inline_formula``, ``footer_image``, ``header_image``,
    ``vertical_text``) that glmocr's ``label_task_mapping`` doesn't know,
    causing those blocks to be silently dropped by ``paddle_to_all_results``.
    Passing glmocr's dict makes paddle→torch label routing transparent.
    If unset, ``PADDLE_ID2LABEL`` is used — fine for standalone tests,
    but expect class-name drops in production."""
    if id2label is None:
        id2label = PADDLE_ID2LABEL
    threshold_by_class = threshold_by_class or {}
    orig_sizes_wh: List[Tuple[int, int]] = [img.size for img in pil_images]

    paddle_format_all: List[List[Dict[str, Any]]] = []

    for chunk_start in range(0, len(pil_images), batch_size):
        chunk = pil_images[chunk_start : chunk_start + batch_size]
        chunk_sizes = orig_sizes_wh[chunk_start : chunk_start + batch_size]
        pixel_tensor = _preprocess_batch(chunk)
        feed = _build_session_inputs(pixel_tensor, chunk_sizes)

        outputs = sess.run(None, feed)
        # Paddle2ONNX export emits 3 outputs named fetch_name_0..2 —
        # lock to positional indexing so a future rename doesn't break us.
        fetch_0, fetch_1 = outputs[0], outputs[1]
        # fetch_2 is the masks tensor; unused today (we build rectangular
        # polygons from boxes), kept here as documentation.

        # Pre-threshold to cut the NMS input; glmocr applies its own
        # per-class threshold inside np_apply_layout_postprocess.
        pre_threshold = threshold
        if threshold_by_class:
            pre_threshold = min(threshold, min(threshold_by_class.values()))

        per_image = _ungroup_detections(
            fetch_0, fetch_1,
            batch_size=len(chunk),
            score_threshold=pre_threshold,
        )
        raw = _rescale_and_build_raw(per_image, chunk_sizes)
        # Down below, np_apply_layout_postprocess will use id2label to
        # resolve class IDs → strings. Because class IDs are identical
        # between the Paddle and torch exports, passing glmocr's dict here
        # routes everything through glmocr's naming convention correctly.
        paddle_chunk = np_apply_layout_postprocess(
            raw_results=raw,
            id2label=id2label,
            img_sizes=chunk_sizes,
            layout_nms=layout_nms,
            layout_unclip_ratio=layout_unclip_ratio,
            layout_merge_bboxes_mode=layout_merge_bboxes_mode,
        )
        paddle_format_all.extend(paddle_chunk)

    return paddle_to_all_results(
        paddle_format_all,
        orig_sizes_wh,
        label_task_mapping,
    )
