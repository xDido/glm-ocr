"""Numpy reimplementation of PP-DocLayoutV3's post-processing pipeline.

Replaces the torch-backed path that lives across three upstream sites
(transformers' PPDocLayoutV3ImageProcessor.post_process_object_detection,
glmocr.layout.PPDocLayoutDetector._apply_per_class_threshold, and
glmocr.utils.layout_postprocess_utils.apply_layout_postprocess) with a
single numpy pipeline. This lets runtime_app.py drop torch from the
request path entirely once LAYOUT_POSTPROC=numpy is selected.

The torch-math subset (sigmoid, top-K over flattened (N, C), gather,
cxcywh→xyxy, target-size rescale, pairwise-vote reading-order decoder,
per-query mask threshold + polygon extraction, per-class threshold
filter) is ported here. Everything downstream (NMS, unclip, merge,
paddle-format conversion) is already numpy upstream; we vendor-copy it
verbatim with the `.cpu().numpy()` hops stripped.

Semantic parity targets match upstream bit-for-bit except where stated:

- Top-K over flattened scores uses numpy stable argsort, which
  tie-breaks by input index; torch's top-K tie-break is
  implementation-defined. Since scores are sigmoid floats over disjoint
  query/class slots, exact ties are vanishingly rare and do not affect
  the post-threshold output.
- Order-decoder scatter matches torch's behaviour exactly (the
  triu/tril partial-sum formulation is deterministic).
- Polygon extraction uses cv2 (identical to upstream).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Low-level math
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable branchless form. Avoids overflow for very negative/positive logits.
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos].astype(np.float32)))
    ex = np.exp(x[neg].astype(np.float32))
    out[neg] = ex / (1.0 + ex)
    return out


def _np_get_order_seqs(order_logits: np.ndarray) -> np.ndarray:
    """Port of PPDocLayoutV3ImageProcessor._get_order_seqs.

    Upstream (torch):
        order_scores = sigmoid(order_logits)                # (B, N, N)
        votes = (order_scores.triu(diag=1).sum(dim=1)
                 + (1 - order_scores.T).tril(diag=-1).sum(dim=1))
        pointers = argsort(votes, dim=1)                    # (B, N) asc
        order_seq.scatter_(1, pointers, ranks)              # rank at pointer

    The vote per query j is: (how many i < j prefer order[i] < order[j])
    plus (how many i > j prefer order[j] < order[i]). Lower votes → earlier.
    """
    order_scores = _sigmoid(order_logits)
    batch_size, n, _ = order_scores.shape

    # triu(diagonal=1): rows i, cols j with i < j (strict upper).
    triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    # tril(diagonal=-1) on transpose(1,2) picks i > j positions of original.
    tril_mask = np.tril(np.ones((n, n), dtype=bool), k=-1)

    triu_part = np.where(triu_mask[None, :, :], order_scores, 0.0).sum(axis=1)
    scores_T = np.transpose(order_scores, (0, 2, 1))
    tril_part = np.where(tril_mask[None, :, :], 1.0 - scores_T, 0.0).sum(axis=1)
    votes = triu_part + tril_part  # (B, N)

    pointers = np.argsort(votes, axis=1, kind="stable")  # ascending
    order_seq = np.empty_like(pointers)
    ranks = np.broadcast_to(np.arange(n, dtype=pointers.dtype), (batch_size, n))
    np.put_along_axis(order_seq, pointers, ranks, axis=1)
    return order_seq


# ---------------------------------------------------------------------------
# Polygon extraction (cv2 + numpy; direct port of upstream processor helpers)
# ---------------------------------------------------------------------------


def _extract_custom_vertices(polygon: np.ndarray,
                             sharp_angle_thresh: float = 45.0) -> List[Tuple[float, float]]:
    # Verbatim port of PPDocLayoutV3ImageProcessor.extract_custom_vertices.
    poly = np.array(polygon)
    n = len(poly)
    res: List[Tuple[float, float]] = []
    i = 0
    while i < n:
        previous_point = poly[(i - 1) % n]
        current_point = poly[i]
        next_point = poly[(i + 1) % n]
        vector_1 = previous_point - current_point
        vector_2 = next_point - current_point
        cross = vector_1[1] * vector_2[0] - vector_1[0] * vector_2[1]
        if cross < 0:
            denom = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
            if denom == 0:
                res.append(tuple(current_point))
                i += 1
                continue
            angle_cos = np.clip((vector_1 @ vector_2) / denom, -1.0, 1.0)
            angle = np.degrees(np.arccos(angle_cos))
            if abs(angle - sharp_angle_thresh) < 1:
                dir_vec = (vector_1 / np.linalg.norm(vector_1)
                           + vector_2 / np.linalg.norm(vector_2))
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm == 0:
                    res.append(tuple(current_point))
                else:
                    dir_vec = dir_vec / dir_norm
                    step_size = (np.linalg.norm(vector_1)
                                 + np.linalg.norm(vector_2)) / 2
                    new_point = current_point + dir_vec * step_size
                    res.append(tuple(new_point))
            else:
                res.append(tuple(current_point))
        i += 1
    return res


def _mask2polygon(mask: np.ndarray, epsilon_ratio: float = 0.004):
    # Verbatim port of PPDocLayoutV3ImageProcessor._mask2polygon.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    approx_contours = cv2.approxPolyDP(contour, epsilon, True)
    polygon_points = approx_contours.squeeze()
    polygon_points = np.atleast_2d(polygon_points)
    return _extract_custom_vertices(polygon_points)


def _extract_polygon_points_by_masks(boxes: np.ndarray,
                                     masks: np.ndarray,
                                     scale_ratio: Tuple[float, float]) -> List[np.ndarray]:
    """Port of PPDocLayoutV3ImageProcessor._extract_polygon_points_by_masks.

    boxes: (N, 4) xyxy in image space.
    masks: (N, Hm, Wm) int.
    scale_ratio: (width_ratio, height_ratio) = processor_size / target_size.
    """
    scale_width = scale_ratio[0] / 4
    scale_height = scale_ratio[1] / 4
    mask_height, mask_width = masks.shape[1:]
    polygon_points = []

    for i in range(len(boxes)):
        x_min, y_min, x_max, y_max = boxes[i].astype(np.int32)
        box_w, box_h = int(x_max - x_min), int(y_max - y_min)

        rect = np.array(
            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
            dtype=np.float32,
        )
        if box_w <= 0 or box_h <= 0:
            polygon_points.append(rect)
            continue

        x_coords = [int(round(float(x_min) * scale_width)),
                    int(round(float(x_max) * scale_width))]
        x_start, x_end = np.clip(x_coords, 0, mask_width)
        y_coords = [int(round(float(y_min) * scale_height)),
                    int(round(float(y_max) * scale_height))]
        y_start, y_end = np.clip(y_coords, 0, mask_height)
        cropped_mask = masks[i, y_start:y_end, x_start:x_end]

        resized_mask = cv2.resize(
            cropped_mask.astype(np.uint8), (box_w, box_h),
            interpolation=cv2.INTER_NEAREST,
        )

        polygon = _mask2polygon(resized_mask)
        if polygon is not None and len(polygon) < 4:
            polygon_points.append(rect)
            continue
        if polygon is not None and len(polygon) > 0:
            polygon = np.array(polygon) + np.array([x_min, y_min])

        polygon_points.append(polygon)
    return polygon_points


# ---------------------------------------------------------------------------
# Raw post-processing (sigmoid, top-K, box decode, rescale, gather, sort)
# ---------------------------------------------------------------------------


def np_post_process_object_detection(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    order_logits: np.ndarray,
    out_masks: np.ndarray,
    target_sizes: np.ndarray,
    threshold: float,
    processor_size: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Numpy port of PPDocLayoutV3ImageProcessor.post_process_object_detection.

    Inputs are the raw ONNX outputs. Returns a list (one per image) of dicts
    containing numpy arrays for scores/labels/boxes/order_seq plus the
    polygon_points list produced by cv2.
    """
    batch, num_queries, num_classes = logits.shape

    # Reading-order decoder (B, N).
    order_seqs = _np_get_order_seqs(order_logits)

    # cxcywh → xyxy and rescale to target image coords.
    centers = pred_boxes[..., :2]
    dims = pred_boxes[..., 2:]
    boxes_xyxy = np.concatenate([centers - 0.5 * dims, centers + 0.5 * dims], axis=-1)
    img_h = np.asarray(target_sizes)[:, 0].astype(boxes_xyxy.dtype)
    img_w = np.asarray(target_sizes)[:, 1].astype(boxes_xyxy.dtype)
    scale = np.stack([img_w, img_h, img_w, img_h], axis=1)  # (B, 4)
    boxes_xyxy = boxes_xyxy * scale[:, None, :]

    # Sigmoid over (B, N, C), flatten, stable sort descending, take top N.
    num_top = num_queries
    scores_full = _sigmoid(logits)
    flat = scores_full.reshape(batch, -1)
    order_idx = np.argsort(-flat, axis=-1, kind="stable")  # (B, N*C)
    indices = order_idx[:, :num_top]
    row = np.arange(batch)[:, None]
    scores_topk = flat[row, indices].astype(np.float32)
    labels = (indices % num_classes).astype(np.int64)
    query_idx = (indices // num_classes).astype(np.int64)

    # Gather boxes: (B, num_top, 4)
    boxes_topk = np.take_along_axis(
        boxes_xyxy,
        np.broadcast_to(query_idx[..., None], (batch, num_top, 4)),
        axis=1,
    )

    # Gather masks: (B, num_top, Hm, Wm), then sigmoid + threshold.
    mask_h, mask_w = out_masks.shape[-2:]
    masks_topk = np.take_along_axis(
        out_masks,
        np.broadcast_to(query_idx[..., None, None], (batch, num_top, mask_h, mask_w)),
        axis=1,
    )
    masks_topk = (_sigmoid(masks_topk) > threshold).astype(np.int32)

    # Gather order_seqs: (B, num_top)
    order_topk = np.take_along_axis(order_seqs, query_idx, axis=1)

    target_sizes = np.asarray(target_sizes)
    results: List[Dict[str, Any]] = []
    for b in range(batch):
        score_b = scores_topk[b]
        label_b = labels[b]
        box_b = boxes_topk[b]
        order_b = order_topk[b]
        mask_b = masks_topk[b]
        target_size = target_sizes[b]

        keep = score_b >= threshold
        filt_order = order_b[keep]
        sort_idx = np.argsort(filt_order, kind="stable")

        filt_boxes = box_b[keep][sort_idx]
        filt_masks = mask_b[keep][sort_idx]
        scale_ratio = (
            processor_size["width"] / float(target_size[1]),
            processor_size["height"] / float(target_size[0]),
        )
        polygon_points = _extract_polygon_points_by_masks(filt_boxes, filt_masks, scale_ratio)

        results.append({
            "scores": score_b[keep][sort_idx],
            "labels": label_b[keep][sort_idx],
            "boxes": filt_boxes,
            "polygon_points": polygon_points,
            "order_seq": filt_order[sort_idx],
        })
    return results


# ---------------------------------------------------------------------------
# Per-class threshold filter
# ---------------------------------------------------------------------------


def np_apply_per_class_threshold(
    raw_results: List[Dict[str, Any]],
    threshold: float,
    threshold_by_class: Dict[Union[str, int], float],
    id2label: Dict[int, str],
) -> List[Dict[str, Any]]:
    """Port of PPDocLayoutDetector._apply_per_class_threshold."""
    label2id = {name: int(cls_id) for cls_id, name in id2label.items()}
    class_thresholds: Dict[int, float] = {}
    for key, value in threshold_by_class.items():
        if isinstance(key, str):
            if key in label2id:
                class_thresholds[label2id[key]] = float(value)
            # Unknown class names are silently ignored; upstream logs a warning,
            # but the behaviour on the kept-detections list is identical.
        else:
            class_thresholds[int(key)] = float(value)

    filtered: List[Dict[str, Any]] = []
    for result in raw_results:
        scores = result["scores"]
        labels = result["labels"]
        thresholds = np.full_like(scores, threshold, dtype=scores.dtype)
        for cls_id, t in class_thresholds.items():
            thresholds[labels == cls_id] = t
        keep = scores >= thresholds

        new: Dict[str, Any] = {
            "scores": scores[keep],
            "labels": labels[keep],
            "boxes": result["boxes"][keep],
        }
        if "order_seq" in result:
            new["order_seq"] = result["order_seq"][keep]
        if "polygon_points" in result:
            keep_list = keep.tolist()
            new["polygon_points"] = [
                p for p, k in zip(result["polygon_points"], keep_list) if k
            ]
        filtered.append(new)
    return filtered


# ---------------------------------------------------------------------------
# Vendored from glmocr.utils.layout_postprocess_utils (already numpy upstream —
# copy verbatim, strip the two `.cpu().numpy()` hops that assumed torch inputs).
# ---------------------------------------------------------------------------


def _iou(box1, box2) -> float:
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    x1_i = max(x1, x1_p)
    y1_i = max(y1, y1_p)
    x2_i = min(x2, x2_p)
    y2_i = min(y2, y2_p)
    inter_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    return inter_area / float(box1_area + box2_area - inter_area)


def _is_contained(box1, box2) -> bool:
    _, _, x1, y1, x2, y2 = box1
    _, _, x1_p, y1_p, x2_p, y2_p = box2
    box1_area = (x2 - x1) * (y2 - y1)
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    iw = max(0, xi2 - xi1)
    ih = max(0, yi2 - yi1)
    inter_area = iw * ih
    iou_val = inter_area / box1_area if box1_area > 0 else 0
    return iou_val >= 0.8


def _nms(boxes: np.ndarray, iou_same: float = 0.6, iou_diff: float = 0.95) -> List[int]:
    scores = boxes[:, 1]
    indices = np.argsort(scores)[::-1]
    selected: List[int] = []
    while len(indices) > 0:
        current = indices[0]
        current_box = boxes[current]
        current_class = current_box[0]
        current_coords = current_box[2:]
        selected.append(int(current))
        indices = indices[1:]
        filtered: List[int] = []
        for i in indices:
            box = boxes[i]
            box_class = box[0]
            box_coords = box[2:]
            iou_value = _iou(current_coords, box_coords)
            thr = iou_same if current_class == box_class else iou_diff
            if iou_value < thr:
                filtered.append(int(i))
        indices = np.array(filtered, dtype=int) if filtered else np.array([], dtype=int)
    return selected


def _check_containment(boxes: np.ndarray, preserve_indices=None,
                       category_index=None, mode=None):
    n = len(boxes)
    contains_other = np.zeros(n, dtype=int)
    contained_by_other = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if preserve_indices is not None and boxes[i][0] in preserve_indices:
                continue
            if category_index is not None and mode is not None:
                if mode == "large" and boxes[j][0] == category_index:
                    if _is_contained(boxes[i], boxes[j]):
                        contained_by_other[i] = 1
                        contains_other[j] = 1
                if mode == "small" and boxes[i][0] == category_index:
                    if _is_contained(boxes[i], boxes[j]):
                        contained_by_other[i] = 1
                        contains_other[j] = 1
            else:
                if _is_contained(boxes[i], boxes[j]):
                    contained_by_other[i] = 1
                    contains_other[j] = 1
    return contains_other, contained_by_other


def _unclip_boxes(boxes: np.ndarray, unclip_ratio=None) -> np.ndarray:
    if unclip_ratio is None:
        return boxes
    if isinstance(unclip_ratio, dict):
        expanded = []
        for box in boxes:
            class_id, score, x1, y1, x2, y2 = box[:6]
            if class_id in unclip_ratio:
                width_ratio, height_ratio = unclip_ratio[class_id]
                width = x2 - x1
                height = y2 - y1
                new_w = width * width_ratio
                new_h = height * height_ratio
                cx = x1 + width / 2
                cy = y1 + height / 2
                new_box = [class_id, score,
                           cx - new_w / 2, cy - new_h / 2,
                           cx + new_w / 2, cy + new_h / 2]
                if len(box) > 6:
                    new_box.extend(box[6:])
                expanded.append(new_box)
            else:
                expanded.append(box)
        return np.array(expanded)
    widths = boxes[:, 4] - boxes[:, 2]
    heights = boxes[:, 5] - boxes[:, 3]
    new_w = widths * unclip_ratio[0]
    new_h = heights * unclip_ratio[1]
    cx = boxes[:, 2] + widths / 2
    cy = boxes[:, 3] + heights / 2
    expanded = np.column_stack(
        (boxes[:, 0], boxes[:, 1],
         cx - new_w / 2, cy - new_h / 2,
         cx + new_w / 2, cy + new_h / 2)
    )
    if boxes.shape[1] > 6:
        expanded = np.column_stack((expanded, boxes[:, 6:]))
    return expanded


def np_apply_layout_postprocess(
    raw_results: List[Dict[str, Any]],
    id2label: Dict[int, str],
    img_sizes: List[Tuple[int, int]],
    layout_nms: bool = True,
    layout_unclip_ratio=None,
    layout_merge_bboxes_mode=None,
) -> List[List[Dict[str, Any]]]:
    """Port of glmocr.utils.layout_postprocess_utils.apply_layout_postprocess.

    Upstream takes torch tensors and calls `.cpu().numpy()` internally.
    Our inputs are numpy already; everything else is identical.
    """
    all_labels = list(id2label.values())
    paddle_format_results: List[List[Dict[str, Any]]] = []

    for img_idx, result in enumerate(raw_results):
        scores = np.asarray(result["scores"])
        labels = np.asarray(result["labels"])
        boxes = np.asarray(result["boxes"])
        order_seq = np.asarray(result["order_seq"])
        polygon_points = result.get("polygon_points", [])
        img_size = img_sizes[img_idx]

        boxes_with_order = []
        for i in range(len(scores)):
            cls_id = int(labels[i])
            score = float(scores[i])
            x1, y1, x2, y2 = boxes[i]
            order = int(order_seq[i])
            boxes_with_order.append([cls_id, score, x1, y1, x2, y2, order])

        if not boxes_with_order:
            paddle_format_results.append([])
            continue

        boxes_array = np.array(boxes_with_order)

        if layout_nms:
            selected = _nms(boxes_array[:, :6], iou_same=0.6, iou_diff=0.98)
            boxes_array = boxes_array[selected]

        # Filter oversized "image" labels (>= 82%/93% of page area).
        filter_large_image = True
        if filter_large_image and len(boxes_array) > 1:
            area_thres = 0.82 if img_size[0] > img_size[1] else 0.93
            image_index = all_labels.index("image") if "image" in all_labels else None
            img_area = img_size[0] * img_size[1]
            filtered_boxes = []
            for box in boxes_array:
                label_index, _score, xmin, ymin, xmax, ymax = box[:6]
                if label_index == image_index:
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(img_size[0], xmax)
                    ymax = min(img_size[1], ymax)
                    box_area = (xmax - xmin) * (ymax - ymin)
                    if box_area <= area_thres * img_area:
                        filtered_boxes.append(box)
                else:
                    filtered_boxes.append(box)
            if filtered_boxes:
                boxes_array = np.array(filtered_boxes)

        if layout_merge_bboxes_mode:
            preserve_labels = ["image", "seal", "chart"]
            preserve_indices = set()
            for label in preserve_labels:
                if label in all_labels:
                    preserve_indices.add(all_labels.index(label))

            if isinstance(layout_merge_bboxes_mode, str):
                assert layout_merge_bboxes_mode in ("union", "large", "small"), (
                    f"layout_merge_bboxes_mode must be one of "
                    f"['union', 'large', 'small'], got {layout_merge_bboxes_mode}"
                )
                if layout_merge_bboxes_mode != "union":
                    contains_other, contained_by_other = _check_containment(
                        boxes_array[:, :6], preserve_indices
                    )
                    if layout_merge_bboxes_mode == "large":
                        boxes_array = boxes_array[contained_by_other == 0]
                    elif layout_merge_bboxes_mode == "small":
                        boxes_array = boxes_array[
                            (contains_other == 0) | (contained_by_other == 1)
                        ]
            elif isinstance(layout_merge_bboxes_mode, dict):
                keep_mask = np.ones(len(boxes_array), dtype=bool)
                for category_index, layout_mode in layout_merge_bboxes_mode.items():
                    assert layout_mode in ("union", "large", "small")
                    if layout_mode == "union":
                        continue
                    contains_other, contained_by_other = _check_containment(
                        boxes_array[:, :6], preserve_indices,
                        category_index, mode=layout_mode,
                    )
                    if layout_mode == "large":
                        keep_mask &= contained_by_other == 0
                    elif layout_mode == "small":
                        keep_mask &= (contains_other == 0) | (contained_by_other == 1)
                boxes_array = boxes_array[keep_mask]

        if len(boxes_array) == 0:
            paddle_format_results.append([])
            continue

        sorted_idx = np.argsort(boxes_array[:, 6])
        boxes_array = boxes_array[sorted_idx]

        if layout_unclip_ratio:
            if isinstance(layout_unclip_ratio, float):
                layout_unclip_ratio = (layout_unclip_ratio, layout_unclip_ratio)
            elif isinstance(layout_unclip_ratio, (tuple, list)):
                assert len(layout_unclip_ratio) == 2
            elif not isinstance(layout_unclip_ratio, dict):
                raise ValueError(
                    f"layout_unclip_ratio must be float/tuple/dict, "
                    f"got {type(layout_unclip_ratio)}"
                )
            boxes_array = _unclip_boxes(boxes_array, layout_unclip_ratio)

        img_width, img_height = img_size
        image_results: List[Dict[str, Any]] = []
        for i, box_data in enumerate(boxes_array):
            cls_id = int(box_data[0])
            score = float(box_data[1])
            x1, y1, x2, y2 = box_data[2:6]
            order = int(box_data[6]) if box_data[6] > 0 else None
            label_name = id2label.get(cls_id, f"class_{cls_id}")

            x1 = max(0, min(float(x1), img_width))
            y1 = max(0, min(float(y1), img_height))
            x2 = max(0, min(float(x2), img_width))
            y2 = max(0, min(float(y2), img_height))
            if x1 >= x2 or y1 >= y2:
                continue

            poly = None
            if len(polygon_points) > 0:
                for orig_idx in range(len(boxes)):
                    if np.allclose(boxes[orig_idx], box_data[2:6], atol=1.0):
                        if orig_idx < len(polygon_points):
                            candidate = polygon_points[orig_idx]
                            if candidate is not None:
                                poly = np.asarray(candidate).astype(np.float32)
                        break
            if poly is None:
                poly = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
                )
            else:
                poly[:, 0] = np.clip(poly[:, 0], 0, img_width)
                poly[:, 1] = np.clip(poly[:, 1], 0, img_height)

            image_results.append({
                "cls_id": cls_id,
                "label": label_name,
                "score": score,
                "coordinate": [int(x1), int(y1), int(x2), int(y2)],
                "order": order,
                "polygon_points": poly,
            })

        paddle_format_results.append(image_results)

    return paddle_format_results


# ---------------------------------------------------------------------------
# Top-level orchestrators — mirror PPDocLayoutDetector.process but numpy-only.
# ---------------------------------------------------------------------------


def compute_paddle_format_results(
    pixel_values: np.ndarray,
    ort_run,
    img_sizes_wh: List[Tuple[int, int]],
    *,
    id2label: Dict[int, str],
    threshold: float,
    threshold_by_class: Dict[Union[str, int], float],
    layout_nms: bool,
    layout_unclip_ratio,
    layout_merge_bboxes_mode,
    processor_size: Dict[str, int],
) -> List[List[Dict[str, Any]]]:
    """Run ORT + numpy post-proc through apply_layout_postprocess.

    Returns the paddle-format intermediate (list-of-list of dicts with
    keys cls_id/label/score/coordinate/order/polygon_points). This is what
    glmocr's visualization helper `draw_layout_boxes` consumes.
    """
    logits, pred_boxes, order_logits, out_masks, _last_hs = ort_run(pixel_values)
    target_sizes = np.array([(h, w) for (w, h) in img_sizes_wh], dtype=np.int64)

    if threshold_by_class:
        pre_threshold = min(threshold, min(threshold_by_class.values()))
    else:
        pre_threshold = threshold

    raw_results = np_post_process_object_detection(
        logits, pred_boxes, order_logits, out_masks,
        target_sizes=target_sizes,
        threshold=pre_threshold,
        processor_size=processor_size,
    )

    if threshold_by_class:
        raw_results = np_apply_per_class_threshold(
            raw_results, threshold, threshold_by_class, id2label,
        )

    return np_apply_layout_postprocess(
        raw_results=raw_results,
        id2label=id2label,
        img_sizes=img_sizes_wh,
        layout_nms=layout_nms,
        layout_unclip_ratio=layout_unclip_ratio,
        layout_merge_bboxes_mode=layout_merge_bboxes_mode,
    )


def paddle_to_all_results(
    paddle_format_results: List[List[Dict[str, Any]]],
    img_sizes_wh: List[Tuple[int, int]],
    label_task_mapping: Dict[str, Any],
) -> List[List[Dict[str, Any]]]:
    """Convert paddle-format to glmocr's public `all_results` shape.

    Applies the task_type filter and produces 0–1000 normalized coords,
    mirroring the tail of PPDocLayoutDetector.process.
    """
    all_results: List[List[Dict[str, Any]]] = []
    for img_idx, paddle_results in enumerate(paddle_format_results):
        img_width, img_height = img_sizes_wh[img_idx]
        results: List[Dict[str, Any]] = []
        valid_index = 0
        for item in paddle_results:
            label = item["label"]
            score = item["score"]
            box = item["coordinate"]
            task_type = None
            for task_item, labels in label_task_mapping.items():
                if isinstance(labels, list) and label in labels:
                    task_type = task_item
                    break
            if task_type is None or task_type == "abandon":
                continue
            x1, y1, x2, y2 = box
            x1n = int(float(x1) / img_width * 1000)
            y1n = int(float(y1) / img_height * 1000)
            x2n = int(float(x2) / img_width * 1000)
            y2n = int(float(y2) / img_height * 1000)
            poly_array = item["polygon_points"]
            polygon = [
                [int(float(p[0]) / img_width * 1000),
                 int(float(p[1]) / img_height * 1000)]
                for p in poly_array
            ]
            results.append({
                "index": valid_index,
                "label": label,
                "score": float(score),
                "bbox_2d": [x1n, y1n, x2n, y2n],
                "polygon": polygon,
                "task_type": task_type,
            })
            valid_index += 1
        all_results.append(results)
    return all_results


def run_numpy_layout_pipeline(
    pixel_values: np.ndarray,
    ort_run,
    img_sizes_wh: List[Tuple[int, int]],
    *,
    id2label: Dict[int, str],
    label_task_mapping: Dict[str, Any],
    threshold: float,
    threshold_by_class: Dict[Union[str, int], float],
    layout_nms: bool,
    layout_unclip_ratio,
    layout_merge_bboxes_mode,
    processor_size: Dict[str, int],
    global_start_idx: int = 0,
) -> List[List[Dict[str, Any]]]:
    """Convenience wrapper: paddle-format → all_results in one call."""
    paddle_format_results = compute_paddle_format_results(
        pixel_values=pixel_values,
        ort_run=ort_run,
        img_sizes_wh=img_sizes_wh,
        id2label=id2label,
        threshold=threshold,
        threshold_by_class=threshold_by_class,
        layout_nms=layout_nms,
        layout_unclip_ratio=layout_unclip_ratio,
        layout_merge_bboxes_mode=layout_merge_bboxes_mode,
        processor_size=processor_size,
    )
    return paddle_to_all_results(
        paddle_format_results, img_sizes_wh, label_task_mapping,
    )


# ---------------------------------------------------------------------------
# Fused-graph orchestrator (Phase 2) — the deterministic-arithmetic subset of
# post-proc lives in ONNX now. The numpy side does only:
#   - mask sigmoid + threshold (cheap; trivially vectorized)
#   - per-image score-threshold filter (data-dependent shape)
#   - sort by order_seq
#   - polygon extraction via cv2 (can't live in ONNX)
#   - per-class threshold filter + apply_layout_postprocess (reuse Phase-1 code)
# ---------------------------------------------------------------------------


def _post_process_from_fused(
    scores_topk: np.ndarray,
    labels_topk: np.ndarray,
    boxes_topk: np.ndarray,
    order_seq_topk: np.ndarray,
    masks_topk_logits: np.ndarray,
    target_sizes: np.ndarray,
    threshold: float,
    processor_size: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Numpy tail after the fused ONNX graph.

    Inputs match the fused graph's outputs (all pre-threshold, per-query
    top-K results of fixed shape). This function does only the
    data-dependent steps: score threshold filter, sort by order_seq,
    mask binarization, polygon extraction via cv2.
    """
    batch = scores_topk.shape[0]
    # Mask sigmoid + threshold — we kept this out of the graph because it's
    # vanishingly cheap and keeping masks as fp32 logits halves the fused
    # output volume in the I/O boundary case.
    masks_bool = (_sigmoid(masks_topk_logits) > threshold).astype(np.int32)

    target_sizes = np.asarray(target_sizes)
    results: List[Dict[str, Any]] = []
    for b in range(batch):
        score_b = scores_topk[b]
        label_b = labels_topk[b]
        box_b = boxes_topk[b]
        order_b = order_seq_topk[b]
        mask_b = masks_bool[b]
        target_size = target_sizes[b]

        keep = score_b >= threshold
        filt_order = order_b[keep]
        sort_idx = np.argsort(filt_order, kind="stable")

        filt_boxes = box_b[keep][sort_idx]
        filt_masks = mask_b[keep][sort_idx]
        scale_ratio = (
            processor_size["width"] / float(target_size[1]),
            processor_size["height"] / float(target_size[0]),
        )
        polygon_points = _extract_polygon_points_by_masks(
            filt_boxes, filt_masks, scale_ratio,
        )

        results.append({
            "scores": score_b[keep][sort_idx],
            "labels": label_b[keep][sort_idx],
            "boxes": filt_boxes,
            "polygon_points": polygon_points,
            "order_seq": filt_order[sort_idx],
        })
    return results


def compute_paddle_format_results_from_fused(
    pixel_values: np.ndarray,
    ort_run_fused,
    img_sizes_wh: List[Tuple[int, int]],
    *,
    id2label: Dict[int, str],
    threshold: float,
    threshold_by_class: Dict[Union[str, int], float],
    layout_nms: bool,
    layout_unclip_ratio,
    layout_merge_bboxes_mode,
    processor_size: Dict[str, int],
) -> List[List[Dict[str, Any]]]:
    """Run the fused ONNX graph + numpy tail through apply_layout_postprocess.

    `ort_run_fused` takes (pixel_values, target_sizes) and returns the
    6-tuple from the fused graph:
        (scores_topk, labels_topk, boxes_topk, order_seq_topk,
         masks_topk_logits, last_hidden_state)
    """
    target_sizes = np.array(
        [(h, w) for (w, h) in img_sizes_wh], dtype=np.int64,
    )

    if threshold_by_class:
        pre_threshold = min(threshold, min(threshold_by_class.values()))
    else:
        pre_threshold = threshold

    (scores_topk, labels_topk, boxes_topk, order_seq_topk,
     masks_topk_logits, _last_hs) = ort_run_fused(pixel_values, target_sizes)

    raw_results = _post_process_from_fused(
        scores_topk, labels_topk, boxes_topk, order_seq_topk,
        masks_topk_logits,
        target_sizes=target_sizes,
        threshold=pre_threshold,
        processor_size=processor_size,
    )

    if threshold_by_class:
        raw_results = np_apply_per_class_threshold(
            raw_results, threshold, threshold_by_class, id2label,
        )

    return np_apply_layout_postprocess(
        raw_results=raw_results,
        id2label=id2label,
        img_sizes=img_sizes_wh,
        layout_nms=layout_nms,
        layout_unclip_ratio=layout_unclip_ratio,
        layout_merge_bboxes_mode=layout_merge_bboxes_mode,
    )
