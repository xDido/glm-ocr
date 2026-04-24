"""Parity probe: current torch-exported layout vs alex-dinh Paddle2ONNX layout.

For N deterministic OmniDocBench pages, compute:
 - current pipeline boxes+labels  (via POST /glmocr/parse → json_result[0])
 - paddle model boxes+labels      (run alex-dinh ONNX directly + decode config)
 - match by IoU >= 0.5, report per-image precision/recall/F1 and class agreement

Runs inside the cpu container (has onnxruntime + PIL + numpy + urllib + HTTP
loopback to localhost:5002). Invoke via stdin:
    docker exec -i glmocr-cpu python - < scripts/parity_probe_paddle_onnx.py
"""
from __future__ import annotations

import json
import random
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

PADDLE_MODEL = "/tmp/alex-dinh/PP-DocLayoutV3.onnx"
IMAGES_DIR = Path("/app/datasets/OmniDocBench/images")
ENDPOINT = "http://localhost:5002/glmocr/parse"
CONTAINER_URL_PREFIX = "file:///app/datasets/OmniDocBench/images"
SEED = 42
N = 10
SCORE_TH = 0.5
IOU_TH = 0.5

PADDLE_LABELS = [
    "abstract", "algorithm", "aside_text", "chart", "content",
    "display_formula", "doc_title", "figure_title", "footer",
    "footer_image", "footnote", "formula_number", "header",
    "header_image", "image", "inline_formula", "number",
    "paragraph_title", "reference", "reference_content", "seal",
    "table", "text", "vertical_text", "vision_footnote",
]


def pick_images() -> list[Path]:
    all_ = sorted(IMAGES_DIR.iterdir())
    rng = random.Random(SEED)
    return rng.sample(all_, N)


def torch_boxes_for(image_rel: str) -> tuple[list[tuple[float, float, float, float]], list[str]]:
    body = json.dumps({"images": [f"{CONTAINER_URL_PREFIX}/{image_rel}"]}).encode()
    req = urllib.request.Request(ENDPOINT, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        payload = json.loads(r.read())
    jr = payload.get("json_result") or []
    if not jr or not isinstance(jr[0], list):
        return [], []
    boxes, labels = [], []
    for block in jr[0]:
        b = block.get("bbox_2d")
        lab = block.get("label") or ""
        if not b or len(b) != 4:
            continue
        x1, y1, x2, y2 = b
        boxes.append((float(x1), float(y1), float(x2), float(y2)))
        labels.append(lab)
    return boxes, labels


def preprocess_for_paddle(img_path: Path) -> tuple[np.ndarray, tuple[int, int], tuple[float, float]]:
    """Per config.json: Resize(800x800, keep_ratio=False) → NormalizeImage(mean=0,std=1,norm=none)
       → Permute HWC->CHW. No /255."""
    im = Image.open(img_path).convert("RGB")
    orig_w, orig_h = im.size
    resized = im.resize((800, 800), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0      # HWC, 0-1 (Paddle's implicit /255)
    arr = arr.transpose(2, 0, 1)[None]                       # 1,3,800,800
    # PP-DocLayoutV3 uses scale_factor = (h_orig/h_in, w_orig/w_in) to recover coords.
    # The model outputs boxes in ORIGINAL image space when scale_factor is passed correctly.
    scale = (orig_h / 800.0, orig_w / 800.0)
    return arr, (orig_h, orig_w), scale


def paddle_boxes_for(sess: ort.InferenceSession, img_path: Path) -> tuple[list[tuple[float, float, float, float]], list[str]]:
    img, (h, w), (sh, sw) = preprocess_for_paddle(img_path)
    # Paddle convention: im_shape = original HxW; scale_factor = (h_orig/h_in, w_orig/w_in)
    feed = {
        "image": img.astype(np.float32),
        "im_shape": np.array([[h, w]], dtype=np.float32),
        "scale_factor": np.array([[sh, sw]], dtype=np.float32),
    }
    out = sess.run(None, feed)
    # out[0]: (N_det, 7) rows = [class, score, x1, y1, x2, y2, ?]
    dets = out[0]
    boxes, labels = [], []
    for row in dets:
        cls = int(row[0])
        score = float(row[1])
        if score < SCORE_TH or cls < 0 or cls >= len(PADDLE_LABELS):
            continue
        x1, y1, x2, y2 = float(row[2]), float(row[3]), float(row[4]), float(row[5])
        boxes.append((x1, y1, x2, y2))
        labels.append(PADDLE_LABELS[cls])
    return boxes, labels


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = ua + ub - inter
    return inter / den if den > 0 else 0.0


def normalize_boxes(boxes: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    """Normalize to [0,1] using the page's max observed coord per axis.
    Handles the torch-vs-paddle coord-space mismatch without inspecting glmocr
    internals — both models see the same page, so max coords = implicit scale."""
    if not boxes:
        return []
    max_x = max(max(b[0], b[2]) for b in boxes) or 1.0
    max_y = max(max(b[1], b[3]) for b in boxes) or 1.0
    return [(b[0] / max_x, b[1] / max_y, b[2] / max_x, b[3] / max_y) for b in boxes]


def greedy_match(torch_boxes, paddle_boxes):
    used_paddle = set()
    matches = []  # list of (t_idx, p_idx, iou)
    for ti, tb in enumerate(torch_boxes):
        best_iou = 0.0; best_p = -1
        for pi, pb in enumerate(paddle_boxes):
            if pi in used_paddle:
                continue
            v = iou(tb, pb)
            if v > best_iou:
                best_iou = v; best_p = pi
        if best_p >= 0 and best_iou >= IOU_TH:
            matches.append((ti, best_p, best_iou))
            used_paddle.add(best_p)
    return matches


def main() -> None:
    so = ort.SessionOptions(); so.intra_op_num_threads = 2; so.log_severity_level = 3
    sess = ort.InferenceSession(PADDLE_MODEL, sess_options=so, providers=["CPUExecutionProvider"])
    print(f"[probe] seed={SEED} n={N} iou_th={IOU_TH} score_th={SCORE_TH}", flush=True)

    imgs = pick_images()
    per_image = []
    for i, img in enumerate(imgs, 1):
        rel = img.name
        t0 = time.perf_counter()
        t_boxes, t_labels = torch_boxes_for(rel)
        t1 = time.perf_counter()
        p_boxes, p_labels = paddle_boxes_for(sess, img)
        t2 = time.perf_counter()
        t_norm = normalize_boxes(t_boxes)
        p_norm = normalize_boxes(p_boxes)
        matches = greedy_match(t_norm, p_norm)
        class_agree = sum(1 for (ti, pi, _) in matches if t_labels[ti] == p_labels[pi])
        # Precision/recall treating torch as "reference"
        tp = len(matches)
        fn = len(t_boxes) - tp
        fp = len(p_boxes) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_image.append({
            "name": rel, "t_n": len(t_boxes), "p_n": len(p_boxes),
            "matched": tp, "class_agree": class_agree,
            "P": prec, "R": rec, "F1": f1,
            "t_ms": (t1 - t0) * 1000, "p_ms": (t2 - t1) * 1000,
        })
        print(f"[{i}/{N}] {rel[:60]}  torch={len(t_boxes)} paddle={len(p_boxes)} "
              f"matched={tp} cls_agree={class_agree}  P={prec:.2f} R={rec:.2f} F1={f1:.2f}  "
              f"t_torch={(t1-t0)*1000:.0f}ms t_paddle={(t2-t1)*1000:.0f}ms", flush=True)

    print("\n[aggregate]")
    m = lambda k: sum(x[k] for x in per_image) / len(per_image)
    s = lambda k: sum(x[k] for x in per_image)
    print(f"  mean F1:        {m('F1'):.3f}")
    print(f"  mean precision: {m('P'):.3f}")
    print(f"  mean recall:    {m('R'):.3f}")
    print(f"  totals: torch_boxes={s('t_n')}  paddle_boxes={s('p_n')}  matched={s('matched')}  class_agree={s('class_agree')}")
    print(f"  mean latency: torch={m('t_ms'):.0f}ms  paddle_onnx_only={m('p_ms'):.0f}ms")


if __name__ == "__main__":
    main()
