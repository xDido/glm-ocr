"""Profile the current layout ONNX graph with a real-size input.

Writes a chrome-trace JSON under /tmp inside the container, then
summarizes the top ops by cumulative wall time. Run inside the cpu
container via `docker exec glmocr-cpu python /app/scripts/profile_layout.py`.
"""
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import PPDocLayoutV3ImageProcessor

MODEL_DIR = os.environ.get("LAYOUT_MODEL_DIR", "PaddlePaddle/PP-DocLayoutV3_safetensors")
HF_HOME = os.environ.get("HF_HOME", "/root/.cache/huggingface")
ONNX_PATH = Path(HF_HOME) / "glmocr-layout-onnx" / "pp_doclayout_v3.onnx"
IMG_PATH = "/app/datasets/OmniDocBench/images/PPT_1001115_eng_page_003.png"
INTRA_OP = int(os.environ.get("LAYOUT_ONNX_THREADS", "3"))
N_CALLS = int(os.environ.get("PROFILE_N_CALLS", "20"))

print(f"[profile] graph={ONNX_PATH}  intra_op={INTRA_OP}  N={N_CALLS}")

opts = ort.SessionOptions()
opts.intra_op_num_threads = INTRA_OP
opts.enable_profiling = True
opts.profile_file_prefix = "/tmp/layout_profile"
sess = ort.InferenceSession(str(ONNX_PATH), opts, providers=["CPUExecutionProvider"])

processor = PPDocLayoutV3ImageProcessor.from_pretrained(MODEL_DIR)
img = Image.open(IMG_PATH).convert("RGB")
inputs = processor(images=[img], return_tensors="np")
pv = inputs["pixel_values"]
print(f"[profile] pixel_values shape={pv.shape}  dtype={pv.dtype}")

for _ in range(3):
    sess.run(None, {"pixel_values": pv})

wall = []
t0_all = time.perf_counter()
for i in range(N_CALLS):
    t0 = time.perf_counter()
    sess.run(None, {"pixel_values": pv})
    wall.append((time.perf_counter() - t0) * 1000)
total_ms = (time.perf_counter() - t0_all) * 1000
profile_path = sess.end_profiling()
print(f"[profile] profile written to {profile_path}")
print(f"[profile] wall per call: mean={np.mean(wall):.1f} ms  "
      f"p50={np.percentile(wall, 50):.1f}  p95={np.percentile(wall, 95):.1f}  "
      f"total={total_ms:.0f} ms over {N_CALLS} calls")

with open(profile_path) as f:
    events = json.load(f)

op_totals = defaultdict(lambda: {"count": 0, "total_us": 0, "max_us": 0})
op_name_totals = defaultdict(lambda: {"count": 0, "total_us": 0})

for ev in events:
    if ev.get("cat") != "Node":
        continue
    args = ev.get("args") or {}
    op_name = args.get("op_name")
    node_name = ev.get("name")
    dur = ev.get("dur", 0)
    if not op_name:
        continue
    key = (op_name, node_name)
    op_totals[key]["count"] += 1
    op_totals[key]["total_us"] += dur
    op_totals[key]["max_us"] = max(op_totals[key]["max_us"], dur)
    op_name_totals[op_name]["count"] += 1
    op_name_totals[op_name]["total_us"] += dur

print()
print("[profile] top op_names by cumulative us (across all nodes, all calls):")
print(f"{'op_name':<30} {'count':>8} {'total_ms':>12} {'per_call_ms':>14}")
for op_name, stats in sorted(op_name_totals.items(), key=lambda kv: -kv[1]["total_us"])[:12]:
    tot_ms = stats["total_us"] / 1000
    per_call = tot_ms / N_CALLS
    print(f"{op_name:<30} {stats['count']:>8} {tot_ms:>12.1f} {per_call:>14.2f}")

print()
print("[profile] top individual nodes by cumulative us:")
print(f"{'op_name':<24} {'node':<48} {'count':>5} {'total_ms':>10}")
for (op_name, node_name), stats in sorted(op_totals.items(), key=lambda kv: -kv[1]["total_us"])[:15]:
    tot_ms = stats["total_us"] / 1000
    print(f"{op_name:<24} {node_name[:48]:<48} {stats['count']:>5} {tot_ms:>10.1f}")
