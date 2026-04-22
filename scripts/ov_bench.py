"""Micro-bench: layout ONNX forward on CPU EP vs OpenVINO EP.

Runs the same pre-processed image through both providers 20× each after
3 warmup passes, prints side-by-side wall time. Use inside a throwaway
container that has onnxruntime-openvino installed; preprocessor is
imported from transformers which the glmocr-cpu:local image already has.
"""
import os
import time
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
N_CALLS = int(os.environ.get("BENCH_N_CALLS", "20"))

print(f"ort={ort.__version__}  providers_available={ort.get_available_providers()}")
print(f"graph={ONNX_PATH}  intra_op={INTRA_OP}  N={N_CALLS}")

processor = PPDocLayoutV3ImageProcessor.from_pretrained(MODEL_DIR)
img = Image.open(IMG_PATH).convert("RGB")
pv = processor(images=[img], return_tensors="np")["pixel_values"]
print(f"pixel_values shape={pv.shape} dtype={pv.dtype}")


def bench(provider_list, label):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = INTRA_OP
    sess = ort.InferenceSession(str(ONNX_PATH), opts, providers=provider_list)
    print(f"\n[{label}] session providers={sess.get_providers()}")
    # Warmup
    for _ in range(3):
        sess.run(None, {"pixel_values": pv})
    wall = []
    t0_all = time.perf_counter()
    for _ in range(N_CALLS):
        t0 = time.perf_counter()
        sess.run(None, {"pixel_values": pv})
        wall.append((time.perf_counter() - t0) * 1000)
    total_ms = (time.perf_counter() - t0_all) * 1000
    print(f"[{label}] wall per call: mean={np.mean(wall):.1f} ms  "
          f"p50={np.percentile(wall, 50):.1f}  p95={np.percentile(wall, 95):.1f}  "
          f"min={np.min(wall):.1f}  total={total_ms:.0f} ms")
    return np.mean(wall)


cpu_mean = bench(["CPUExecutionProvider"], "CPU EP")
if "OpenVINOExecutionProvider" in ort.get_available_providers():
    ov_mean = bench(
        [("OpenVINOExecutionProvider", {"device_type": "CPU"}), "CPUExecutionProvider"],
        "OpenVINO EP (device=CPU)",
    )
    delta = (ov_mean - cpu_mean) / cpu_mean * 100
    direction = "faster" if ov_mean < cpu_mean else "slower"
    print(f"\nOpenVINO vs CPU EP: {ov_mean:.1f} ms vs {cpu_mean:.1f} ms  "
          f"({delta:+.1f}% -> {direction})")
else:
    print("\nOpenVINOExecutionProvider not available")
