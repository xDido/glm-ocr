"""Layout forward benchmark: PyTorch eager on CPU vs the shipped Paddle2ONNX CPU EP.

Runs PP-DocLayoutV3 via HF transformers on the raw SafeTensors weights
(no ORT export, no graph tracing). PyTorch eager handles dynamic batch
natively — no baked `1` anywhere — so the only open question is how its
kernel speed compares to the Paddle2ONNX CPU EP path we ship today.

Context: https://huggingface.co/PaddlePaddle/PP-DocLayoutV3/discussions/8
claims PyTorch is 1.3-1.5× faster than ONNX Runtime end-to-end, but on
GPU (RTX 5060 Ti 16 GB). We don't have that headroom on the 8 GB dev
card, so this bench answers the CPU-side question.

Invoked from the host via a throwaway container:
    docker run --rm -v ...:/work --entrypoint bash glmocr-cpu:local \\
        -c "python /work/scripts/bench_torch_eager.py"
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

MODEL_ID = "PaddlePaddle/PP-DocLayoutV3_safetensors"
IMAGES_DIR = Path("/app/datasets/OmniDocBench/images")
SMOKE_TEST_NAMES = [
    "PPT_lay_linalg5_01_05_page_003.png",
    "docstructbench_llm-raw-scihub-o.O-j.chroma.2005.11.024.pdf_7.jpg",
    "yanbaopptmerge_yanbaoPPT_5675.jpg",
    "jiaocaineedrop_jiaocai_needrop_en_3114.jpg",
    "jiaocaineedrop_jiaocai_needrop_en_1826.jpg",
    "jiaocaineedrop_jiaocai_needrop_en_101.jpg",
    "eastmoney_03e39833d520af7d1abb96dd624dcd15f2b57d061969d196e80922e2f590503f.pdf_22.jpg",
    "yanbaopptmerge_yanbaoPPT_5245.jpg",
    "docstructbench_llm-raw-scihub-o.O-ijc.22994.pdf_3.jpg",
]


def _load(device: str) -> tuple[AutoModelForObjectDetection, AutoImageProcessor]:
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_ID)
    model.eval()
    model.to(device)
    return model, proc


def _make_batch(proc, batch_size: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    pick_idx = rng.choice(len(SMOKE_TEST_NAMES), size=batch_size, replace=True)
    images = [Image.open(IMAGES_DIR / SMOKE_TEST_NAMES[int(i)]).convert("RGB") for i in pick_idx]
    return proc(images=images, return_tensors="pt")


def bench(model, proc, device: str, batch_sizes: list[int], warmup: int, repeats: int) -> dict:
    out: dict = {}
    torch.set_num_threads(3)  # match LAYOUT_ONNX_THREADS=3 in .env for fair compare

    for bs in batch_sizes:
        batch = _make_batch(proc, bs, seed=bs)
        pixel_values = batch["pixel_values"].to(device)

        with torch.no_grad():
            for _ in range(warmup):
                _ = model(pixel_values=pixel_values)

        times = []
        with torch.no_grad():
            for _ in range(repeats):
                if device == "cuda":
                    torch.cuda.synchronize()
                t = time.perf_counter()
                _ = model(pixel_values=pixel_values)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t)

        out[bs] = {
            "mean_ms": mean(times) * 1000,
            "p50_ms": median(times) * 1000,
            "p95_ms": sorted(times)[int(len(times) * 0.95)] * 1000 if len(times) >= 20 else max(times) * 1000,
            "per_image_ms": mean(times) * 1000 / bs,
        }
        print(f"[torch-{device}] batch={bs}: mean={out[bs]['mean_ms']:.0f}ms  per_image={out[bs]['per_image_ms']:.0f}ms  p50={out[bs]['p50_ms']:.0f}ms", flush=True)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=8)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available in this container")
    print(f"torch={torch.__version__} device={args.device}")
    if args.device == "cuda":
        print(f"cuda device: {torch.cuda.get_device_name(0)}")

    model, proc = _load(args.device)
    print(f"model loaded, n_params = {sum(p.numel() for p in model.parameters()):,}")
    bench(model, proc, args.device, args.batch_sizes, args.warmup, args.repeats)


if __name__ == "__main__":
    main()
