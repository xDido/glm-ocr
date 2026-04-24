"""Layout-forward benchmark: CPU EP vs OpenVINO EP on the Paddle2ONNX graph.

Loads pp_doclayout_v3_paddle2onnx.onnx twice (once per EP), runs 20 real
OmniDocBench pages at batch=1, 2, 4, 8, and measures per-call latency +
output parity (fetch_name_0 top-k must agree).

Assumes it runs in a throwaway container where BOTH onnxruntime AND
onnxruntime-openvino are installed side by side — except the two wheels
conflict, so only one can be present at a time. We handle this by
spawning a subprocess for each EP so they don't share the runtime.

Invoked from the host as:
    docker run --rm -v .../:/work ... this-container python scripts/bench_paddle_ep.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median

import numpy as np

MODEL_PATH = "/root/.cache/huggingface/glmocr-layout-onnx/pp_doclayout_v3_paddle2onnx.onnx"
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


def _preprocess(name: str) -> tuple[np.ndarray, tuple[int, int]]:
    from PIL import Image  # local import so it's available in the subprocess too
    im = Image.open(IMAGES_DIR / name).convert("RGB")
    w, h = im.size
    arr = np.asarray(im.resize((800, 800), Image.BILINEAR), dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1), (w, h)


def _make_batch(batch_size: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pick_idx = rng.choice(len(SMOKE_TEST_NAMES), size=batch_size, replace=True)
    tensors = []
    shapes = []
    for i in pick_idx:
        t, (w, h) = _preprocess(SMOKE_TEST_NAMES[int(i)])
        tensors.append(t)
        shapes.append((h, w))
    image = np.stack(tensors).astype(np.float32)
    im_shape = np.array(shapes, dtype=np.float32)
    scale_factor = np.array([(h / 800.0, w / 800.0) for (h, w) in shapes], dtype=np.float32)
    return image, im_shape, scale_factor


def _run_provider(provider: str, batch_sizes: list[int], warmup: int, repeats: int) -> dict:
    """Returns dict keyed by batch_size → stats + checksum."""
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"[{provider}] onnxruntime={ort.__version__} providers_available={providers}", flush=True)
    if provider not in providers:
        return {"error": f"provider {provider} not available in {providers}"}

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 3

    provider_opts = {}
    if provider == "OpenVINOExecutionProvider":
        provider_opts = {"device_type": "CPU"}
    sess = ort.InferenceSession(
        MODEL_PATH,
        sess_options=so,
        providers=[provider],
        provider_options=[provider_opts] if provider_opts else None,
    )

    out: dict = {}
    for bs in batch_sizes:
        image, im_shape, scale_factor = _make_batch(bs, seed=bs)
        feed = {"image": image, "im_shape": im_shape, "scale_factor": scale_factor}

        for _ in range(warmup):
            _ = sess.run(None, feed)

        times = []
        for _ in range(repeats):
            t = time.perf_counter()
            results = sess.run(None, feed)
            times.append(time.perf_counter() - t)

        # Checksum over the detection scores (top 5 by score) for parity check.
        dets = results[0]
        top5 = dets[np.argsort(-dets[:, 1])[:5]]
        checksum = [[int(r[0]), float(f"{r[1]:.3f}"),
                    float(f"{r[2]:.1f}"), float(f"{r[3]:.1f}"),
                    float(f"{r[4]:.1f}"), float(f"{r[5]:.1f}")]
                    for r in top5]

        out[bs] = {
            "mean_ms": mean(times) * 1000,
            "p50_ms": median(times) * 1000,
            "p95_ms": sorted(times)[int(len(times) * 0.95)] * 1000 if len(times) >= 20 else max(times) * 1000,
            "per_image_ms": mean(times) * 1000 / bs,
            "checksum": checksum,
        }
        print(f"[{provider}] batch={bs}: mean={out[bs]['mean_ms']:.0f}ms  per_image={out[bs]['per_image_ms']:.0f}ms  p50={out[bs]['p50_ms']:.0f}ms  p95={out[bs]['p95_ms']:.0f}ms", flush=True)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["CPUExecutionProvider", "OpenVINOExecutionProvider"])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.provider is None:
        # Orchestrator mode: run two subprocesses, one per provider, in sequence.
        results: dict = {}
        for prov in ["CPUExecutionProvider", "OpenVINOExecutionProvider"]:
            print(f"\n=== {prov} ===", flush=True)
            json_path = Path(f"/tmp/bench_{prov}.json")
            subprocess.run(
                [sys.executable, __file__,
                 "--provider", prov,
                 "--batch-sizes", *[str(b) for b in args.batch_sizes],
                 "--warmup", str(args.warmup),
                 "--repeats", str(args.repeats),
                 "--out", str(json_path)],
                check=False,
            )
            if json_path.exists():
                results[prov] = json.loads(json_path.read_text())
            else:
                results[prov] = {"error": "subprocess produced no output"}

        # Side-by-side table
        print("\n=== SIDE BY SIDE ===")
        cpu = results.get("CPUExecutionProvider", {})
        ov = results.get("OpenVINOExecutionProvider", {})
        if "error" in cpu or "error" in ov:
            print("CPU error:", cpu.get("error"))
            print("OV  error:", ov.get("error"))
            return
        print(f"{'batch':>5}  {'CPU mean':>10}  {'OV mean':>10}  {'speedup':>8}  {'CPU/img':>9}  {'OV/img':>9}  {'parity?':>8}")
        for bs in sorted(set(map(int, cpu.keys())) & set(map(int, ov.keys()))):
            c = cpu[str(bs)]
            o = ov[str(bs)]
            speedup = c["mean_ms"] / o["mean_ms"] if o["mean_ms"] > 0 else 0.0
            parity = "match" if c["checksum"] == o["checksum"] else "DIVERGE"
            print(f"  {bs:>3}  {c['mean_ms']:>9.0f}ms  {o['mean_ms']:>9.0f}ms  {speedup:>7.2f}x  {c['per_image_ms']:>7.0f}ms  {o['per_image_ms']:>7.0f}ms  {parity:>8}")
        # Dump parity diffs if any
        for bs in sorted(set(map(int, cpu.keys())) & set(map(int, ov.keys()))):
            c = cpu[str(bs)]; o = ov[str(bs)]
            if c["checksum"] != o["checksum"]:
                print(f"\nparity diff at batch={bs}:")
                for (cr, or_) in zip(c["checksum"], o["checksum"]):
                    print(f"  CPU={cr}")
                    print(f"   OV={or_}")
        return

    # Worker mode: run one provider, write JSON.
    result = _run_provider(
        provider=args.provider,
        batch_sizes=args.batch_sizes,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    if args.out:
        args.out.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
