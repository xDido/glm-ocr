"""Three-way layout forward benchmark: which backend + runtime is fastest on CPU?

  A. paddle2onnx + ORT CPUExecutionProvider        (shipped in main)
  B. paddle2onnx + ORT OpenVINOExecutionProvider   (revalidated 2026-04-24)
  C. native Paddle → OpenVINO IR (no ONNX step)    (this branch's new path)

Requires both `onnxruntime-openvino` and `openvino` pip packages in the
container. Each configuration runs in a subprocess for clean isolation,
writes per-configuration JSON to /tmp, then the orchestrator prints a
side-by-side table and a parity checksum diff.

Run from the host:
    MSYS_NO_PATHCONV=1 docker run --rm \\
        -v C:/Users/Dido/Desktop/GLM-OCR:/work \\
        -v C:/Users/Dido/Desktop/GLM-OCR/hf-cache:/root/.cache/huggingface \\
        -v C:/Users/Dido/Desktop/GLM-OCR/datasets:/app/datasets \\
        --entrypoint bash glmocr-cpu:local \\
        -c 'pip install -q onnxruntime-openvino openvino >/dev/null 2>&1 && \\
            python /work/scripts/bench_three_way.py'
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

ONNX_PATH = "/root/.cache/huggingface/glmocr-layout-onnx/pp_doclayout_v3_paddle2onnx.onnx"
PADDLE_DIR = "/root/.cache/huggingface/glmocr-layout-paddle"
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


def _make_batch(batch_size: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from PIL import Image
    rng = np.random.default_rng(seed)
    pick_idx = rng.choice(len(SMOKE_TEST_NAMES), size=batch_size, replace=True)
    tensors, shapes = [], []
    for i in pick_idx:
        im = Image.open(IMAGES_DIR / SMOKE_TEST_NAMES[int(i)]).convert("RGB")
        w, h = im.size
        arr = np.asarray(im.resize((800, 800), Image.BILINEAR), dtype=np.float32) / 255.0
        tensors.append(arr.transpose(2, 0, 1))
        shapes.append((h, w))
    image = np.stack(tensors).astype(np.float32)
    im_shape = np.array(shapes, dtype=np.float32)
    scale_factor = np.array([(h / 800.0, w / 800.0) for (h, w) in shapes], dtype=np.float32)
    return image, im_shape, scale_factor


def _checksum(dets_2d: np.ndarray) -> list[list[float]]:
    """Sort by score desc, take top 5, round for stable cross-runtime compare."""
    if dets_2d is None or dets_2d.size == 0:
        return []
    top5 = dets_2d[np.argsort(-dets_2d[:, 1])[:5]]
    return [[int(r[0]), float(f"{r[1]:.3f}"),
             float(f"{r[2]:.1f}"), float(f"{r[3]:.1f}"),
             float(f"{r[4]:.1f}"), float(f"{r[5]:.1f}")]
            for r in top5]


def _stats(times: list[float], batch_size: int) -> dict:
    return {
        "mean_ms": mean(times) * 1000,
        "p50_ms": median(times) * 1000,
        "p95_ms": sorted(times)[int(len(times) * 0.95)] * 1000 if len(times) >= 20 else max(times) * 1000,
        "per_image_ms": mean(times) * 1000 / batch_size,
    }


def run_ort(provider: str, batch_sizes: list[int], warmup: int, repeats: int) -> dict:
    import onnxruntime as ort
    avail = ort.get_available_providers()
    print(f"[ort/{provider}] onnxruntime={ort.__version__}  avail={avail}", flush=True)
    if provider not in avail:
        return {"error": f"provider {provider} not available"}

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 3
    provider_opts = [{"device_type": "CPU"}] if provider == "OpenVINOExecutionProvider" else None
    sess = ort.InferenceSession(ONNX_PATH, sess_options=so,
                                providers=[provider], provider_options=provider_opts)

    out: dict = {}
    for bs in batch_sizes:
        image, im_shape, scale_factor = _make_batch(bs, seed=bs)
        feed = {"image": image, "im_shape": im_shape, "scale_factor": scale_factor}
        for _ in range(warmup):
            _ = sess.run(None, feed)
        times = []
        results = None
        for _ in range(repeats):
            t = time.perf_counter()
            results = sess.run(None, feed)
            times.append(time.perf_counter() - t)
        out[bs] = {**_stats(times, bs), "checksum": _checksum(results[0])}
        print(f"[ort/{provider}] batch={bs}: mean={out[bs]['mean_ms']:.0f}ms per_image={out[bs]['per_image_ms']:.0f}ms", flush=True)
    return out


def run_ov_native(batch_sizes: list[int], warmup: int, repeats: int) -> dict:
    import openvino as ov
    print(f"[ov-native] openvino={ov.__version__}", flush=True)
    core = ov.Core()
    model_json = f"{PADDLE_DIR}/inference.json"
    weights = f"{PADDLE_DIR}/inference.pdiparams"
    # OpenVINO's Paddle frontend: pass both files positionally.
    model = core.read_model(model_json, weights)
    print(f"[ov-native] inputs: {[(i.any_name, i.partial_shape.to_string()) for i in model.inputs]}", flush=True)
    print(f"[ov-native] outputs: {[(o.any_name, o.partial_shape.to_string()) for o in model.outputs]}", flush=True)
    compiled = core.compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": 3})

    # Build a name→port lookup since Paddle models use 'image', 'im_shape', 'scale_factor'
    inputs = {i.any_name: i for i in compiled.inputs}

    out: dict = {}
    for bs in batch_sizes:
        image, im_shape, scale_factor = _make_batch(bs, seed=bs)
        feed = {inputs["image"]: image,
                inputs["im_shape"]: im_shape,
                inputs["scale_factor"]: scale_factor}
        for _ in range(warmup):
            _ = compiled(feed)
        times = []
        results = None
        for _ in range(repeats):
            t = time.perf_counter()
            results = compiled(feed)
            times.append(time.perf_counter() - t)
        # results is a dict keyed by Output objects; unpack positionally
        dets = next(iter(results.values()))
        # For native Paddle, outputs might be in different order — try to find (N, 7) array
        np_results = [v for v in results.values() if isinstance(v, np.ndarray)]
        dets_7col = next((a for a in np_results if a.ndim == 2 and a.shape[1] == 7), None)
        out[bs] = {**_stats(times, bs), "checksum": _checksum(dets_7col)}
        print(f"[ov-native] batch={bs}: mean={out[bs]['mean_ms']:.0f}ms per_image={out[bs]['per_image_ms']:.0f}ms", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ort-cpu", "ort-ov", "ov-native"])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.mode is None:
        # Orchestrator: run all three in separate subprocesses.
        results: dict = {}
        for mode in ("ort-cpu", "ort-ov", "ov-native"):
            print(f"\n=== {mode} ===", flush=True)
            json_path = Path(f"/tmp/bench_{mode}.json")
            subprocess.run(
                [sys.executable, __file__,
                 "--mode", mode,
                 "--batch-sizes", *[str(b) for b in args.batch_sizes],
                 "--warmup", str(args.warmup),
                 "--repeats", str(args.repeats),
                 "--out", str(json_path)],
                check=False,
            )
            results[mode] = json.loads(json_path.read_text()) if json_path.exists() else {"error": "no output"}

        print("\n=== SIDE BY SIDE ===")
        bss = sorted({int(bs) for r in results.values() if isinstance(r, dict) and "error" not in r
                      for bs in r.keys()})
        print(f"{'batch':>5}  {'ort-cpu':>12}  {'ort-ov':>12}  {'ov-native':>12}  {'ov-native/cpu':>14}  parity")
        for bs in bss:
            row = []
            for mode in ("ort-cpu", "ort-ov", "ov-native"):
                r = results[mode]
                if "error" in r:
                    row.append(None)
                elif str(bs) in r:
                    row.append(r[str(bs)])
                else:
                    row.append(None)
            if all(x is not None for x in row):
                cpu, ovep, ovn = row
                speedup_cpu_vs_ovn = cpu["mean_ms"] / ovn["mean_ms"] if ovn["mean_ms"] else 0
                # parity: ov-native vs ort-cpu (if match, both paths produce same output)
                parity_ovn = "match" if ovn["checksum"] == cpu["checksum"] else "DIVERGE"
                parity_ov_ep = "match" if row[1]["checksum"] == cpu["checksum"] else "DIVERGE"
                print(f"  {bs:>3}  {cpu['mean_ms']:>9.0f} ms  {ovep['mean_ms']:>9.0f} ms  {ovn['mean_ms']:>9.0f} ms  {speedup_cpu_vs_ovn:>10.2f}×   ov-ep:{parity_ov_ep} ov-native:{parity_ovn}")

        # Dump any parity divergence
        for bs in bss:
            for mode in ("ort-ov", "ov-native"):
                r = results[mode]
                ref = results["ort-cpu"]
                if "error" in r or "error" in ref:
                    continue
                if r[str(bs)]["checksum"] != ref[str(bs)]["checksum"]:
                    print(f"\nparity diff @ batch={bs}, {mode} vs ort-cpu:")
                    for a, b in zip(r[str(bs)]["checksum"], ref[str(bs)]["checksum"]):
                        print(f"  {mode:9s}: {a}")
                        print(f"  ort-cpu  : {b}")
        return

    # Worker mode
    if args.mode == "ort-cpu":
        r = run_ort("CPUExecutionProvider", args.batch_sizes, args.warmup, args.repeats)
    elif args.mode == "ort-ov":
        r = run_ort("OpenVINOExecutionProvider", args.batch_sizes, args.warmup, args.repeats)
    elif args.mode == "ov-native":
        r = run_ov_native(args.batch_sizes, args.warmup, args.repeats)
    else:
        raise SystemExit(f"unknown mode {args.mode}")
    if args.out:
        args.out.write_text(json.dumps(r))


if __name__ == "__main__":
    main()
