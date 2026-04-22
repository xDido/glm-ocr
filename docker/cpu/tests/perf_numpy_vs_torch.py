"""Per-page latency and resource comparison: numpy vs torch postproc.

Runs each path through the actual runtime_app.instrument_pipeline() setup
(just like gunicorn does) and measures:
  - wall time per ld.process([img]) call (mean / p95)
  - RSS delta after startup
  - thread count delta after startup

This is an in-container micro-benchmark — NOT a replacement for the full
asyncio-matrix, which also exercises gunicorn concurrency and the batcher.
But it catches obvious regressions in the hot path.

Usage inside container:
    LAYOUT_BACKEND=onnx LAYOUT_POSTPROC=torch python /app/perf_numpy_vs_torch.py
    LAYOUT_BACKEND=onnx LAYOUT_POSTPROC=numpy python /app/perf_numpy_vs_torch.py

Invoke twice (once per path) and compare the two stdout blocks.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/app")


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return float("nan")


def _thread_count() -> int:
    return threading.active_count()


def main() -> int:
    backend = os.environ.get("LAYOUT_BACKEND", "<unset>")
    postproc = os.environ.get("LAYOUT_POSTPROC", "<unset>")
    print(f"[perf] LAYOUT_BACKEND={backend} LAYOUT_POSTPROC={postproc}")
    print(f"[perf] startup  RSS={_rss_mb():.1f} MB  threads={_thread_count()}")

    from glmocr.server import create_app  # type: ignore
    from glmocr.config import load_config  # type: ignore

    cfg = load_config("/app/config.yaml")
    app = create_app(cfg)
    pipeline = app.config.get("pipeline")
    pipeline.start()

    import runtime_app
    runtime_app.instrument_pipeline(pipeline)
    print(f"[perf] post-start RSS={_rss_mb():.1f} MB  threads={_thread_count()}")

    ld = pipeline.layout_detector
    img = Image.open("/app/smoke_test.png").convert("RGB")

    # Warmup — first call may pay lazy-init cost.
    ld.process([img], save_visualization=False, global_start_idx=0, use_polygon=False)
    print(f"[perf] post-warmup RSS={_rss_mb():.1f} MB  threads={_thread_count()}")

    iters = int(os.environ.get("PERF_ITERS", "10"))
    times_ms = []
    for i in range(iters):
        t0 = time.perf_counter()
        ld.process([img], save_visualization=False, global_start_idx=0, use_polygon=False)
        times_ms.append((time.perf_counter() - t0) * 1000)

    times_ms = np.array(times_ms)
    print(f"[perf] {iters} iters:")
    print(f"  mean     = {times_ms.mean():.1f} ms")
    print(f"  median   = {float(np.median(times_ms)):.1f} ms")
    print(f"  p95      = {float(np.percentile(times_ms, 95)):.1f} ms")
    print(f"  min      = {times_ms.min():.1f} ms")
    print(f"  max      = {times_ms.max():.1f} ms")
    print(f"[perf] post-bench RSS={_rss_mb():.1f} MB  threads={_thread_count()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
