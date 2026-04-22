"""Smoke test the LAYOUT_POSTPROC=numpy wiring inside the container.

Doesn't go through gunicorn — constructs a minimal Pipeline-like object,
passes it through runtime_app.instrument_pipeline(), then calls
ld.process([image]) and prints the result. Validates:

  * layout_postprocess.py can be imported from /app
  * the numpy path installs without errors
  * ORT session loads
  * _numpy_process closure runs end-to-end against a real page
  * output shape matches what glmocr.layout.PPDocLayoutDetector.process returns

Usage inside container:
    LAYOUT_BACKEND=onnx LAYOUT_POSTPROC=numpy \
        python /app/smoke_runtime_numpy.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, "/app")

# Caller sets LAYOUT_BACKEND and LAYOUT_POSTPROC in the shell env; we leave
# them alone so this smoke test can exercise either configuration.
print(f"[smoke] LAYOUT_BACKEND={os.environ.get('LAYOUT_BACKEND', '<unset>')} "
      f"LAYOUT_POSTPROC={os.environ.get('LAYOUT_POSTPROC', '<unset>')}")


def main() -> int:
    # Live pipeline on the filesystem: build a real glmocr pipeline so the
    # layout detector is initialised exactly as gunicorn does it.
    # Mirror wsgi.py's load path.
    from glmocr.server import create_app  # type: ignore
    from glmocr.config import load_config  # type: ignore

    cfg = load_config("/app/config.yaml")
    app = create_app(cfg)
    pipeline = app.config.get("pipeline")
    assert pipeline is not None, "create_app did not attach pipeline"
    pipeline.start()
    print("[smoke] pipeline started", flush=True)

    # Invoke the instrumentation code path — this is what wsgi.py does.
    import runtime_app
    runtime_app.instrument_pipeline(pipeline)
    print("[smoke] instrument_pipeline() returned", flush=True)

    img = Image.open("/app/smoke_test.png").convert("RGB")
    print(f"[smoke] calling ld.process on {img.size}", flush=True)
    ld = pipeline.layout_detector
    results, vis = ld.process(
        [img], save_visualization=False, global_start_idx=0, use_polygon=False,
    )
    print(f"[smoke] got {len(results)} result list(s)")
    if results:
        print(f"[smoke] first page has {len(results[0])} detections")
        for i, r in enumerate(results[0][:3]):
            print(f"  det {i}: label={r['label']} score={r['score']:.3f} "
                  f"bbox={r['bbox_2d']} task={r['task_type']}")
    assert isinstance(results, list)
    assert isinstance(vis, dict)
    print("[smoke] PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
