"""
Gunicorn entry point for glmocr.server.

Upstream defines `app` inside `create_app(config)` with no module-level
export, so gunicorn can't target `glmocr.server:app` directly. We load the
config once per worker fork and expose `app` at import time.

The import paths are best-effort across glmocr versions; we try the common
locations before falling back to a manual config load.
"""
from __future__ import annotations

import os

CONFIG_PATH = os.environ.get("GLMOCR_CONFIG", "/app/config.yaml")


def _load_config(path: str):
    try:
        from glmocr.config import load_config  # type: ignore
        return load_config(path)
    except Exception:
        pass
    try:
        from glmocr.config import Config  # type: ignore
        return Config.from_yaml(path)
    except Exception:
        pass
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build_app():
    from glmocr.server import create_app  # type: ignore
    cfg = _load_config(CONFIG_PATH)
    return create_app(cfg)


app = _build_app()

# Runtime-integrity endpoint. Non-fatal on failure — worker still serves OCR.
try:
    from runtime_app import install as _install_runtime
    app = _install_runtime(app)
except Exception as _exc:  # pragma: no cover
    import sys
    print(f"[wsgi] runtime blueprint not installed: {_exc!r}", file=sys.stderr)
