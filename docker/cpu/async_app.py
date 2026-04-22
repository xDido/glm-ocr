"""FastAPI async handler for /glmocr/parse (Phase 4 experiment).

Runs as a uvicorn sidecar to the existing gunicorn/Flask stack. Exposes
the same POST /glmocr/parse contract at a different port (5003 by
default) so the load-test harness can target it by flipping CPU_URL.

What we actually get from async here:
  * Request parsing + response serialization moves off a worker thread
    into the asyncio event loop (cheap, minor win).
  * pipeline.process() is still a sync blocking call — we run it under
    asyncio.to_thread so the event loop stays responsive. Inside the
    thread, glmocr's OCR fan-out still uses a ThreadPoolExecutor with
    OCR_MAX_WORKERS workers per request, identical to the sync handler.
  * The theoretical gain: uvicorn can accept N HTTP connections into the
    event loop without dedicating a gthread per request, so under
    bursts the connection admission capacity is much higher than the
    gunicorn (4×16=64) ceiling.

What we do NOT get here:
  * True async OCR fan-out (would require replacing glmocr.ocr_client with
    httpx.AsyncClient — a separate, larger refactor).
  * Faster layout forward — that's CPU-bound and already saturated at 8
    cores after Fix 3.

Exposes:
  GET  /health               — always 200 {"status": "ok"}
  POST /glmocr/parse         — same contract as upstream glmocr.server
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("glmocr-async")
logging.basicConfig(level=logging.INFO)

CONFIG_PATH = os.environ.get("GLMOCR_CONFIG", "/app/config.yaml")

app = FastAPI(title="glmocr-async")


def _build_response(json_result: Any, markdown_result: str) -> dict[str, Any]:
    # Mirror glmocr.server._build_response shape.
    return {"json_result": json_result, "markdown_result": markdown_result}


@app.on_event("startup")
async def _startup() -> None:
    t0 = time.perf_counter()
    # Reuse glmocr.server's loader so pipeline config is identical to the
    # sync handler's. We don't call create_app() — we just need the Pipeline.
    from glmocr.server import create_app  # type: ignore

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

    cfg = _load_config(CONFIG_PATH)
    flask_app = create_app(cfg)
    pipeline = flask_app.config.get("pipeline")
    if pipeline is None:
        raise RuntimeError("create_app did not attach a pipeline")
    pipeline.start()
    app.state.pipeline = pipeline
    logger.info("pipeline.start() done in %.1fs", time.perf_counter() - t0)

    # Install the Fix-1/Phase-1/Fix-3 runtime wrappers (numpy postproc,
    # LAYOUT_POSTPROC=numpy path, batcher, intra_op setting). This is the
    # same instrumentation the sync gunicorn workers apply in wsgi.py.
    try:
        import runtime_app
        runtime_app.instrument_pipeline(pipeline)
        logger.info("runtime_app.instrument_pipeline() applied")
    except Exception:
        logger.warning("runtime_app.instrument_pipeline() skipped:\n%s",
                       traceback.format_exc())


@app.on_event("shutdown")
async def _shutdown() -> None:
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is not None:
        try:
            pipeline.stop()
        except Exception:
            logger.warning("pipeline.stop() raised:\n%s", traceback.format_exc())


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"}, status_code=200)


@app.post("/glmocr/parse")
async def parse(request: Request) -> JSONResponse:
    # Validate Content-Type, matching upstream glmocr.server.create_app.
    ct = request.headers.get("Content-Type", "").split(";", 1)[0].strip()
    if ct != "application/json":
        return JSONResponse(
            {"error": "Invalid Content-Type. Expected 'application/json'."},
            status_code=400,
        )

    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON payload"}, status_code=400)

    images = data.get("images", [])
    if isinstance(images, str):
        images = [images]
    if not images and "file" in data:
        file_val = data["file"]
        if isinstance(file_val, str) and file_val:
            images = [file_val]
    if not images:
        return JSONResponse({"error": "No images provided"}, status_code=400)

    # Build the messages payload in the shape pipeline.process expects
    # (identical to glmocr.server's sync handler).
    messages = [{"role": "user", "content": []}]
    for image_url in images:
        messages[0]["content"].append(
            {"type": "image_url", "image_url": {"url": image_url}}
        )
    request_data = {"messages": messages}

    pipeline = app.state.pipeline

    def _run_sync() -> list:
        # pipeline.process is a generator that yields one result per input
        # unit. We materialize it inside the thread so the caller gets the
        # full list.
        return list(
            pipeline.process(request_data, save_layout_visualization=False)
        )

    try:
        results = await asyncio.to_thread(_run_sync)
    except Exception as e:
        logger.error("parse error: %s", e)
        logger.debug(traceback.format_exc())
        return JSONResponse(
            {"error": f"Parse error: {e}"}, status_code=500,
        )

    if not results:
        return JSONResponse(_build_response(None, ""), status_code=200)
    if len(results) == 1:
        r = results[0]
        return JSONResponse(
            _build_response(r.json_result, r.markdown_result or ""),
            status_code=200,
        )
    json_result = [r.json_result for r in results]
    markdown_result = "\n\n---\n\n".join(
        r.markdown_result or "" for r in results
    )
    return JSONResponse(
        _build_response(json_result, markdown_result), status_code=200,
    )
