"""Quick OCR-quality smoke over 10 OmniDocBench pages.

Picks 10 deterministic docs from datasets/OmniDocBench/images/, POSTs each
to the CPU container's /glmocr/parse endpoint (which proxies into SGLang),
and writes one markdown file per doc to results/ containing:
  - source image + container-visible URL
  - HTTP status + wall-clock latency
  - the markdown_result body GLM-OCR produced

Stdlib-only so it runs from a Windows host without extra deps.

Usage:
    python scripts/ocr_quality_test.py
"""
from __future__ import annotations

import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IMAGES_DIR = REPO / "datasets" / "OmniDocBench" / "images"
CONTAINER_PREFIX = "file:///app/datasets/OmniDocBench/images"
RESULTS_DIR = REPO / "results"
ENDPOINT = "http://localhost:5002/glmocr/parse"
N = 10
SEED = 42
TIMEOUT = 600  # per-request, seconds


def pick_images() -> list[Path]:
    all_images = sorted(IMAGES_DIR.iterdir())
    if len(all_images) < N:
        sys.exit(f"only {len(all_images)} images in {IMAGES_DIR}, need {N}")
    rng = random.Random(SEED)
    return rng.sample(all_images, N)


def post_one(image: Path) -> tuple[int, float, dict | None, str | None]:
    body = json.dumps({"images": [f"{CONTAINER_PREFIX}/{image.name}"]}).encode()
    req = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            elapsed = time.perf_counter() - started
            raw = resp.read()
            return resp.status, elapsed, json.loads(raw), None
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - started
        return e.code, elapsed, None, e.read().decode(errors="replace")[:2000]
    except Exception as e:
        elapsed = time.perf_counter() - started
        return 0, elapsed, None, f"{type(e).__name__}: {e}"


def extract_markdown(payload: dict) -> str:
    """glmocr returns {"results": [{"json_result":..., "markdown_result":...}]}
    (list-per-image) but some builds flatten to a single object. Handle both."""
    if not isinstance(payload, dict):
        return f"(unexpected payload type: {type(payload).__name__})"
    if "results" in payload and isinstance(payload["results"], list) and payload["results"]:
        first = payload["results"][0]
        if isinstance(first, dict):
            md = first.get("markdown_result") or first.get("markdown")
            if md is not None:
                return md
    for k in ("markdown_result", "markdown"):
        if k in payload:
            return payload[k]
    return "(no markdown field found; raw payload below)\n\n```json\n" + json.dumps(payload, indent=2, ensure_ascii=False)[:8000] + "\n```"


def write_md(image: Path, status: int, latency: float, payload: dict | None, err: str | None) -> Path:
    out = RESULTS_DIR / f"{image.stem}.md"
    lines = [
        f"# OCR result — {image.name}",
        "",
        f"- **Source**: `{image.relative_to(REPO).as_posix()}`",
        f"- **Container URL**: `{CONTAINER_PREFIX}/{image.name}`",
        f"- **HTTP status**: `{status}`",
        f"- **Latency**: `{latency:.2f}s`",
        "",
    ]
    if err is not None:
        lines += ["## Error", "", "```", err, "```"]
    else:
        md = extract_markdown(payload or {})
        lines += ["## markdown_result", "", md]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def wait_healthy(timeout_s: int = 600) -> None:
    """Block until /health returns 200 or timeout."""
    health = ENDPOINT.replace("/glmocr/parse", "/health")
    deadline = time.time() + timeout_s
    last_err = "unset"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=5) as r:
                if r.status == 200:
                    print(f"[wait] {health} OK", flush=True)
                    return
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(3)
    sys.exit(f"[wait] {health} never returned 200 within {timeout_s}s (last: {last_err})")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"[plan] endpoint={ENDPOINT} n={N} seed={SEED}", flush=True)
    wait_healthy()
    images = pick_images()
    print("[plan] selected:", flush=True)
    for img in images:
        print(f"   - {img.name}", flush=True)

    ok = fail = 0
    for i, img in enumerate(images, 1):
        print(f"[{i}/{N}] {img.name} ...", flush=True, end=" ")
        status, elapsed, payload, err = post_one(img)
        path = write_md(img, status, elapsed, payload, err)
        tag = "OK " if status == 200 and err is None else "FAIL"
        print(f"{tag} status={status} {elapsed:.1f}s -> {path.name}", flush=True)
        if tag.strip() == "OK":
            ok += 1
        else:
            fail += 1

    print(f"\n[done] ok={ok} fail={fail} results_dir={RESULTS_DIR}", flush=True)
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
