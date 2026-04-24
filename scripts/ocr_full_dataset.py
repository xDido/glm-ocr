"""OCR every page in OmniDocBench and emit one prediction .md per page.

Output dir: eval/predictions/<image_path_stem>.md   (matches evaluator schema)

Concurrency via a thread pool firing synchronous POSTs to /glmocr/parse.
Stdlib-only. Idempotent — skips pages whose .md already exists and is
non-empty, so rerunning after a crash resumes where it left off.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/ocr_full_dataset.py
    PYTHONIOENCODING=utf-8 python scripts/ocr_full_dataset.py --concurrency 24
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GT_JSON = REPO / "datasets" / "OmniDocBench" / "OmniDocBench.json"
IMAGES_DIR = REPO / "datasets" / "OmniDocBench" / "images"
CONTAINER_PREFIX = "file:///app/datasets/OmniDocBench/images"
OUT_DIR = REPO / "eval" / "predictions"
ENDPOINT = "http://localhost:5002/glmocr/parse"
TIMEOUT = 600


def load_pages() -> list[str]:
    with GT_JSON.open(encoding="utf-8") as f:
        gt = json.load(f)
    return [item["page_info"]["image_path"] for item in gt]


def extract_markdown(payload: dict) -> str:
    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list) and payload["results"]:
            first = payload["results"][0]
            if isinstance(first, dict):
                md = first.get("markdown_result") or first.get("markdown")
                if md is not None:
                    return md
        for k in ("markdown_result", "markdown"):
            if k in payload:
                return payload[k]
    return ""


def _single_attempt(image_path: str) -> tuple[str, int, float, str | None]:
    body = json.dumps({"images": [f"{CONTAINER_PREFIX}/{image_path}"]}).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            elapsed = time.perf_counter() - started
            payload = json.loads(resp.read())
            md = extract_markdown(payload)
            if not md:
                return "", resp.status, elapsed, "empty-markdown"
            return md, resp.status, elapsed, None
    except urllib.error.HTTPError as e:
        return "", e.code, time.perf_counter() - started, f"HTTP {e.code}: {e.read()[:400].decode(errors='replace')}"
    except Exception as e:
        return "", 0, time.perf_counter() - started, f"{type(e).__name__}: {e}"


def ocr_one(image_path: str, attempts: int = 2) -> tuple[str, str, int, float, str | None]:
    """-> (image_path, markdown, status, elapsed_s, error_or_none). Retries once on failure."""
    last = ("", 0, 0.0, "no-attempt")
    total_elapsed = 0.0
    for i in range(attempts):
        md, status, elapsed, err = _single_attempt(image_path)
        total_elapsed += elapsed
        if err is None and status == 200 and md:
            return image_path, md, status, total_elapsed, None
        last = (md, status, elapsed, err)
        if i + 1 < attempts:
            time.sleep(0.5 + i)  # small backoff
    md, status, _, err = last
    return image_path, md, status, total_elapsed, err


def already_done(image_path: str) -> bool:
    out = OUT_DIR / f"{Path(image_path).stem}.md"
    return out.exists() and out.stat().st_size > 0


def write_md(image_path: str, md: str) -> None:
    out = OUT_DIR / f"{Path(image_path).stem}.md"
    out.write_text(md, encoding="utf-8")


def wait_healthy(timeout_s: int = 600) -> None:
    health = ENDPOINT.replace("/glmocr/parse", "/health")
    deadline = time.time() + timeout_s
    last_err = "unset"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(3)
    sys.exit(f"[wait] {health} never became healthy within {timeout_s}s (last: {last_err})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="if >0, only process first N pages (debug)")
    ap.add_argument("--resume", action="store_true", default=True, help="skip pages whose .md exists (default on)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[plan] gt={GT_JSON.name} out={OUT_DIR} endpoint={ENDPOINT} c={args.concurrency}", flush=True)
    wait_healthy()

    pages = load_pages()
    if args.limit > 0:
        pages = pages[: args.limit]
    total = len(pages)

    pending = [p for p in pages if not already_done(p)] if args.resume else pages
    skipped = total - len(pending)
    print(f"[plan] total={total} pending={len(pending)} skipped={skipped}", flush=True)
    if not pending:
        print("[done] nothing to do", flush=True)
        return

    ok = fail = 0
    lat_samples: list[float] = []
    lock = threading.Lock()
    started_all = time.perf_counter()
    fail_log = OUT_DIR.parent / "ocr_failures.log"
    fail_log.write_text("", encoding="utf-8")

    def _progress() -> str:
        with lock:
            done = ok + fail
        eta_s = 0.0
        if lat_samples and done > 0:
            mean_wall = (time.perf_counter() - started_all) / done
            eta_s = mean_wall * (len(pending) - done)
        return f"[{ok + fail}/{len(pending)}] ok={ok} fail={fail} eta={eta_s/60:.1f}min"

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(ocr_one, p): p for p in pending}
        for i, fut in enumerate(as_completed(futures), 1):
            image_path, md, status, elapsed, err = fut.result()
            with lock:
                lat_samples.append(elapsed)
                if err is None and status == 200:
                    write_md(image_path, md)
                    ok += 1
                else:
                    fail += 1
                    write_md(image_path, "")
                    with fail_log.open("a", encoding="utf-8") as lf:
                        lf.write(f"{image_path}\tstatus={status}\terr={err}\n")
            if i % 25 == 0 or i == len(pending):
                print(_progress() + f"  last: {Path(image_path).stem[:60]} {status} {elapsed:.1f}s"
                      + (f" ERR={err[:120]}" if err else ""), flush=True)

    wall = time.perf_counter() - started_all
    mean_lat = sum(lat_samples) / len(lat_samples) if lat_samples else 0
    print(f"\n[done] ok={ok} fail={fail} wall={wall/60:.2f}min mean_latency={mean_lat:.2f}s "
          f"effective_rps={len(pending)/wall:.2f}", flush=True)
    sys.exit(0 if fail == 0 else 2)


if __name__ == "__main__":
    main()
