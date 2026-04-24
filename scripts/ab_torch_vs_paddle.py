"""Head-to-head: current backend vs stored torch smoke results.

Re-fetches markdown + json_result for the 10 original smoke-test pages
against whichever backend is currently live (check docker logs to know
which one), compares detection counts and markdown char-length against
the torch baseline stored in results/*.md from the initial session.

Invoke:
    PYTHONIOENCODING=utf-8 python scripts/ab_torch_vs_paddle.py
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results"
CONTAINER_PREFIX = "file:///app/datasets/OmniDocBench/images"
ENDPOINT = "http://localhost:5002/glmocr/parse"
TIMEOUT = 240


def extract_torch_markdown(md_file: Path) -> str:
    """Pull the actual OCR text out of the wrapping smoke-test md."""
    text = md_file.read_text(encoding="utf-8")
    m = re.search(r"## markdown_result\n\n(.*)", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def source_image_name_for(md_file: Path) -> str:
    """Figure the image filename from the wrapped md.
    We stored 'Source: datasets/.../images/<name>.(png|jpg|jpeg)' as a line."""
    text = md_file.read_text(encoding="utf-8")
    m = re.search(r"`datasets/OmniDocBench/images/([^`]+)`", text)
    if not m:
        raise RuntimeError(f"no source line in {md_file}")
    return m.group(1)


def fetch_current(image_name: str) -> tuple[str, int]:
    body = json.dumps({"images": [f"{CONTAINER_PREFIX}/{image_name}"]}).encode()
    req = urllib.request.Request(ENDPOINT, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        d = json.loads(r.read())
    md = d.get("markdown_result") or ""
    jr = d.get("json_result") or [[]]
    n_blocks = len(jr[0]) if jr and jr[0] else 0
    return md, n_blocks


def main() -> None:
    md_files = sorted(RESULTS_DIR.glob("*.md"))
    if not md_files:
        sys.exit(f"no stored torch smoke results in {RESULTS_DIR}")

    print(f"{'image':<60}  {'md_torch':>8} {'md_curr':>8} {'Δmd':>6}   {'blk_curr':>8}")
    print("-" * 100)
    totals = {"md_t": 0, "md_c": 0, "blk_c": 0}
    for md_file in md_files[:10]:
        torch_md = extract_torch_markdown(md_file)
        image_name = source_image_name_for(md_file)
        curr_md, curr_blk = fetch_current(image_name)
        dmd = len(curr_md) - len(torch_md)
        pct = (dmd / len(torch_md) * 100) if len(torch_md) else 0
        totals["md_t"] += len(torch_md)
        totals["md_c"] += len(curr_md)
        totals["blk_c"] += curr_blk
        print(f"  {image_name[:58]:<58}  {len(torch_md):>8} {len(curr_md):>8} {pct:>+5.0f}%   {curr_blk:>8}")
    print("-" * 100)
    dpct = ((totals["md_c"] - totals["md_t"]) / totals["md_t"] * 100) if totals["md_t"] else 0
    print(f"  {'TOTAL':<58}  {totals['md_t']:>8} {totals['md_c']:>8} {dpct:>+5.0f}%   {totals['blk_c']:>8}")


if __name__ == "__main__":
    main()
