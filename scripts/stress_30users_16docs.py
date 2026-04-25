"""Concurrent stress test: 30 users, each sending a 16-page document.

Each "user" is a single HTTP POST to /glmocr/parse with `images` containing
16 distinct OmniDocBench page URLs. All 30 users fire concurrently (one
gthread per user). The 480 unique pages are partitioned 16-per-user
deterministically (seed=42) so users don't share inputs — this is a
worst-case stress for KV-cache reuse, the opposite of a friendly
prefix-cache scenario.

Output: per-user latency, error decomposition, server-side stage
breakdown via /metrics deltas, plus SGLang-side TTFT/decode/cache stats.

Notes:
- GUNICORN_TIMEOUT=480 (8 min) is the per-request kill ceiling. Some
  users will likely exceed this on the 8 GB dev card. That's data.
- Client timeout set to 1500s so gunicorn-side timeouts surface as the
  failure mode, not client-side.
- 480 pages × ~17 regions/page ≈ 8000 SGLang calls total. At the
  shipped c=16 warm steady-state (~0.63 rps for single-page reqs),
  expect total wall in the 5-30 minute range.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import random
import re
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IMAGES_DIR = REPO / "datasets" / "OmniDocBench" / "images"
ENDPOINT = "http://localhost:5002/glmocr/parse"
CPU_METRICS = "http://localhost:5002/metrics"
SGL_METRICS = "http://localhost:30000/metrics"
USERS = 30
PAGES_PER_USER = 16
SEED = 42
CLIENT_TIMEOUT = 1500


def fetch_metrics(url: str) -> dict[str, float]:
    out: dict[str, float] = {}
    txt = urllib.request.urlopen(url, timeout=10).read().decode()
    for line in txt.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r'^([a-zA-Z0-9_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$', line)
        if m:
            out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
    return out


def fire_user(user_id: int, image_paths: list[Path]) -> dict:
    images = [f"file:///app/datasets/OmniDocBench/images/{p.name}" for p in image_paths]
    body = json.dumps({"images": images}).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=CLIENT_TIMEOUT) as r:
            raw = r.read()
            status = r.status
        dt = time.perf_counter() - t0
        d = json.loads(raw)
        jr = d.get("json_result") or []
        md = d.get("markdown_result") or ""
        # multi-image responses come back with json_result as a list of per-image results
        n_pages = len(jr) if isinstance(jr, list) and jr and isinstance(jr[0], list) else (1 if md else 0)
        n_blocks = sum(len(p) for p in jr) if isinstance(jr, list) and jr and isinstance(jr[0], list) else (len(jr[0]) if isinstance(jr, list) and jr else 0)
        return {
            "user": user_id, "status": "ok", "http": status, "wall_s": dt,
            "pages_returned": n_pages, "md_chars": len(md), "blocks": n_blocks,
        }
    except urllib.error.HTTPError as e:
        return {"user": user_id, "status": "http_err", "http": e.code,
                "wall_s": time.perf_counter() - t0,
                "error": f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"}
    except Exception as e:
        return {"user": user_id, "status": "exception",
                "wall_s": time.perf_counter() - t0,
                "error": f"{type(e).__name__}: {str(e)[:200]}"}


def main() -> None:
    rng = random.Random(SEED)
    all_imgs = sorted(IMAGES_DIR.iterdir())
    if len(all_imgs) < USERS * PAGES_PER_USER:
        raise RuntimeError(f"need {USERS * PAGES_PER_USER} images, found {len(all_imgs)}")
    selected = rng.sample(all_imgs, USERS * PAGES_PER_USER)
    user_groups = [selected[i * PAGES_PER_USER:(i + 1) * PAGES_PER_USER] for i in range(USERS)]

    print(f"[stress] {USERS} users × {PAGES_PER_USER} pages = {USERS * PAGES_PER_USER} total pages")
    print(f"[stress] all users fire concurrently with c={USERS}")
    print(f"[stress] client timeout per user: {CLIENT_TIMEOUT}s   gunicorn cap: 480s\n", flush=True)

    print("[stress] taking pre-run metrics snapshot...", flush=True)
    cpu0 = fetch_metrics(CPU_METRICS)
    sgl0 = fetch_metrics(SGL_METRICS)

    print("[stress] firing all users...", flush=True)
    t_start = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=USERS) as ex:
        futures = {ex.submit(fire_user, i, user_groups[i]): i for i in range(USERS)}
        results = []
        completed = 0
        for fut in cf.as_completed(futures):
            r = fut.result()
            completed += 1
            tag = "OK " if r["status"] == "ok" else "ERR"
            print(f"  [{completed:>2}/{USERS}] user {r['user']:>2}: {tag} {r['wall_s']:>6.1f}s  "
                  f"pages={r.get('pages_returned', 0):>2} blocks={r.get('blocks', 0):>3} "
                  f"md={r.get('md_chars', 0):>5}c"
                  + (f"  err={r['error'][:80]}" if 'error' in r else ""),
                  flush=True)
            results.append(r)
    total_wall = time.perf_counter() - t_start

    print("[stress] taking post-run metrics snapshot...", flush=True)
    cpu1 = fetch_metrics(CPU_METRICS)
    sgl1 = fetch_metrics(SGL_METRICS)

    # ---------------- aggregate ----------------
    ok = [r for r in results if r["status"] == "ok"]
    errs = [r for r in results if r["status"] != "ok"]
    lats = sorted([r["wall_s"] for r in ok])
    total_pages = sum(r.get("pages_returned", 0) for r in ok)
    total_blocks = sum(r.get("blocks", 0) for r in ok)
    total_md = sum(r.get("md_chars", 0) for r in ok)

    print(f"\n=========================================================")
    print(f"30-USER × 16-PAGE STRESS TEST RESULTS")
    print(f"=========================================================")
    print(f"total wall                {total_wall:.1f}s ({total_wall/60:.1f} min)")
    print(f"successful users          {len(ok)}/{USERS}")
    print(f"failed users              {len(errs)}")
    print(f"total pages OCR'd         {total_pages}/{USERS * PAGES_PER_USER}")
    print(f"total blocks detected     {total_blocks:,}")
    print(f"total markdown chars      {total_md:,}")
    print(f"effective document rps    {len(ok) / total_wall:.3f}")
    print(f"effective page rps        {total_pages / total_wall:.3f}")
    print(f"effective block rps       {total_blocks / total_wall:.2f}")
    print()
    if lats:
        print("per-USER (16-page request) latency:")
        print(f"  mean      {sum(lats) / len(lats):>7.1f}s")
        print(f"  p50       {lats[len(lats) // 2]:>7.1f}s")
        print(f"  p95       {lats[min(len(lats) - 1, int(len(lats) * 0.95))]:>7.1f}s")
        print(f"  p99       {lats[min(len(lats) - 1, int(len(lats) * 0.99))]:>7.1f}s")
        print(f"  min       {lats[0]:>7.1f}s")
        print(f"  max       {lats[-1]:>7.1f}s")
    if errs:
        print("\nerrors:")
        for e in errs:
            print(f"  user {e['user']:>2}: {e['status']:>10s}  {e.get('error', '')[:120]}")

    # ---------------- server-stage decomposition ----------------
    def cd(k: str) -> float: return cpu1.get(k, 0) - cpu0.get(k, 0)
    def sd(k: str) -> float: return sgl1.get(k, 0) - sgl0.get(k, 0)

    parse_n = 0
    parse_s = 0
    for k, v in cpu1.items():
        if k.startswith('flask_http_request_duration_seconds_count') and 'glmocr/parse' in k:
            parse_n += v - cpu0.get(k, 0)
        if k.startswith('flask_http_request_duration_seconds_sum') and 'glmocr/parse' in k:
            parse_s += v - cpu0.get(k, 0)
    layout_n = cd('glmocr_layout_seconds_count')
    layout_s = cd('glmocr_layout_seconds_sum')
    region_n = cd('glmocr_ocr_region_seconds_count')
    region_s = cd('glmocr_ocr_region_seconds_sum')

    ttft_s = 0
    ttft_n = 0
    e2e_s = 0
    e2e_n = 0
    cache_tok = 0
    compute_tok = 0
    decode_tok = 0
    for k in sgl1:
        if k.startswith('sglang:time_to_first_token_seconds_sum'):
            ttft_s += sd(k)
        if k.startswith('sglang:time_to_first_token_seconds_count'):
            ttft_n += sd(k)
        if k.startswith('sglang:e2e_request_latency_seconds_sum'):
            e2e_s += sd(k)
        if k.startswith('sglang:e2e_request_latency_seconds_count'):
            e2e_n += sd(k)
        if 'mode="prefill_cache"' in k and 'realtime_tokens_total' in k:
            cache_tok += sd(k)
        if 'mode="prefill_compute"' in k and 'realtime_tokens_total' in k:
            compute_tok += sd(k)
        if 'mode="decode"' in k and 'realtime_tokens_total' in k:
            decode_tok += sd(k)

    hit_pct = (cache_tok / (cache_tok + compute_tok) * 100) if (cache_tok + compute_tok) else 0

    print(f"\n=========================================================")
    print(f"SERVER-SIDE DECOMPOSITION (deltas over the run)")
    print(f"=========================================================")
    print(f"CPU container (per /glmocr/parse request, mean):")
    print(f"  flask_http              n={parse_n:>4.0f}  mean={parse_s / parse_n if parse_n else 0:>7.1f}s")
    print(f"  glmocr_layout           n={layout_n:>4.0f}  mean={layout_s / layout_n if layout_n else 0:>7.2f}s/call")
    print(f"  glmocr_ocr_region       n={region_n:>4.0f}  mean={region_s / region_n if region_n else 0:>7.2f}s/region")
    print(f"  regions per request                {region_n / parse_n if parse_n else 0:>7.1f}")
    print(f"  pages per request                  {layout_n / parse_n if parse_n else 0:>7.1f} (layout calls)")
    print()
    print(f"SGLang (per region):")
    print(f"  e2e (queue+prefill+decode)         n={e2e_n:>5.0f}  mean={e2e_s / e2e_n if e2e_n else 0:>7.2f}s")
    print(f"  TTFT  (queue+prefill)              n={ttft_n:>5.0f}  mean={ttft_s / ttft_n if ttft_n else 0:>7.2f}s")
    print(f"  decode-only (e2e - TTFT)           mean={(e2e_s - ttft_s) / e2e_n if e2e_n else 0:>7.2f}s")
    print()
    print(f"SGLang token-throughput accounting:")
    print(f"  prefill_compute tokens             {compute_tok:>10,.0f}")
    print(f"  prefill_cache (hit) tokens         {cache_tok:>10,.0f}")
    print(f"  decode tokens generated            {decode_tok:>10,.0f}")
    print(f"  prefix cache hit rate              {hit_pct:>10.1f}%")

    out = REPO / "loadtest" / "results" / f"stress-30u-16d-2026-04-25.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"USERS": USERS, "PAGES_PER_USER": PAGES_PER_USER, "SEED": SEED, "CLIENT_TIMEOUT": CLIENT_TIMEOUT},
        "total_wall_s": total_wall,
        "results": results,
        "server_decomposition": {
            "cpu": {
                "parse_n": parse_n, "parse_mean_s": parse_s / parse_n if parse_n else 0,
                "layout_n": layout_n, "layout_mean_s": layout_s / layout_n if layout_n else 0,
                "region_n": region_n, "region_mean_s": region_s / region_n if region_n else 0,
            },
            "sgl": {
                "ttft_n": ttft_n, "ttft_mean_s": ttft_s / ttft_n if ttft_n else 0,
                "e2e_n": e2e_n, "e2e_mean_s": e2e_s / e2e_n if e2e_n else 0,
                "decode_only_mean_s": (e2e_s - ttft_s) / e2e_n if e2e_n else 0,
                "prefix_cache_hit_pct": hit_pct,
                "compute_tok": compute_tok, "cache_tok": cache_tok, "decode_tok": decode_tok,
            },
        },
    }, indent=2))
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
