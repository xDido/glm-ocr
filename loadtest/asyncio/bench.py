"""
asyncio + aiohttp micro-bench for the GLM-OCR CPU container.

Fires N concurrent POSTs to /glmocr/parse, measures wall-clock latency per
request, prints throughput + p50/p95/p99. Lightweight, no Locust/k6
required — good for fast concurrency sweeps while tuning SGLang batching.

Examples:
    python bench.py --host http://localhost:5002 --concurrency 16 --total 128
    python bench.py --host http://localhost:5002 --image-url file:///app/samples/receipt.png
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

try:
    import aiohttp
except ImportError:  # pragma: no cover
    sys.stderr.write("aiohttp missing: pip install aiohttp\n")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://localhost:5002",
                   help="CPU container base URL")
    p.add_argument("--endpoint", default="/glmocr/parse")
    p.add_argument("--concurrency", type=int, default=8,
                   help="concurrent in-flight requests")
    p.add_argument("--total", type=int, default=64,
                   help="total requests to send")
    p.add_argument("--image-url", action="append",
                   help="image URL(s) sent in the request body; repeatable. "
                        "defaults to the samples served inside the container")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="per-request timeout (s)")
    p.add_argument("--warmup", type=int, default=2,
                   help="warmup requests (excluded from stats)")
    p.add_argument("--json-out", type=Path, default=None,
                   help="write results as JSON")
    return p.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


async def one_call(session: aiohttp.ClientSession, url: str,
                   body: dict[str, Any], timeout: float) -> tuple[bool, float, int, str | None]:
    start = time.perf_counter()
    try:
        async with session.post(url, json=body,
                                timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            payload = await resp.read()
            elapsed = time.perf_counter() - start
            ok = resp.status == 200
            err = None if ok else f"status={resp.status} body={payload[:200]!r}"
            return ok, elapsed, resp.status, err
    except Exception as exc:
        return False, time.perf_counter() - start, 0, repr(exc)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    url = args.host.rstrip("/") + args.endpoint
    images = args.image_url or ["file:///app/samples/receipt.png"]
    body = {"images": images}

    sem = asyncio.Semaphore(args.concurrency)
    results: list[tuple[bool, float, int, str | None]] = []

    async with aiohttp.ClientSession() as session:
        async def worker(_i: int) -> None:
            async with sem:
                results.append(await one_call(session, url, body, args.timeout))

        if args.warmup:
            print(f"[warmup] firing {args.warmup} request(s)...")
            await asyncio.gather(*[worker(-1) for _ in range(args.warmup)])
            results.clear()

        print(f"[bench] host={args.host} total={args.total} concurrency={args.concurrency}")
        wall_start = time.perf_counter()
        await asyncio.gather(*[worker(i) for i in range(args.total)])
        wall = time.perf_counter() - wall_start

    oks = [r for r in results if r[0]]
    fails = [r for r in results if not r[0]]
    ok_lat_ms = [r[1] * 1000 for r in oks]

    summary = {
        "host": args.host,
        "endpoint": args.endpoint,
        "concurrency": args.concurrency,
        "total": args.total,
        "wall_seconds": wall,
        "throughput_rps": (len(oks) / wall) if wall > 0 else 0.0,
        "successes": len(oks),
        "failures": len(fails),
        "latency_ms": {
            "p50": percentile(ok_lat_ms, 50),
            "p90": percentile(ok_lat_ms, 90),
            "p95": percentile(ok_lat_ms, 95),
            "p99": percentile(ok_lat_ms, 99),
            "mean": statistics.fmean(ok_lat_ms) if ok_lat_ms else math.nan,
            "min": min(ok_lat_ms) if ok_lat_ms else math.nan,
            "max": max(ok_lat_ms) if ok_lat_ms else math.nan,
        },
        "error_samples": [r[3] for r in fails[:5]],
    }
    return summary


def print_summary(s: dict[str, Any]) -> None:
    lat = s["latency_ms"]
    print()
    print("=" * 60)
    print(f"  requests       : {s['total']} ({s['successes']} ok, {s['failures']} fail)")
    print(f"  concurrency    : {s['concurrency']}")
    print(f"  wall time      : {s['wall_seconds']:.2f}s")
    print(f"  throughput     : {s['throughput_rps']:.2f} req/s")
    print(f"  latency ms     : p50={lat['p50']:.0f}  p90={lat['p90']:.0f}  "
          f"p95={lat['p95']:.0f}  p99={lat['p99']:.0f}")
    print(f"                   mean={lat['mean']:.0f}  min={lat['min']:.0f}  max={lat['max']:.0f}")
    if s["error_samples"]:
        print(f"  error samples  :")
        for e in s["error_samples"]:
            print(f"    - {e}")
    print("=" * 60)


def main() -> int:
    args = parse_args()
    summary = asyncio.run(run(args))
    print_summary(summary)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2))
        print(f"[bench] wrote {args.json_out}")
    return 0 if summary["failures"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
