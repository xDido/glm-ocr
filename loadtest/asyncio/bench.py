"""
asyncio + aiohttp micro-bench for the GLM-OCR CPU container.

Fires N concurrent POSTs to /glmocr/parse, measures wall-clock latency per
request, prints throughput + p50/p95/p99. Lightweight, no Locust/k6
required — good for fast concurrency sweeps while tuning SGLang batching.

Examples:
    scripts/omnidoc_asyncio.sh        # full OmniDocBench run
    python bench.py --host http://localhost:5002 --concurrency 16 --total 128 \
        --image-url file:///app/datasets/OmniDocBench/images/<name>.jpg
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
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
    p.add_argument(
        "--host", default="http://localhost:5002", help="CPU container base URL"
    )
    p.add_argument("--endpoint", default="/glmocr/parse")
    p.add_argument(
        "--concurrency", type=int, default=8, help="concurrent in-flight requests"
    )
    p.add_argument("--total", type=int, default=64, help="total requests to send")
    p.add_argument(
        "--image-url",
        action="append",
        help="image URL(s) to sample from; repeatable. Each "
        "request picks one at random. Defaults to the "
        "samples served inside the container.",
    )
    p.add_argument(
        "--image-list-file",
        type=Path,
        default=None,
        help="File with one image URL per line. Merged with "
        "--image-url. Useful when the pool is large.",
    )
    p.add_argument(
        "--timeout", type=float, default=300.0, help="per-request timeout (s)"
    )
    p.add_argument(
        "--warmup", type=int, default=2, help="warmup requests (excluded from stats)"
    )
    p.add_argument(
        "--interval-seconds",
        type=float,
        default=0.0,
        help="baseline/paced mode: fire requests serially, "
        "sleeping this many seconds between submissions "
        "(measured from the start of one request to the "
        "start of the next). When >0, --concurrency is "
        "forced to 1 and --max-fail-rate abort is "
        "evaluated between requests.",
    )
    p.add_argument(
        "--pool-seed",
        type=int,
        default=None,
        help="seed the per-request random pool draw for "
        "reproducibility; unset = non-deterministic",
    )
    p.add_argument(
        "--max-fail-rate",
        type=float,
        default=None,
        help="abort the run mid-flight once fail/attempted "
        "exceeds this fraction after --min-sample-for-abort "
        "observations. Unset = never abort.",
    )
    p.add_argument(
        "--min-sample-for-abort",
        type=int,
        default=40,
        help="minimum completed requests before --max-fail-rate "
        "can trigger an abort (guards against unlucky "
        "warmup streaks).",
    )
    p.add_argument("--json-out", type=Path, default=None, help="write results as JSON")
    p.add_argument(
        "--pushgateway-url",
        type=str,
        default=None,
        help="Prometheus pushgateway URL (e.g. "
        "http://localhost:9091). When set, pushes final "
        "summary as glmocr_asyncio_* metrics so Grafana can "
        "chart them alongside server-side metrics.",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run identifier label applied to pushed metrics.",
    )
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


async def one_call(
    session: aiohttp.ClientSession, url: str, body: dict[str, Any], timeout: float
) -> tuple[bool, float, int, str | None, str]:
    """Returns (ok, elapsed_s, status, err_str_or_None, image_url). The
    image_url is captured so the caller can build per-doc failure details."""
    image_url = (body.get("images") or ["?"])[0]
    start = time.perf_counter()
    try:
        async with session.post(
            url, json=body, timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            payload = await resp.read()
            elapsed = time.perf_counter() - start
            ok = resp.status == 200
            err = None if ok else f"status={resp.status} body={payload[:200]!r}"
            return ok, elapsed, resp.status, err, image_url
    except Exception as exc:
        return False, time.perf_counter() - start, 0, repr(exc), image_url


def _resolve_image_pool(args: argparse.Namespace) -> list[str]:
    pool: list[str] = list(args.image_url or [])
    if args.image_list_file:
        with args.image_list_file.open("r", encoding="utf-8") as fh:
            pool.extend(
                line.strip()
                for line in fh
                if line.strip() and not line.lstrip().startswith("#")
            )
    if not pool:
        sys.stderr.write(
            "bench.py: no image pool — pass --image-url or --image-list-file, "
            "or run via scripts/omnidoc_asyncio.sh\n"
        )
        sys.exit(2)
    return pool


async def run(args: argparse.Namespace) -> dict[str, Any]:
    url = args.host.rstrip("/") + args.endpoint
    pool = _resolve_image_pool(args)

    if args.pool_seed is not None:
        random.seed(args.pool_seed)

    paced = args.interval_seconds > 0
    effective_concurrency = 1 if paced else args.concurrency
    sem = asyncio.Semaphore(effective_concurrency)
    results: list[tuple[bool, float, int, str | None, str]] = []
    aborted_event = asyncio.Event()

    async with aiohttp.ClientSession() as session:

        async def worker(_i: int) -> None:
            # If abort already tripped before the slot opened, don't even
            # send the request — saves the GPU from pointless work.
            if aborted_event.is_set():
                return
            # Each request independently samples one image from the pool;
            # same URL when pool size == 1, random otherwise.
            body = {"images": [random.choice(pool)]}
            async with sem:
                if aborted_event.is_set():
                    return
                r = await one_call(session, url, body, args.timeout)
            results.append(r)
            if (
                args.max_fail_rate is not None
                and len(results) >= args.min_sample_for_abort
            ):
                fails = sum(1 for x in results if not x[0])
                if fails / len(results) > args.max_fail_rate:
                    aborted_event.set()

        if args.warmup:
            print(f"[warmup] firing {args.warmup} request(s)...")
            await asyncio.gather(*[worker(-1) for _ in range(args.warmup)])
            results.clear()

        print(
            f"[bench] host={args.host} total={args.total} "
            f"concurrency={effective_concurrency} pool_size={len(pool)}"
            + (f" interval={args.interval_seconds}s" if paced else "")
            + (
                f" max_fail_rate={args.max_fail_rate:.0%}"
                if args.max_fail_rate is not None
                else ""
            )
        )
        wall_start = time.perf_counter()
        if paced:
            # Serial paced mode: fire -> complete -> sleep remaining slice.
            # Sleep is measured from each request's start, so if a request
            # takes longer than the interval we send the next one
            # immediately (no piling up).
            for i in range(args.total):
                if aborted_event.is_set():
                    break
                tick = time.perf_counter()
                await worker(i)
                elapsed = time.perf_counter() - tick
                if i < args.total - 1:
                    remaining = args.interval_seconds - elapsed
                    if remaining > 0:
                        await asyncio.sleep(remaining)
        else:
            await asyncio.gather(
                *[worker(i) for i in range(args.total)], return_exceptions=True
            )
        wall = time.perf_counter() - wall_start
        if aborted_event.is_set():
            print(
                f"[bench] aborted at {len(results)}/{args.total} "
                f"(fail% exceeded {args.max_fail_rate:.0%})"
            )

    oks = [r for r in results if r[0]]
    fails = [r for r in results if not r[0]]
    ok_lat_ms = [r[1] * 1000 for r in oks]

    summary = {
        "host": args.host,
        "endpoint": args.endpoint,
        "concurrency": effective_concurrency,
        "interval_seconds": args.interval_seconds if paced else 0.0,
        "total": args.total,
        "pool_size": len(pool),
        "pool_seed": args.pool_seed,
        "aborted": aborted_event.is_set(),
        "requests_attempted": len(results),
        "max_fail_rate": args.max_fail_rate,
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
        # Per-failure breakdown: which doc failed, how long it spent
        # before failing, what the error was. Useful to identify
        # pathological documents in the pool.
        "failure_details": [
            {
                "image_url": r[4],
                "elapsed_ms": r[1] * 1000,
                "status": r[2],
                "error": r[3],
            }
            for r in fails
        ],
    }
    return summary


def print_summary(s: dict[str, Any]) -> None:
    lat = s["latency_ms"]
    print()
    print("=" * 60)
    print(
        f"  requests       : {s['total']} ({s['successes']} ok, {s['failures']} fail)"
    )
    print(f"  concurrency    : {s['concurrency']}")
    print(f"  wall time      : {s['wall_seconds']:.2f}s")
    print(f"  throughput     : {s['throughput_rps']:.2f} req/s")
    print(
        f"  latency ms     : p50={lat['p50']:.0f}  p90={lat['p90']:.0f}  "
        f"p95={lat['p95']:.0f}  p99={lat['p99']:.0f}"
    )
    print(
        f"                   mean={lat['mean']:.0f}  min={lat['min']:.0f}  max={lat['max']:.0f}"
    )
    if s["error_samples"]:
        print("  error samples  :")
        for e in s["error_samples"]:
            print(f"    - {e}")
    print("=" * 60)


def push_to_pushgateway(url: str, run_id: str, summary: dict[str, Any]) -> None:
    """Push final summary as glmocr_asyncio_* metrics to Prometheus
    Pushgateway. Silently no-op if prometheus_client isn't installed."""
    try:
        from prometheus_client import (  # type: ignore
            CollectorRegistry,
            Gauge,
            push_to_gateway,
        )
    except ImportError:
        print(
            "[bench] prometheus_client not installed; skipping pushgateway "
            "(pip install prometheus-client to enable)",
            file=sys.stderr,
        )
        return

    registry = CollectorRegistry()
    common_labels = ["run_id"]
    label_values = [run_id or ""]

    flat = {
        "throughput_rps": summary.get("throughput_rps", 0.0),
        "successes": summary.get("successes", 0),
        "failures": summary.get("failures", 0),
        "concurrency": summary.get("concurrency", 0),
        "wall_seconds": summary.get("wall_seconds", 0.0),
        "latency_p50_ms": summary["latency_ms"].get("p50", 0.0),
        "latency_p90_ms": summary["latency_ms"].get("p90", 0.0),
        "latency_p95_ms": summary["latency_ms"].get("p95", 0.0),
        "latency_p99_ms": summary["latency_ms"].get("p99", 0.0),
        "latency_mean_ms": summary["latency_ms"].get("mean", 0.0),
        "latency_max_ms": summary["latency_ms"].get("max", 0.0),
    }
    for name, value in flat.items():
        try:
            fval = float(value) if value == value else 0.0  # NaN guard
        except (TypeError, ValueError):
            continue
        g = Gauge(
            f"glmocr_asyncio_{name}",
            f"asyncio bench: {name}",
            labelnames=common_labels,
            registry=registry,
        )
        g.labels(*label_values).set(fval)

    try:
        push_to_gateway(url, job="glmocr_asyncio", registry=registry)
        print(f"[bench] pushed summary to {url}")
    except Exception as exc:  # pragma: no cover
        print(f"[bench] pushgateway push failed: {exc!r}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    summary = asyncio.run(run(args))
    print_summary(summary)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2))
        print(f"[bench] wrote {args.json_out}")
    if args.pushgateway_url:
        push_to_pushgateway(args.pushgateway_url, args.run_id, summary)
    return 0 if summary["failures"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
