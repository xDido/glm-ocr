"""Poll /metrics from the CPU container AND SGLang every INTERVAL seconds,
append JSONL. One sample per line:

  in_flight         - glmocr_in_flight_requests (CPU container)
  sglang_running    - sglang:num_running_reqs   (GPU batch)
  sglang_queued     - sglang:num_queue_reqs     (GPU queue)
  cpu_cores_cpu     - rate of container_cpu_usage_seconds_total (CPU ctr)
  cpu_cores_sglang  - rate of container_cpu_usage_seconds_total (SGLang ctr)
  mem_rss_cpu_mb    - container_memory_rss (CPU ctr)   / 1MiB
  mem_rss_sglang_mb - container_memory_rss (SGLang ctr)/ 1MiB
  vram_used_mb      - DCGM_FI_DEV_FB_USED (GPU VRAM, MiB)
  vram_free_mb      - DCGM_FI_DEV_FB_FREE (GPU VRAM, MiB)
  gpu_util_pct      - DCGM_FI_DEV_GPU_UTIL (0-100)

We hit SGLang's /metrics directly (bypassing /runtime/summary) because
runtime_app._filter_sglang_metrics splits on the last space and therefore
treats the labelled form as the dict key - so sgl_metrics.get(
"sglang:num_running_reqs") always returns None. Parsing SGLang's
exposition ourselves gets the bare metric name regardless of labels.

Resource metrics (CPU/RAM/VRAM/GPU) are fetched via Prometheus HTTP API
because cAdvisor and DCGM exporter are scraped there already — no need
to duplicate the scrape here. Prometheus is optional: if the query fails
the field is null and the report renderer tolerates it.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

CPU_URL = os.environ.get("CPU_URL", "http://localhost:5002")
SGLANG_URL = os.environ.get("SGLANG_URL", "http://localhost:30000")
PROM_URL = os.environ.get("PROM_URL", "http://localhost:9090")
CPU_CONTAINER = os.environ.get("PROBE_CPU_CONTAINER", "glmocr-cpu")
SGL_CONTAINER = os.environ.get("PROBE_SGL_CONTAINER", "glmocr-sglang")
INTERVAL = float(os.environ.get("PROBE_INTERVAL", "2.0"))

# Captures "metric_name{optional labels}  value" and the metric_name is
# the first group regardless of whether labels are present.
METRIC_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+)",
    re.MULTILINE,
)


def _fetch_metric(url: str, name: str) -> float | None:
    """Grab the first value for `name` from a Prometheus /metrics page."""
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            text = r.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    for m in METRIC_LINE_RE.finditer(text):
        if m.group(1) == name:
            try:
                return float(m.group(2))
            except ValueError:
                continue
    return None


def _prom_query(expr: str) -> float | None:
    """Run an instant Prometheus query and return the first series value."""
    url = f"{PROM_URL}/api/v1/query?query=" + urllib.parse.quote(expr, safe="")
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            payload = json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception:
        return None
    if payload.get("status") != "success":
        return None
    result = payload.get("data", {}).get("result", [])
    if not result:
        return None
    try:
        return float(result[0]["value"][1])
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def main() -> int:
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "probe.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[probe] polling cpu={CPU_URL} sglang={SGLANG_URL} "
        f"every {INTERVAL}s -> {out_path}",
        flush=True,
    )

    cpu_metrics_url = f"{CPU_URL}/metrics"
    sglang_metrics_url = f"{SGLANG_URL}/metrics"

    # Prometheus expressions for container + GPU resource signals. Rate
    # windows of 30s are wide enough to smooth out the 5s scrape jitter
    # but narrow enough to track load-test phases.
    cpu_rate_cpu = (
        f'rate(container_cpu_usage_seconds_total{{name="{CPU_CONTAINER}"}}[30s])'
    )
    cpu_rate_sgl = (
        f'rate(container_cpu_usage_seconds_total{{name="{SGL_CONTAINER}"}}[30s])'
    )
    mem_cpu = f'container_memory_rss{{name="{CPU_CONTAINER}"}}'
    mem_sgl = f'container_memory_rss{{name="{SGL_CONTAINER}"}}'
    vram_used = 'DCGM_FI_DEV_FB_USED'
    vram_free = 'DCGM_FI_DEV_FB_FREE'
    gpu_util = 'DCGM_FI_DEV_GPU_UTIL'

    def _mb(val: float | None) -> float | None:
        if val is None:
            return None
        return val / (1024.0 * 1024.0)

    try:
        while True:
            t0 = time.time()
            sample = {
                "ts": t0,
                "in_flight": _fetch_metric(
                    cpu_metrics_url, "glmocr_in_flight_requests"
                ),
                "sglang_running": _fetch_metric(
                    sglang_metrics_url, "sglang:num_running_reqs"
                ),
                "sglang_queued": _fetch_metric(
                    sglang_metrics_url, "sglang:num_queue_reqs"
                ),
                "cpu_cores_cpu":     _prom_query(cpu_rate_cpu),
                "cpu_cores_sglang":  _prom_query(cpu_rate_sgl),
                "mem_rss_cpu_mb":    _mb(_prom_query(mem_cpu)),
                "mem_rss_sglang_mb": _mb(_prom_query(mem_sgl)),
                "vram_used_mb":      _prom_query(vram_used),
                "vram_free_mb":      _prom_query(vram_free),
                "gpu_util_pct":      _prom_query(gpu_util),
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(sample) + "\n")

            elapsed = time.time() - t0
            time.sleep(max(0.0, INTERVAL - elapsed))
    except KeyboardInterrupt:
        print(f"\n[probe] stopped, wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
