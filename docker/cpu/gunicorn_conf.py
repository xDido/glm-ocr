"""Gunicorn hooks for Prometheus multi-worker metrics + SIGKILL diagnosis.

Prometheus cleanup:
  prometheus-flask-exporter's GunicornPrometheusMetrics aggregates counters
  across gunicorn workers via files in $PROMETHEUS_MULTIPROC_DIR. When a
  worker exits, its files must be cleaned up so dead workers' values don't
  linger in the aggregation. `mark_process_dead` does that.

SIGKILL diagnosis:
  When gunicorn's master decides a worker is stuck, it sends SIGTERM first.
  If the worker doesn't exit inside --graceful-timeout, master escalates
  to SIGKILL (which can't be caught). To see what each thread was doing
  at the moment of SIGTERM, we register Python's faulthandler to dump
  ALL thread stacks to stderr on SIGTERM, BEFORE gunicorn's own SIGTERM
  handler runs. The dumps show up in `docker logs glmocr-cpu` with the
  worker's pid so we can correlate with the later SIGKILL error line.
"""
from __future__ import annotations

import faulthandler
import os
import signal
import sys
import time


def post_fork(server, worker):
    """Per-worker setup: enable faulthandler + register SIGTERM/SIGUSR1
    dumps. Runs inside each worker process right after fork."""
    try:
        faulthandler.enable()
        # Dump all thread stacks on SIGTERM (gunicorn's graceful-shutdown
        # signal). `chain=True` then runs gunicorn's own SIGTERM handler.
        faulthandler.register(signal.SIGTERM, all_threads=True, chain=True)
        # Also on SIGUSR1 so operators can inspect a live worker:
        #   `docker exec glmocr-cpu kill -USR1 <pid>`
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
        sys.stderr.write(
            f"[gunicorn_conf] worker pid={os.getpid()} "
            f"registered faulthandler on SIGTERM/SIGUSR1\n"
        )
        sys.stderr.flush()
    except Exception as exc:  # pragma: no cover
        # Never block a worker from starting because of diagnostic setup.
        sys.stderr.write(
            f"[gunicorn_conf] post_fork faulthandler setup failed: {exc!r}\n"
        )


def worker_int(worker):
    """Called when the master sends SIGINT (Ctrl+C path). Note-level log."""
    worker.log.warning(
        f"[gunicorn_conf] worker {worker.pid} got SIGINT "
        f"(alive {int(time.time() - getattr(worker, 'start_time', time.time()))}s)"
    )


def worker_abort(worker):
    """Called on SIGABRT. Gunicorn raises this when the worker hangs past
    --timeout. We log thread states so we can see what was stuck."""
    sys.stderr.write(
        f"[gunicorn_conf] worker {worker.pid} SIGABRT (stuck past --timeout); "
        f"thread dump follows:\n"
    )
    sys.stderr.flush()
    faulthandler.dump_traceback(all_threads=True)
    sys.stderr.flush()


def child_exit(server, worker):
    try:
        from prometheus_client import multiprocess  # type: ignore
    except ImportError:
        return
    multiprocess.mark_process_dead(worker.pid)
