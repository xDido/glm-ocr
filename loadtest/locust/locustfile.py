"""Locust load scenario for the CPU glmocr container.

Run headless:
    locust -f locustfile.py --headless -u 50 -r 5 -t 2m \
           --host http://localhost:5002 --csv ../results/locust

Env overrides:
    LOCUST_IMAGES  - comma-separated image URLs (required; set by scripts/omnidoc_locust.sh)
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, events, task


def _images() -> list[str]:
    raw = os.environ.get("LOCUST_IMAGES")
    if not raw:
        raise RuntimeError(
            "locustfile.py: LOCUST_IMAGES env var is required. Run via "
            "scripts/omnidoc_locust.sh, or set LOCUST_IMAGES=<csv> yourself."
        )
    return [u.strip() for u in raw.split(",") if u.strip()]


IMAGES = _images()


class OCRUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def parse_single(self) -> None:
        body = {"images": [random.choice(IMAGES)]}
        with self.client.post(
            "/glmocr/parse",
            json=body,
            name="parse:single",
            catch_response=True,
            timeout=300,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status={resp.status_code} body={resp.text[:200]}")

    @task(1)
    def parse_batch(self) -> None:
        body = {"images": random.sample(IMAGES, k=min(2, len(IMAGES)))}
        with self.client.post(
            "/glmocr/parse",
            json=body,
            name="parse:batch",
            catch_response=True,
            timeout=300,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status={resp.status_code} body={resp.text[:200]}")


@events.test_start.add_listener
def _on_start(environment, **_kw) -> None:  # pragma: no cover
    print(f"[locust] images={IMAGES}")
