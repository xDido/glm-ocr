---
name: Load-test / observability domain familiarity
description: User is learning load-testing + observability terminology; prefers conceptual explanations before jumping to implementation
type: user
originSessionId: 588e706c-18b9-49b3-aeaf-23e564a90c10
---
The user is not deeply familiar with classical load-testing or observability
terminology. In one session they asked "what is a driver? what is the
difference between driver and worker and thread?" and asked to have
observability rendered as Grafana dashboards rather than JSON files.

**How to apply:**

- When introducing load-testing concepts (driver, worker, thread, VU,
  concurrency pools, percentiles), lead with a short conceptual explanation
  + a concrete mapping to *this* repo's config knobs (e.g., `CPU_WORKERS=2`,
  `CPU_THREADS=8`). Don't assume the vocabulary.
- Default to visual / dashboard-style observability (Grafana, charts) over
  raw JSON dumps or terminal tables when the user wants to "see" system
  behavior during load.
- Keep the layer-by-layer view explicit: inbound HTTP pool → per-request
  fan-out → GPU batching. The user finds it easier to reason about the
  system when each concurrency layer is named and located.
- Safe to use technical precision — but define each term the first time it
  appears in a thread.
