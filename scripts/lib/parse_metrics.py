"""Minimal Prometheus text-exposition parser for load-test reports.

Designed for the CPU-focus renderer which needs to pull p50/p95/p99 out
of `glmocr_layout_seconds`, `glmocr_ocr_region_seconds`, and
`flask_http_request_duration_seconds` — all histograms exported in
multiproc mode by `docker/cpu/runtime_app.py`.

Two things to be careful about:

  1. Histograms reported in multiproc mode include `_bucket`, `_sum`,
     and `_count` series. The bucket counts are *cumulative* by `le`,
     so p95 is found by scanning for the first bucket whose count
     crosses `total * 0.95`.

  2. The `+Inf` bucket is the last one. If the target percentile falls
     inside it, returning the previous finite edge as a number would
     silently under-report the tail. We return `None` with an
     `overflow_label` like `">20s"` so the renderer can print that
     literal rather than a misleading value.

Not a general-purpose parser. Assumes well-formed exposition from
prometheus-flask-exporter + prometheus-client in multiproc mode; no
escaping, no exemplars, no stateset semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_labels(raw: str) -> dict[str, str]:
    """Parse `{k="v",k2="v2"}` → dict. Unescape `\"` and `\\`."""
    raw = raw.strip()
    if not raw or raw == "{}":
        return {}
    if raw.startswith("{"):
        raw = raw[1:]
    if raw.endswith("}"):
        raw = raw[:-1]
    out: dict[str, str] = {}
    i = 0
    while i < len(raw):
        eq = raw.find("=", i)
        if eq < 0:
            break
        key = raw[i:eq].strip()
        # value is always quoted
        if eq + 1 >= len(raw) or raw[eq + 1] != '"':
            break
        j = eq + 2
        val_chars: list[str] = []
        while j < len(raw):
            c = raw[j]
            if c == "\\" and j + 1 < len(raw):
                val_chars.append(raw[j + 1])
                j += 2
                continue
            if c == '"':
                break
            val_chars.append(c)
            j += 1
        out[key] = "".join(val_chars)
        i = j + 1
        # skip comma
        while i < len(raw) and raw[i] in (",", " "):
            i += 1
    return out


@dataclass
class Sample:
    name: str
    labels: dict[str, str]
    value: float


def parse_prom_text(text: str) -> list[Sample]:
    """Return every numeric sample as a `Sample`.

    Skips `# HELP`, `# TYPE`, blank lines, and lines that don't parse
    as `name{labels} value` or `name value`.
    """
    out: list[Sample] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # `name{labels} value` or `name value`. labels may be absent.
        brace = line.find("{")
        if brace >= 0:
            name = line[:brace]
            close = line.find("}", brace)
            if close < 0:
                continue
            labels = _parse_labels(line[brace:close + 1])
            rest = line[close + 1:].strip()
        else:
            sp = line.find(" ")
            if sp < 0:
                continue
            name = line[:sp]
            labels = {}
            rest = line[sp + 1:].strip()
        # value is first whitespace-separated token; ignore optional timestamp
        val_str = rest.split()[0] if rest else ""
        try:
            value = float(val_str)
        except ValueError:
            continue
        out.append(Sample(name=name, labels=labels, value=value))
    return out


# ---------------------------------------------------------------------------
# Histogram reconstruction
# ---------------------------------------------------------------------------


@dataclass
class Histogram:
    name: str                  # base name ("glmocr_layout_seconds")
    label_key: tuple[tuple[str, str], ...]  # non-`le` labels, as hashable
    buckets: list[tuple[float, float]]  # sorted (le, count), includes +Inf as math.inf
    count: float
    sum: float

    def mean(self) -> float | None:
        if self.count <= 0:
            return None
        return self.sum / self.count

    def max_approx(self) -> tuple[float | None, str | None]:
        """Upper-bound estimate of the worst observation.

        Prometheus histograms don't store exact max, so we return the
        smallest `le` bucket whose cumulative count reaches the total —
        i.e. the upper edge of the bucket that contains the max sample.
        If observations fall in the +Inf bucket, returns
        `(None, ">{last_finite_edge}s")` like `quantile`.
        """
        if self.count <= 0 or not self.buckets:
            return (None, None)
        last_finite: float | None = None
        for le, cnt in self.buckets:
            if le != float("inf") and cnt > 0:
                last_finite = le
            if cnt >= self.count:
                if le == float("inf"):
                    label = (
                        f">{last_finite:g}s" if last_finite is not None else ">Inf"
                    )
                    return (None, label)
                return (le, None)
        return (None, None)

    def quantile(self, q: float) -> tuple[float | None, str | None]:
        """Linear bucket-edge interpolation.

        Returns (value, overflow_label). If the target lands in the +Inf
        bucket, `value` is `None` and `overflow_label` is `">{last_finite_edge}s"`.
        """
        if self.count <= 0 or not self.buckets:
            return (None, None)
        target = q * self.count
        # buckets are cumulative, sorted ascending
        prev_edge = 0.0
        prev_count = 0.0
        last_finite_edge: float | None = None
        for le, cnt in self.buckets:
            if le == float("inf"):
                if target > prev_count:
                    # the target falls inside the +Inf bucket
                    label = (
                        f">{last_finite_edge:g}s"
                        if last_finite_edge is not None else ">Inf"
                    )
                    return (None, label)
                return (prev_edge, None)
            last_finite_edge = le
            if cnt >= target:
                # linear interp within [prev_edge, le)
                span_count = cnt - prev_count
                if span_count <= 0:
                    return (le, None)
                frac = (target - prev_count) / span_count
                return (prev_edge + frac * (le - prev_edge), None)
            prev_edge = le
            prev_count = cnt
        return (prev_edge, None)


def collect_histograms(samples: Iterable[Sample]) -> dict[tuple[str, tuple[tuple[str, str], ...]], Histogram]:
    """Group `_bucket`, `_sum`, `_count` samples back into `Histogram`s.

    Returned key is `(metric_base_name, tuple(sorted((k, v) for k, v in labels if k != "le")))`.
    """
    buckets: dict[tuple[str, tuple[tuple[str, str], ...]], list[tuple[float, float]]] = {}
    sums: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
    counts: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}

    for s in samples:
        if s.name.endswith("_bucket"):
            base = s.name[:-len("_bucket")]
            le_raw = s.labels.get("le")
            if le_raw is None:
                continue
            try:
                le = float("inf") if le_raw in ("+Inf", "Inf") else float(le_raw)
            except ValueError:
                continue
            key = _histogram_key(base, s.labels)
            buckets.setdefault(key, []).append((le, s.value))
        elif s.name.endswith("_sum"):
            base = s.name[:-len("_sum")]
            key = _histogram_key(base, s.labels)
            sums[key] = s.value
        elif s.name.endswith("_count"):
            base = s.name[:-len("_count")]
            key = _histogram_key(base, s.labels)
            counts[key] = s.value

    out: dict[tuple[str, tuple[tuple[str, str], ...]], Histogram] = {}
    for key, bkts in buckets.items():
        base, lbl = key
        bkts.sort(key=lambda t: t[0])
        out[key] = Histogram(
            name=base,
            label_key=lbl,
            buckets=bkts,
            count=counts.get(key, 0.0),
            sum=sums.get(key, 0.0),
        )
    return out


def _histogram_key(base: str, labels: dict[str, str]) -> tuple[str, tuple[tuple[str, str], ...]]:
    lbl = tuple(sorted((k, v) for k, v in labels.items() if k != "le"))
    return (base, lbl)


# ---------------------------------------------------------------------------
# Diffing (post − pre) so a report covers only the cell's observations
# ---------------------------------------------------------------------------


def diff_histograms(
    post: dict[tuple[str, tuple[tuple[str, str], ...]], Histogram],
    pre: dict[tuple[str, tuple[tuple[str, str], ...]], Histogram],
) -> dict[tuple[str, tuple[tuple[str, str], ...]], Histogram]:
    """Return new Histograms = post − pre.

    Counter/Histogram metrics are monotonically increasing in a single
    container lifetime, so subtracting is sound. Any bucket present in
    `post` but not `pre` is taken as-is (fresh container since the
    snapshot). If a key is in `pre` only, it's omitted — nothing new.
    """
    out: dict[tuple[str, tuple[tuple[str, str], ...]], Histogram] = {}
    for key, post_h in post.items():
        pre_h = pre.get(key)
        if pre_h is None:
            out[key] = post_h
            continue
        # Align buckets by le; counts subtract cumulatively.
        pre_by_le = dict(pre_h.buckets)
        diff_buckets: list[tuple[float, float]] = []
        for le, cnt in post_h.buckets:
            diff_buckets.append((le, max(0.0, cnt - pre_by_le.get(le, 0.0))))
        out[key] = Histogram(
            name=post_h.name,
            label_key=post_h.label_key,
            buckets=diff_buckets,
            count=max(0.0, post_h.count - pre_h.count),
            sum=max(0.0, post_h.sum - pre_h.sum),
        )
    return out


# ---------------------------------------------------------------------------
# Convenience lookup
# ---------------------------------------------------------------------------


def find_histogram(
    hists: dict[tuple[str, tuple[tuple[str, str], ...]], Histogram],
    name: str,
    match_labels: dict[str, str] | None = None,
) -> Histogram | None:
    """Find a histogram by base name + optional label match (all specified
    labels must equal)."""
    match_labels = match_labels or {}
    for (base, lbl_key), h in hists.items():
        if base != name:
            continue
        d = dict(lbl_key)
        if all(d.get(k) == v for k, v in match_labels.items()):
            return h
    return None


__all__ = [
    "Sample",
    "Histogram",
    "parse_prom_text",
    "collect_histograms",
    "diff_histograms",
    "find_histogram",
]
