"""Scrape /metrics and print the README tables.

    python bench/report.py                      # after a k6 run
    python bench/report.py --url http://api:8000

Prometheus histograms are CUMULATIVE bucket counters, not percentiles:

    gw_stage_latency_seconds_bucket{stage="embed",le="0.01"} 4823
    gw_stage_latency_seconds_bucket{stage="embed",le="0.025"} 5102

To get p95 you find the bucket containing the 95th-percentile rank and
interpolate linearly inside it -- the same thing PromQL's histogram_quantile
does, reimplemented here so the project needs no Prometheus server to report.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from collections import defaultdict

SAMPLE = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[^\s]+)$")
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    out: dict[str, list[tuple[dict[str, str], float]]] = defaultdict(list)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = SAMPLE.match(line)
        if not m:
            continue
        labels = dict(LABEL.findall(m.group("labels") or ""))
        try:
            value = float(m.group("value"))
        except ValueError:
            continue                      # NaN on an untouched counter
        out[m.group("name")].append((labels, value))
    return out


def quantile(buckets: list[tuple[float, float]], q: float) -> float | None:
    """`buckets` is [(le, cumulative_count)] sorted ascending, +Inf last.

    Returns None when the histogram is empty -- an honest "no data" beats a
    confident 0.0 that looks like a measurement.
    """
    if not buckets:
        return None
    total = buckets[-1][1]
    if total <= 0:
        return None

    rank = q * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if count >= rank:
            if le == float("inf"):
                return prev_le            # tail beyond the last finite bucket
            if count == prev_count:
                return le
            frac = (rank - prev_count) / (count - prev_count)
            return prev_le + (le - prev_le) * frac
        prev_le, prev_count = le, count
    return buckets[-1][0]


def histogram(samples, key: str, value: str) -> list[tuple[float, float]]:
    """Collect one label-slice of a histogram into sorted (le, count) pairs."""
    pairs = []
    for labels, v in samples:
        if labels.get(key) != value:
            continue
        le = labels.get("le")
        pairs.append((float("inf") if le == "+Inf" else float(le), v))
    return sorted(pairs)


def ms(x: float | None) -> str:
    return "—" if x is None else f"{x * 1000:.1f} ms"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/metrics")
    a = ap.parse_args()

    try:
        with urllib.request.urlopen(a.url, timeout=5) as r:
            text = r.read().decode()
    except Exception as e:                                    # noqa: BLE001
        print(f"could not read {a.url}: {e}", file=sys.stderr)
        print("is the stack up?  docker compose up -d", file=sys.stderr)
        return 1

    m = parse(text)

    # ---- stage breakdown --------------------------------------------------
    stage_samples = m.get("gw_stage_latency_seconds_bucket", [])
    stages = sorted({l.get("stage") for l, _ in stage_samples if l.get("stage")})
    print("### Where the latency goes\n")
    print("| Stage | p50 | p95 | p99 | count |")
    print("|---|---|---|---|---|")
    before_provider = 0.0
    for stage in stages:
        b = histogram(stage_samples, "stage", stage)
        n = b[-1][1] if b else 0
        p50, p95, p99 = (quantile(b, q) for q in (0.50, 0.95, 0.99))
        if stage != "provider" and p50:
            before_provider += p50
        print(f"| {stage} | {ms(p50)} | {ms(p95)} | {ms(p99)} | {n:,.0f} |")
    print(f"| **total before provider (p50)** | **{before_provider * 1000:.1f} ms** | | | |")

    # ---- end-to-end, split by cache outcome -------------------------------
    req = m.get("gw_request_latency_seconds_bucket", [])
    print("\n### End-to-end\n")
    print("| Path | p50 | p95 | p99 | count |")
    print("|---|---|---|---|---|")
    for cached, label in (("true", "cache hit"), ("false", "cache miss")):
        b = histogram(req, "cached", cached)
        n = b[-1][1] if b else 0
        print(f"| {label} | {ms(quantile(b, .50))} | {ms(quantile(b, .95))} "
              f"| {ms(quantile(b, .99))} | {n:,.0f} |")

    # ---- cache outcomes ---------------------------------------------------
    counts = {l.get("result"): v for l, v in m.get("gw_cache_total", [])}
    hit, miss = counts.get("hit", 0), counts.get("miss", 0)
    looked = hit + miss
    print("\n### Cache\n")
    print(f"- lookups: **{looked:,.0f}**  (hit {hit:,.0f} / miss {miss:,.0f})")
    print(f"- hit rate: **{(hit / looked * 100) if looked else 0:.1f}%**")
    for extra in ("skip", "coalesced"):
        if counts.get(extra):
            print(f"- {extra}: {counts[extra]:,.0f}")

    # ---- spend ------------------------------------------------------------
    cost = sum(v for _, v in m.get("gw_cost_usd_total", []))
    reqs = sum(v for _, v in m.get("gw_requests_total", []))
    print("\n### Cost\n")
    print(f"- total: **${cost:.4f}** over {reqs:,.0f} requests")
    if reqs:
        print(f"- per 1,000 requests: **${cost / reqs * 1000:.4f}**")
    print("\n> Derived from published provider pricing and measured token counts, "
          "not from a bill.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
