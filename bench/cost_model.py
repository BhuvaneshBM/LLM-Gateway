"""Cost per 1,000 requests as a function of cache hit rate, with the break-even
point marked. This is the plot that answers 'why does this service exist'."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

from app.config import PRICES     # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

MODEL = "llama-3.3-70b-versatile"
PROMPT_TOKENS, COMPLETION_TOKENS = 800, 250
MONTHLY_REQUESTS = 2_000_000

# Gateway infrastructure: 1 small VM + managed Postgres + Redis, roughly.
GATEWAY_FIXED_USD_PER_MONTH = 25.0


def main():
    p_price, c_price = PRICES[MODEL]
    per_call = float(PROMPT_TOKENS * p_price + COMPLETION_TOKENS * c_price) / 1e6

    hit = np.linspace(0, 0.8, 81)
    per_1k_direct = np.full_like(hit, per_call * 1000)
    per_1k_gateway = per_call * 1000 * (1 - hit) + \
        GATEWAY_FIXED_USD_PER_MONTH / (MONTHLY_REQUESTS / 1000)

    break_even = hit[np.argmax(per_1k_gateway <= per_1k_direct)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hit * 100, per_1k_direct, ls=":", color="gray", label="direct to provider")
    ax.plot(hit * 100, per_1k_gateway, label="through the gateway")
    ax.axvline(break_even * 100, ls="--", color="k",
               label=f"break-even = {break_even*100:.1f}% hit rate")
    ax.set_xlabel("cache hit rate (%)")
    ax.set_ylabel("USD per 1,000 requests")
    ax.set_title(f"Cost vs cache hit rate — {MODEL}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cost_curve.png", dpi=150)

    for h in (0.0, 0.2, 0.4, 0.6):
        monthly = per_call * MONTHLY_REQUESTS * (1 - h) + GATEWAY_FIXED_USD_PER_MONTH
        print(f"hit rate {h:>4.0%}  ->  ${monthly:>9,.2f}/month at "
              f"{MONTHLY_REQUESTS:,} requests")


if __name__ == "__main__":
    main()
