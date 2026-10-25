"""Sweep the similarity threshold and show why a threshold alone is not enough.

Produces bench/results/threshold_curve.png and threshold.json, and prints the
project's central finding: the adversarial and paraphrase similarity
distributions OVERLAP, so no cut point separates them (DESIGN.md §3).

    python bench/threshold_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Runnable as `python bench/threshold_sweep.py` from a checkout, not only after
# `pip install -e .`. Two lines that remove a papercut for anyone cloning this.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

from app.cache.embed import encode           # noqa: E402
from app.cache.semantic import lexical_conflict  # noqa: E402
from tests.test_cache_correctness import ADVERSARIAL, PARAPHRASE  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def sims(pairs):
    return np.array([float(np.dot(encode(a), encode(b))) for a, b in pairs])


def main():
    ok = sims(PARAPHRASE)          # should hit
    bad = sims(ADVERSARIAL)        # must not hit
    guarded = np.array([lexical_conflict(a, b) is not None for a, b in ADVERSARIAL])

    grid = np.linspace(0.60, 0.999, 120)
    useful = [(ok >= t).mean() for t in grid]
    wrong_cosine = [(bad >= t).mean() for t in grid]
    # With the guard, an adversarial pair only counts as a wrong hit if it is
    # above threshold AND the lexical check failed to catch it.
    wrong_guarded = [((bad >= t) & ~guarded).mean() for t in grid]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(grid, useful, label="useful hit rate (paraphrases)", color="steelblue")
    ax.plot(grid, wrong_cosine, label="WRONG hits — cosine only", color="crimson")
    ax.plot(grid, wrong_guarded, label="WRONG hits — cosine + lexical guard",
            color="seagreen", ls="--", lw=2)
    ax.axvspan(bad.max(), ok.max(), alpha=0.12, color="crimson")
    ax.annotate("overlap: no threshold\nseparates the classes",
                xy=((bad.max() + ok.max()) / 2, 0.55), ha="center", fontsize=9)
    ax.set_xlabel("cosine similarity threshold")
    ax.set_ylabel("rate")
    ax.set_title("A threshold alone cannot make a semantic cache safe")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "threshold_curve.png", dpi=150)

    overlaps = bool(bad.max() > ok.min())
    result = {
        "paraphrase": {"min": float(ok.min()), "max": float(ok.max())},
        "adversarial": {"min": float(bad.min()), "max": float(bad.max())},
        "distributions_overlap": overlaps,
        "margin": float(ok.min() - bad.max()),
        "adversarial_caught_by_guard": int(guarded.sum()),
        "adversarial_total": len(ADVERSARIAL),
    }
    (OUT / "threshold.json").write_text(json.dumps(result, indent=2))

    print(f"paraphrase   min={ok.min():.4f}  max={ok.max():.4f}")
    print(f"adversarial  min={bad.min():.4f}  max={bad.max():.4f}")
    print(f"margin (paraphrase floor - adversarial ceiling) = {result['margin']:+.4f}")

    if overlaps:
        print("\nFINDING: the distributions OVERLAP.")
        print(f"  a WRONG pair scores {bad.max():.4f}")
        print(f"  a RIGHT pair scores {ok.min():.4f}")
        print("  -> no threshold admits real paraphrases and rejects the wrong pairs.")
        print(f"  -> the lexical guard catches {guarded.sum()}/{len(ADVERSARIAL)} "
              f"adversarial pairs, including every one above threshold.")
    else:
        print("\nThe distributions separate cleanly; a threshold alone would suffice. "
              "Re-check whether the guard is still earning its place.")


if __name__ == "__main__":
    main()
