"""Plot one example simulated light curve of each class (and subtype).

Usage (from the repository folder):
    python ml/plot_examples.py                      # saves ml/example_lightcurves.png
    python ml/plot_examples.py --out my_plot.png --seed 3

Classes (see simulate.py): 0 = impostor (periodic or giant eruption),
1 = supernova (Ia-like, II-like, IIn-like), 2 = AGN.
The plotted brightness is the total flux, including the host galaxy's light.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                     # draw to a file, no screen needed
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate import simulate_object, BANDS   # noqa: E402

# Which examples to show: (class label, subtype, panel title)
EXAMPLES = [(0, "periodic", "Class 0: impostor, periodic outbursts"),
            (0, "giant", "Class 0: impostor, single giant eruption"),
            (1, "IIn-like", "Class 1: supernova (IIn-like)"),
            (1, "Ia-like", "Class 1: supernova (Ia-like)"),
            (2, "agn", "Class 2: AGN")]
COLORS = {0: "#2e8b57", 1: "#d62728", 2: "#7f3f00"}   # g, r, i


def find_example(label, subtype, seed, min_points=40):
    """Simulate objects of this class until one has the wanted subtype."""
    for s in range(seed, seed + 1000):
        o = simulate_object(np.random.default_rng(s), label)
        if o["subtype"] == subtype and len(o["t"]) >= min_points:
            return o
    raise RuntimeError(f"no {subtype} example found")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0, help="change to see different examples")
    p.add_argument("--out", default="ml/example_lightcurves.png")
    args = p.parse_args()

    fig, axes = plt.subplots(len(EXAMPLES), 1, figsize=(8, 2.3 * len(EXAMPLES)), sharex=True)
    for i, (ax, (label, subtype, title)) in enumerate(zip(axes, EXAMPLES)):
        o = find_example(label, subtype, args.seed + 100 * i)
        dist_mpc = 10 ** ((o["mu"] - 25) / 5)           # distance modulus -> Mpc
        for k, band in enumerate(BANDS):                # one color per band
            m = o["band"] == k
            ax.errorbar(o["t"][m], o["lup"][m] - o["mu"], yerr=o["lup_err"][m],
                        fmt="o", ms=3, lw=0.8, color=COLORS[k], label=band)
        ax.invert_yaxis()                               # brighter = up
        extra = f", period {o['period']:.0f} d" if subtype == "periodic" else ""
        ax.set_title(f"{title}   (host at {dist_mpc:.0f} Mpc{extra})", fontsize=10, loc="left")
        ax.set_ylabel("abs. mag")
        ax.grid(alpha=0.25, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.98))
    axes[-1].set_xlabel("days since first observation")
    fig.suptitle("Simulated LSST-like light curves (total flux including host, asinh magnitude)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
