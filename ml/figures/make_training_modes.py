"""Draw ml/figures/training_modes.png: the three training modes of the transfer-learning experiment.

Run from the repository folder:  python ml/figures/make_training_modes.py
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTED = "#1f2328", "#57606a"
ORACLE_FILL, ORACLE_EDGE = "#dbe8f6", "#2f6db5"
RANDOM_FILL, RANDOM_EDGE = "#ffffff", "#8c959f"
HATCH = "#9dbbe0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "hatch.linewidth": 0.8})

rows = [("From scratch", ("random", "trained"), "Everything is learned from the labeled examples."),
        ("Pretrained ORACLE, fine-tuned", ("oracle", "trained gently"), "ORACLE's knowledge is the starting point and is adjusted."),
        ("Pretrained ORACLE, encoder frozen", ("oracle", "locked"), "ORACLE's knowledge is kept fixed; only the head learns.")]

W, H = 12.5, 6.6
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

def box(x, y, w, h, start, status, title, sub):
    fill, edge = (ORACLE_FILL, ORACLE_EDGE) if start == "oracle" else (RANDOM_FILL, RANDOM_EDGE)
    locked = status == "locked"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fill, ec="none"))
    if locked:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="none", ec=HATCH, lw=0, hatch="////"))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="none", ec=edge, lw=1.6, ls=(0, (4, 2)) if locked else "-"))
    pad = dict(fc=fill, ec="none", pad=1.5)
    ax.text(x + w / 2, y + h * 0.74, title, ha="center", va="center", fontsize=11, weight="bold", color=INK, bbox=pad)
    ax.text(x + w / 2, y + h * 0.50, sub, ha="center", va="center", fontsize=8.5, color=MUTED, bbox=pad)
    s = "start: ORACLE" if start == "oracle" else "start: random"
    ax.text(x + w / 2, y + h * 0.18, f"{s}  ·  {status}", ha="center", va="center", fontsize=8.5,
            color=INK, weight="bold" if locked else "normal", bbox=pad)

def arrow(x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=12, lw=1.2, color=MUTED))

def lightcurve(x, y, w, h):
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 1, 22))
    m = 0.2 + 0.6 * np.exp(-0.5 * ((t - 0.35) / 0.06) ** 2) + 0.5 * np.exp(-0.5 * ((t - 0.75) / 0.05) ** 2)
    m += rng.normal(0, 0.03, t.size)
    ax.plot(x + t * w, y + m / 1.3 * h, "o", ms=2.6, color=INK)
    ax.plot([x, x + w], [y, y], color=MUTED, lw=0.8)

X_IN, X_ENC, W_ENC, X_SUM, X_HEAD, W_HEAD, X_OUT = 0.2, 2.35, 2.55, 5.5, 7.35, 2.45, 10.45
for xc, label in [(X_IN + 0.7, "Input"), (X_ENC + W_ENC / 2, "Encoder"), (X_SUM + 0.65, "Summary"),
                  (X_HEAD + W_HEAD / 2, "Head"), (X_OUT + 0.55, "Output")]:
    ax.text(xc, H - 0.25, label, ha="center", fontsize=10, color=MUTED, weight="bold")

y0, dy, h = 4.45, 1.9, 1.2
for i, (name, enc, note) in enumerate(rows):
    y = y0 - i * dy
    yc = y + h / 2
    ax.text(X_IN, y + h + 0.2, f"{i + 1}. {name}", fontsize=11, weight="bold", color=INK)
    ax.text(X_ENC + W_ENC + 0.6, y + h + 0.2, note, fontsize=9, color=MUTED, style="italic")
    lightcurve(X_IN + 0.05, y + 0.3, 1.35, 0.75)
    ax.text(X_IN + 0.72, y + 0.08, "light curve", ha="center", fontsize=8.5, color=MUTED)
    arrow(X_IN + 1.5, X_ENC - 0.08, yc)
    box(X_ENC, y, W_ENC, h, enc[0], enc[1], "Encoder", "reads the light curve")
    arrow(X_ENC + W_ENC + 0.06, X_SUM - 0.05, yc)
    ax.text(X_SUM + 0.65, yc + 0.13, "16 numbers", ha="center", fontsize=9, color=INK)
    ax.text(X_SUM + 0.65, yc - 0.15, "describing", ha="center", fontsize=8, color=MUTED)
    ax.text(X_SUM + 0.65, yc - 0.38, "the shape", ha="center", fontsize=8, color=MUTED)
    arrow(X_SUM + 1.35, X_HEAD - 0.08, yc)
    box(X_HEAD, y, W_HEAD, h, "random", "trained", "Head", "makes the decision")
    ax.text(X_HEAD + W_HEAD / 2, y - 0.2, "+ host distance (input)", ha="center", fontsize=8.5, color=MUTED)
    arrow(X_HEAD + W_HEAD + 0.06, X_OUT - 0.05, yc)
    for j, c in enumerate(("SN impostor", "supernova", "AGN")):
        ax.text(X_OUT + 0.05, yc + 0.3 - j * 0.3, c, fontsize=9.5, va="center", color=INK)

ly = 0.05
def legend_swatch(x, fill, edge, locked=False):
    ax.add_patch(FancyBboxPatch((x, ly), 0.4, 0.24, boxstyle="round,pad=0.01", fc=fill, ec="none"))
    if locked:
        ax.add_patch(FancyBboxPatch((x, ly), 0.4, 0.24, boxstyle="round,pad=0.01", fc="none", ec=HATCH, lw=0, hatch="////"))
    ax.add_patch(FancyBboxPatch((x, ly), 0.4, 0.24, boxstyle="round,pad=0.01", fc="none", ec=edge, lw=1.4,
                                ls=(0, (4, 2)) if locked else "-"))
legend_swatch(0.2, RANDOM_FILL, RANDOM_EDGE);           ax.text(0.7, ly + 0.12, "starts from random weights", va="center", fontsize=9)
legend_swatch(3.6, ORACLE_FILL, ORACLE_EDGE);           ax.text(4.1, ly + 0.12, "starts from ORACLE's pretrained weights", va="center", fontsize=9)
legend_swatch(8.0, ORACLE_FILL, ORACLE_EDGE, True);     ax.text(8.5, ly + 0.12, "locked: not changed during training", va="center", fontsize=9)

fig.savefig("ml/figures/training_modes.png", dpi=200, bbox_inches="tight", facecolor="white")
