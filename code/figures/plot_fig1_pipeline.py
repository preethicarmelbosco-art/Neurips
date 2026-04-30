"""Figure 1 — CogBench pipeline overview (schematic).

Generation → validation → 6-axis cognitive fingerprint
(5 cognitive primitives + NULL-CC linguistic-register predictor).
Intended for §1 / §3 of the D&B paper.

Usage:
    python -m evaluation.scripts.plots.plot_fig1_pipeline
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import apply_style, savefig

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, xy, w, h, label, color="#E8EEF5", edge="#1F4F8E", fontsize=7.5):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=0.8, edgecolor=edge, facecolor=color,
    ))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, wrap=True)


def arrow(ax, src, dst, text=None):
    ax.add_patch(FancyArrowPatch(
        src, dst, arrowstyle="-|>", mutation_scale=10,
        linewidth=0.8, color="#333333",
    ))
    if text:
        mx, my = (src[0] + dst[0]) / 2, (src[1] + dst[1]) / 2
        ax.text(mx, my + 0.03, text, ha="center", va="bottom",
                fontsize=6.5, style="italic", color="#555555")


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis("off")

    # Row labels
    ax.text(-0.05, 2.4, "Generation",
            ha="right", va="center", fontsize=7.5, style="italic", color="#555555")
    ax.text(-0.05, 0.95, "Evaluation",
            ha="right", va="center", fontsize=7.5, style="italic", color="#555555")

    # Row 1: Generation — role labels only, no model names
    box(ax, (0.1, 2.0), 1.6, 0.8,
        "Generator", color="#FDE8D4", edge="#C57A3A")
    box(ax, (2.0, 2.0), 2.1, 0.8,
        "Bijective pairs\n$\\mathcal{D}_{target}$ / $\\mathcal{D}_{retain}$\n($\\sim$44k total)")
    box(ax, (4.4, 2.0), 2.1, 0.8,
        "Gate:\nregex \\textbf{or}\n3-judge consensus",
        color="#E4F1E4", edge="#2E7D32")
    box(ax, (6.8, 2.0), 1.7, 0.8,
        "Human audit\n3 annotators", color="#F3E5F5", edge="#6A1B9A")
    box(ax, (8.7, 2.0), 1.2, 0.8, "Released\ncorpus", color="#FFFDE7", edge="#B8860B")

    arrow(ax, (1.7, 2.4), (2.0, 2.4))
    arrow(ax, (4.1, 2.4), (4.4, 2.4))
    arrow(ax, (6.5, 2.4), (6.8, 2.4))
    arrow(ax, (8.5, 2.4), (8.7, 2.4))

    # Row 2: Evaluation — aggregate counts, no model names
    box(ax, (0.1, 0.5), 1.9, 0.9,
        "39 open-weight\nmodels\n(1--72B)", color="#E3F2FD", edge="#1565C0")
    box(ax, (2.3, 0.5), 3.7, 0.9,
        "$\\text{CA}(m,c) = \\text{acc}_{target}(m,c) - \\text{acc}_{retain}(m,c)$\n"
        "judged by 2-judge unanimous panel")
    box(ax, (6.3, 0.5), 3.6, 0.9,
        "6-axis cognitive fingerprints\n"
        "(ToM, Causal, Moral, Strat, Spat, Null)\n"
        "$\\rightarrow$ composition regression",
        color="#FFEBEE", edge="#B71C1C")

    arrow(ax, (2.0, 0.95), (2.3, 0.95))
    arrow(ax, (6.0, 0.95), (6.3, 0.95))
    arrow(ax, (4.95, 1.95), (4.95, 1.45), text="evaluate")
    arrow(ax, (9.3, 1.95), (9.3, 1.45), text="yields")

    savefig(fig, "fig1_pipeline")
    plt.close(fig)


if __name__ == "__main__":
    main()
