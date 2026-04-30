"""Figure 6 — Discrimination taxonomy: acc_target vs CA with quadrants.

Visualises the 2D taxonomy defined in Appendix G.3: a single CA value
conflates entangled (high-capacity, ceiling-compressed) from floor
(low-capacity, fails both) — only the (acc_target, CA) plane separates them.

Reads: evaluation/results/cogbench/table5_master.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import apply_style, savefig, RESULTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PRIMITIVE = "ToM"  # Appendix G.3 analyses ToM; richest coverage today.

# Models to annotate — Appendix G.3 killer examples + each quadrant anchor.
ANNOTATE = {
    "mistral-7b-instruct-v0.2": ("Mistral-Instruct\n(entangled)", (10, -18)),
    "smollm-1.7b":              ("SmolLM\n(floor)",               (10,  10)),
    "gemma-2-27b":              ("Gemma-2-27B",                   (10, -14)),
    "llama-3-8b":               ("Llama-3-8B",                    (10,  10)),
    "llama-3.2-3b":             ("Llama-3.2-3B",                  (-55, 10)),
    "phi-3-mini":               ("Phi-3-mini",                    (10, -14)),
    "llama-3.2-1b":             ("Llama-3.2-1B",                  (10,  10)),
    "llama-3.1-8b-instruct":    ("Llama-3.1-Instruct",            (-85, 10)),
}

# Quadrant boundaries (match Appendix G.3 regime ranges).
CA_THRESH = 0.10       # below → entangled/floor; above → discriminates
TARGET_MID = 0.50      # below → low-capability; above → high-capability


def load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "cogbench" / "table5_master.csv")
    df["model"] = df["model"].str.lower()
    tgt, ret = f"{PRIMITIVE}_target", f"{PRIMITIVE}_retain"
    d = df.dropna(subset=[tgt, ret]).copy()
    d["acc_target"] = d[tgt]
    d["acc_retain"] = d[ret]
    d["CA"] = d[tgt] - d[ret]
    return d


def main():
    apply_style()
    d = load()
    if len(d) < 4:
        print(f"[WARN] Only {len(d)} models with {PRIMITIVE} target+retain; figure will be sparse.")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    # Quadrant shading (bottom-right corner is the ceiling-compression trap).
    xmin, xmax = 0.0, 1.0
    ymin, ymax = -0.05, 0.45
    ax.add_patch(Rectangle((TARGET_MID, CA_THRESH), xmax - TARGET_MID, ymax - CA_THRESH,
                           facecolor="#E4F1E4", edgecolor="none", alpha=0.55, zorder=0))
    ax.add_patch(Rectangle((TARGET_MID, ymin), xmax - TARGET_MID, CA_THRESH - ymin,
                           facecolor="#FDE8D4", edgecolor="none", alpha=0.55, zorder=0))
    ax.add_patch(Rectangle((xmin, ymin), TARGET_MID - xmin, CA_THRESH - ymin,
                           facecolor="#F3E5F5", edgecolor="none", alpha=0.45, zorder=0))
    ax.add_patch(Rectangle((xmin, CA_THRESH), TARGET_MID - xmin, ymax - CA_THRESH,
                           facecolor="#FFFDE7", edgecolor="none", alpha=0.55, zorder=0))

    # Quadrant labels
    label_kw = dict(fontsize=7.2, style="italic", color="#333333",
                    ha="center", va="center", alpha=0.85)
    ax.text(0.75, 0.38, "Ideal:\nhigh ability + discriminates", **label_kw)
    ax.text(0.75, -0.02, "Entangled\n(ceiling-compressed)", **label_kw)
    ax.text(0.25, -0.02, "Floor\n(fails both sides)", **label_kw)
    ax.text(0.25, 0.38, "Low ability,\ndiscriminates", **label_kw)

    # Boundary lines
    ax.axhline(CA_THRESH, color="#666666", linewidth=0.6, linestyle="--", zorder=1)
    ax.axvline(TARGET_MID, color="#666666", linewidth=0.6, linestyle="--", zorder=1)
    ax.axhline(0, color="#888888", linewidth=0.5, zorder=1)

    # Points, coloured by size (log param count)
    sizes = pd.to_numeric(d["size_b"], errors="coerce").fillna(1.0)
    sc = ax.scatter(d["acc_target"], d["CA"], c=np.log10(sizes + 1e-3),
                    cmap="viridis", s=60, edgecolor="white", linewidth=0.6,
                    zorder=3)

    # Annotate selected models
    for _, row in d.iterrows():
        m = row["model"]
        if m in ANNOTATE:
            label, (dx, dy) = ANNOTATE[m]
            ax.annotate(label, xy=(row["acc_target"], row["CA"]),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=6.8, ha="left" if dx > 0 else "right",
                        arrowprops=dict(arrowstyle="-", color="#666666",
                                        linewidth=0.5, shrinkA=0, shrinkB=3))

    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(r"$\log_{10}$(params, B)", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$\text{acc}_{target}$ (capability)")
    ax.set_ylabel(r"CA = $\text{acc}_{target} - \text{acc}_{retain}$ (specificity)")
    ax.set_title(f"Discrimination taxonomy on {PRIMITIVE}: same CA, different mechanisms")
    ax.grid(True, linewidth=0.3, alpha=0.4, zorder=1)

    fig.tight_layout()
    savefig(fig, "fig6_discrimination")
    plt.close(fig)


if __name__ == "__main__":
    main()
