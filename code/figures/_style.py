"""Shared NeurIPS-compliant matplotlib style for CogBench figures.

Usage:
    from _style import apply_style, FIGDIR, savefig
    apply_style()
    fig, ax = plt.subplots(...)
    ...
    savefig(fig, "fig2_radar")
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

# Bundle-relative paths: this file lives at code/figures/_style.py;
# results/ and figures/ live at the bundle root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGDIR = PROJECT_ROOT / "figures"
RESULTS = PROJECT_ROOT / "results"

# Seven primary primitives (matches Table 1 / table5_master.csv column order).
PRIMITIVES_7 = ["ToM", "Causal", "Moral", "Strategic", "Spatial", "Proof", "Null"]

# Colorblind-safe palette (Okabe-Ito), one per primitive.
PALETTE = {
    "ToM":       "#0072B2",
    "Causal":    "#D55E00",
    "Moral":     "#009E73",
    "Strategic": "#CC79A7",
    "Spatial":   "#E69F00",
    "Proof":     "#56B4E9",
    "Null":      "#999999",
}


def apply_style():
    """NeurIPS-compliant matplotlib defaults: Times font, 8pt, thin lines."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 10,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "pdf.fonttype": 42,   # embed Type 1 fonts (NeurIPS requirement)
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def savefig(fig, stem: str):
    """Save figure as both PDF (main submission) and PNG (preview)."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGDIR / f"{stem}.pdf"
    png = FIGDIR / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    print(f"Wrote {pdf} and {png}")
