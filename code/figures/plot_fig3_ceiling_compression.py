"""Figure 3 — Ceiling compression: CA collapses as retain rises faster than target.

Two panels:
    (a) acc_target and acc_retain vs model size (log x), per primitive.
    (b) CA = target - retain vs model size.

Reads: evaluation/results/cogbench/table5_master.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import apply_style, savefig, RESULTS, PRIMITIVES_7, PALETTE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRIMITIVES_6 = [p for p in PRIMITIVES_7 if p != "Proof"]


def load_master() -> pd.DataFrame:
    path = RESULTS / "cogbench" / "table5_master.csv"
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    df["size_b"] = pd.to_numeric(df["size_b"], errors="coerce")
    return df.dropna(subset=["size_b"])


def main(primitive: str = "ToM"):
    apply_style()
    df = load_master()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    # Panel (a): target and retain accuracy
    ax = axes[0]
    tgt_col, ret_col = f"{primitive}_target", f"{primitive}_retain"
    d = df.dropna(subset=[tgt_col, ret_col]).sort_values("size_b")
    ax.scatter(d["size_b"], d[tgt_col], color=PALETTE.get(primitive, "#0072B2"),
               marker="o", s=18, label=f"acc$_{{target}}$", zorder=3)
    ax.scatter(d["size_b"], d[ret_col], color="#999999",
               marker="s", s=18, label=f"acc$_{{retain}}$", zorder=3)
    # Connect same model with line to show gap
    for _, row in d.iterrows():
        ax.plot([row["size_b"]] * 2, [row[tgt_col], row[ret_col]],
                color="#BBBBBB", linewidth=0.5, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (B, log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"(a) {primitive}: acc$_{{target}}$ rises, but acc$_{{retain}}$ rises faster")
    ax.grid(True, linewidth=0.3, alpha=0.6)
    ax.legend(loc="lower right", frameon=False)

    # Panel (b): CA vs size, 6 primary corpora (5 cognitive primitives + Null)
    ax = axes[1]
    for prim in PRIMITIVES_6:
        if prim not in df.columns:
            continue
        d = df.dropna(subset=[prim, "size_b"]).sort_values("size_b")
        if len(d) < 3:
            continue
        ax.scatter(d["size_b"], d[prim], color=PALETTE.get(prim, "#333333"),
                   s=12, alpha=0.75, label=prim)
    ax.set_xscale("log")
    ax.axhline(0, color="#888888", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Parameters (B, log scale)")
    ax.set_ylabel("Contrastive Accuracy (CA)")
    ax.set_title("(b) CA vs scale — no monotonic trend (ceiling compression)")
    ax.grid(True, linewidth=0.3, alpha=0.6)
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=6.5)

    fig.tight_layout()
    savefig(fig, "fig3_ceiling_compression")
    plt.close(fig)


if __name__ == "__main__":
    main()
