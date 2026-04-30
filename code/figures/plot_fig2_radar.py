"""Figure 2 — Cognitive fingerprint radar charts.

Small-multiples: ~6 representative models. Each radar shows the
6-axis fingerprint (ToM, Causal, Moral, Strategic, Spatial, Null).
Demonstrates profile dissociation at constant scale (§4.3.2).

Reads: evaluation/results/cogbench/table5_master.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import apply_style, savefig, RESULTS, PRIMITIVES_7

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRIMITIVES_6 = [p for p in PRIMITIVES_7 if p != "Proof"]

# Models chosen to highlight dissociations (falls back to available rows).
REPRESENTATIVES = [
    "granite-3.0-8b",
    "qwen-2.5-14b",
    "qwen-2.5-7b",
    "mistral-7b-instruct",
    "falcon-mamba-7b",
    "deepseek-r1-distill-qwen-7b",
]


def load_master() -> pd.DataFrame:
    path = RESULTS / "cogbench" / "table5_master.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run cogbench_eval.py first.")
    # on_bad_lines='skip' tolerates rows with stray trailing commas
    # (a known pre-camera-ready CSV-emit issue tracked in the manifest).
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    df["model"] = df["model"].str.lower()
    return df


def pick_models(df: pd.DataFrame) -> list[str]:
    have = [m for m in REPRESENTATIVES if m in df["model"].values]
    if len(have) >= 4:
        return have
    # Fallback: top 6 by primitive-coverage count (most non-null primitives)
    df = df.copy()
    df["_cov"] = df[PRIMITIVES_6].notna().sum(axis=1)
    return df.sort_values(["_cov", "Avg"], ascending=[False, False]).head(6)["model"].tolist()


def radar(ax, values, label, color):
    """Draw a 6-axis radar over PRIMITIVES_6 = [ToM, Causal, Moral, Strategic, Spatial, Null]."""
    angles = np.linspace(0, 2 * np.pi, len(PRIMITIVES_6), endpoint=False).tolist()
    angs_loop = angles + [angles[0]]
    vals_loop = list(values) + [values[0]]
    ax.plot(angs_loop, vals_loop, color=color, linewidth=1.3)
    ax.fill(angs_loop, vals_loop, color=color, alpha=0.18)

    ax.set_xticks(angles)
    ax.set_xticklabels(PRIMITIVES_6, fontsize=6.5)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.set_yticklabels(["", "0.1", "0.2", "0.3", "0.4"], fontsize=6)
    ax.set_ylim(-0.05, 0.45)
    ax.set_title(label, fontsize=8, pad=6)
    ax.grid(linewidth=0.3, color="#888888", alpha=0.6)


def main():
    apply_style()
    df = load_master()
    models = pick_models(df)
    n = len(models)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7.0, 2.4 * rows),
                              subplot_kw={"projection": "polar"})
    axes = np.atleast_2d(axes).flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 3)))
    for i, (model, color) in enumerate(zip(models, colors)):
        row = df[df["model"] == model].iloc[0]
        values = [float(row.get(p, 0) or 0) for p in PRIMITIVES_6]
        arch = row.get("arch", "")
        size = row.get("size_b", "")
        label = f"{model}\n({size}B, {arch})"
        radar(axes[i], values, label, color)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Cognitive fingerprints: 6-axis basis (ToM, Causal, Moral, Strategic, Spatial, Null)",
        y=1.01,
    )
    fig.tight_layout()
    savefig(fig, "fig2_radar")
    plt.close(fig)


if __name__ == "__main__":
    main()
