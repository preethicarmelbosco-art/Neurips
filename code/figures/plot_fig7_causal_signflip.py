"""Figure 7 — Causal sign-flip: per-model retain saturation gradient.

Single panel. Models sorted by Causal acc_target (low to high).
Three series: acc_target, acc_retain (CTR-CC), BCOPA-CE.
Visual: target rises monotonically; BCOPA-CE saturates; retain catches
target on the high end, collapsing CA.

Reads: evaluation/results/cogbench/table5_master.csv
       evaluation/results/heldout_chain/heldout_chain_*.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import apply_style, savefig, RESULTS, PALETTE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_master() -> pd.DataFrame:
    path = RESULTS / "cogbench" / "table5_master.csv"
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    df["size_b"] = pd.to_numeric(df["size_b"], errors="coerce")
    return df.dropna(subset=["size_b"])


def load_bcopa() -> dict:
    out = {}
    for fp in (RESULTS / "heldout_chain").glob("heldout_chain_*.json"):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        mk = d.get("model_key")
        bc = d.get("set_a", {}).get("balanced_copa")
        if isinstance(bc, dict):
            v = bc.get("overall_accuracy") or bc.get("accuracy")
            if isinstance(v, (int, float)) and v == v:
                out[mk] = v
    return out


def main():
    apply_style()
    df = load_master()
    bcopa = load_bcopa()

    d = df.dropna(subset=["Causal_target", "Causal_retain"]).copy()
    d["BCOPA_CE"] = d["model"].map(bcopa)
    d = d.dropna(subset=["BCOPA_CE"]).sort_values("Causal_target").reset_index(drop=True)

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 2.6))
    x = np.arange(len(d))
    ax.plot(x, d["Causal_target"], color=PALETTE["Causal"],
            marker="o", markersize=3, linewidth=1.0, label=r"CTR-CC acc$_{target}$")
    ax.plot(x, d["Causal_retain"], color="#888888",
            marker="s", markersize=3, linewidth=1.0, label=r"CTR-CC acc$_{retain}$")
    ax.plot(x, d["BCOPA_CE"], color="#56B4E9",
            marker="^", markersize=3, linewidth=1.0, label="BCOPA-CE")
    # Shade CA gap (target - retain)
    ax.fill_between(x, d["Causal_retain"], d["Causal_target"],
                    color=PALETTE["Causal"], alpha=0.10, label="CA = target − retain")

    ax.set_xticks([])
    ax.set_xlabel(rf"Models, sorted by CTR-CC acc$_{{target}}$ (n={len(d)})")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Causal sign-flip mechanism: retain catches target as scale rises, CA collapses while BCOPA-CE saturates")
    ax.grid(True, linewidth=0.3, alpha=0.6)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=6.5)

    fig.tight_layout()
    savefig(fig, "fig7_causal_signflip")
    plt.close(fig)


if __name__ == "__main__":
    main()
