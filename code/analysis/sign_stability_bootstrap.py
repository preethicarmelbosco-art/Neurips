"""Per-cell bootstrap sign-stability of a priori beta weights.

Reproduces Table 4 in the paper (per-cell sign-stability, B = 5000) and
the full P(beta>0) breakdown in App J.10 (`tab:sign-stability-full`).

For each (domain, a priori primitive) cell we resample model rows with
replacement (B = 5000), refit Ridge on each draw, and record the sign of
the cell's averaged beta (CA + target accuracy averaged together).

Verdict thresholds:
    hit          P(beta > 0) >= 0.95   (positive sign-stable)
    miss         P(beta > 0) <= 0.05   (negative sign-stable; refutes prereg)
    underpowered otherwise              (CI straddles zero; not refutation)

Inputs:
    results/cogbench/table5_master.csv
    results/domain/domain_master.csv

Outputs:
    results/composition/table4_sign_stability.csv

Usage:
    python sign_stability_bootstrap.py
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
PRIM_CSV = ROOT / "results" / "cogbench" / "table5_master.csv"
DOMAIN_CSV = ROOT / "results" / "domain" / "domain_master.csv"
DST = ROOT / "results" / "composition" / "table4_sign_stability.csv"
DST.parent.mkdir(parents=True, exist_ok=True)

B = 5000
SEED = 42
ALPHA = 1.0

PREREG = [
    ("Cybersecurity", "cybermetric", ["ToM", "Causal", "Strategic"]),
    ("Medical",       "medqa",       ["Causal", "Moral", "Spatial"]),
    ("Legal",         "legalbench",  ["Moral"]),
    ("Safety",        "truthfulqa",  ["ToM", "Moral", "Strategic"]),
    ("Financial",     "finqa",       ["ToM", "Causal", "Strategic"]),
    ("Scientific",    "scienceqa",   ["ToM", "Causal", "Spatial"]),
]


def main() -> None:
    feats = pd.read_csv(PRIM_CSV)
    feats["model"] = feats["model"].str.lower().str.strip()
    feats = feats.set_index("model")

    dom = pd.read_csv(DOMAIN_CSV)
    dom["model"] = dom["model"].str.lower().str.strip()
    dom = dom.set_index("model")

    merged = feats.join(dom, how="outer")
    rng = np.random.default_rng(SEED)

    rows = []
    for domain, bench, prims in PREREG:
        if bench not in merged.columns:
            continue
        feat_cols: list[str] = []
        for p in prims:
            feat_cols += [f"{p}_CA", f"{p}_target"]
        feat_cols += ["Null_CA", "Null_target"]

        df = merged[[bench] + feat_cols].dropna(subset=[bench]).copy()
        df = df.dropna(subset=feat_cols, how="all")
        # Mean-impute missing predictor cells column-wise
        for c in feat_cols:
            df[c] = df[c].fillna(df[c].mean())
        if len(df) < 10:
            continue
        X = df[feat_cols].to_numpy(float)
        y = df[bench].to_numpy(float)
        n = len(y)

        for prim in prims:
            ca_idx = feat_cols.index(f"{prim}_CA")
            tg_idx = feat_cols.index(f"{prim}_target")
            beta_avgs = np.empty(B)
            for i in range(B):
                idx = rng.integers(0, n, n)
                m = Ridge(alpha=ALPHA).fit(X[idx], y[idx])
                beta_avgs[i] = (m.coef_[ca_idx] + m.coef_[tg_idx]) / 2

            p_pos = float(np.mean(beta_avgs > 0))
            if p_pos >= 0.95:
                verdict = "hit"
            elif p_pos <= 0.05:
                verdict = "miss"
            else:
                verdict = "underpowered"
            rows.append({
                "domain": domain,
                "primitive": prim,
                "predicted_sign": "+",
                "P_beta_gt_0": round(p_pos, 3),
                "P_beta_lt_0": round(1 - p_pos, 3),
                "verdict": verdict,
            })

    with DST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_hit = sum(1 for r in rows if r["verdict"] == "hit")
    n_miss = sum(1 for r in rows if r["verdict"] == "miss")
    n_under = sum(1 for r in rows if r["verdict"] == "underpowered")
    print(f"wrote {DST}")
    print(f"  hits={n_hit}  misses={n_miss}  underpowered={n_under}  of {len(rows)} testable cells")


if __name__ == "__main__":
    main()
