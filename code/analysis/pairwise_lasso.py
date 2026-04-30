"""Exploratory pairwise interaction Lasso (App J.13).

Power-constrained probe over the C(6,2) = 15 pairwise interactions of the
6 primary predictors (5 cognitive primitives + NULL). At 39:22 ~= 1.8:1
obs:param ratio this is below the descriptive floor; we treat Lasso as a
feature-selection procedure (not inference) and ask whether any of the
7 a priori interaction pairs (Table 27 in the paper) appear among the
survivors.

Inputs:
    results/cogbench/table5_master.csv
    results/domain/domain_master.csv

Outputs:
    results/composition/pairwise_lasso.csv

Usage:
    python pairwise_lasso.py
"""
from __future__ import annotations
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PRIM_CSV = ROOT / "results" / "cogbench" / "table5_master.csv"
DOMAIN_CSV = ROOT / "results" / "domain" / "domain_master.csv"
DST = ROOT / "results" / "composition" / "pairwise_lasso.csv"

PRIMS = ["ToM", "Causal", "Moral", "Strategic", "Spatial", "Null"]

DOMAINS = [
    ("cybermetric", "Cybersecurity"),
    ("medqa",       "Medical"),
    ("legalbench",  "Legal"),
    ("truthfulqa",  "Safety"),
    ("finqa",       "Financial"),
    ("bb_diplomacy","Diplomacy"),
    ("scienceqa",   "Scientific"),
]

A_PRIORI_PAIRS = {
    "Cybersecurity": ("Causal", "Strategic"),
    "Medical":       ("Causal", "Moral"),
    "Legal":         ("Causal", "Moral"),
    "Safety":        ("ToM", "Moral"),
    "Diplomacy":     ("ToM", "Strategic"),
    "Scientific":    ("Spatial", "Causal"),
}


def main() -> None:
    feats = pd.read_csv(PRIM_CSV)
    feats["model"] = feats["model"].str.lower().str.strip()
    feats = feats.set_index("model")
    dom = pd.read_csv(DOMAIN_CSV)
    dom["model"] = dom["model"].str.lower().str.strip()
    dom = dom.set_index("model")
    merged = feats.join(dom, how="outer")

    # Build interaction features (CA-only product per pair)
    rows = []
    for bench, label in DOMAINS:
        if bench not in merged.columns:
            continue
        prim_cols = [f"{p}_CA" for p in PRIMS]
        if any(c not in merged.columns for c in prim_cols):
            continue
        df = merged[[bench] + prim_cols].dropna(subset=[bench]).copy()
        for c in prim_cols:
            df[c] = df[c].fillna(df[c].mean())
        if len(df) < 10:
            continue
        # Construct interactions
        pair_cols: list[tuple[str, str, str]] = []
        for a, b in combinations(PRIMS, 2):
            col = f"{a}_x_{b}"
            df[col] = df[f"{a}_CA"] * df[f"{b}_CA"]
            pair_cols.append((a, b, col))

        X = StandardScaler().fit_transform(df[[c for _, _, c in pair_cols]].to_numpy(float))
        y = df[bench].to_numpy(float)
        try:
            lasso = LassoCV(cv=5, random_state=0, max_iter=20000).fit(X, y)
        except Exception as e:
            print(f"  {label}: Lasso failed -- {e}")
            continue
        survivors = [(a, b, abs(c)) for (a, b, _), c in zip(pair_cols, lasso.coef_) if abs(c) > 1e-9]
        survivors.sort(key=lambda t: -t[2])
        ap = A_PRIORI_PAIRS.get(label)
        ap_str = f"{ap[0]} x {ap[1]}" if ap else "(none)"
        ap_in = "Yes" if ap and any(set([a, b]) == set(ap) for a, b, _ in survivors) else "No"

        rows.append({
            "domain": label,
            "lambda_cv": f"{lasso.alpha_:.4f}",
            "n_survivors": len(survivors),
            "top_survivors": "; ".join(f"{a} x {b} ({c:+.2f})" for a, b, c in survivors[:3]) or "(none)",
            "a_priori_pair": ap_str,
            "a_priori_survives": ap_in,
        })

    if rows:
        with DST.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {DST}")
        ap_yes = sum(1 for r in rows if r["a_priori_survives"] == "Yes")
        print(f"  a priori pair survives Lasso on {ap_yes}/{len(rows)} domains")


if __name__ == "__main__":
    main()
