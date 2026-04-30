"""Kitchen-sink baseline: do cognitive primitives add over the union of
all existing held-out aggregates?

Reproduces the body section 5.3 / App J.4 result: against the union of
~11 existing aggregate benchmarks as predictors, the cognitive primitives
tie or lose on five of six fittable domains (positive lead only on Safety).

For each domain d (excluding self):
    LOO R^2_CV under a Ridge fit on the union of all other aggregates,
    compared to the cognitive primitive subset's LOO R^2_CV from
    composition_regression.py.

Inputs:
    results/cogbench/table5_master.csv
    results/domain/domain_master.csv
    results/composition/preliminary_regression.json (cog R^2_CV per domain)

Outputs:
    results/composition/kitchen_sink.csv

Usage:
    python kitchen_sink.py
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut

ROOT = Path(__file__).resolve().parents[2]
PRIM_CSV = ROOT / "results" / "cogbench" / "table5_master.csv"
DOMAIN_CSV = ROOT / "results" / "domain" / "domain_master.csv"
COG_JSON = ROOT / "results" / "composition" / "preliminary_regression.json"
DST = ROOT / "results" / "composition" / "kitchen_sink.csv"

DOMAINS = [
    ("cybermetric",  "Cybersecurity"),
    ("medqa",        "Medical"),
    ("legalbench",   "Legal"),
    ("truthfulqa",   "Safety"),
    ("finqa",        "Financial"),
    ("scienceqa",    "Scientific"),
]


def loo_r2(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[float, int]:
    keep = ~np.isnan(y)
    Xk, yk = X[keep], y[keep]
    if len(yk) < 5:
        return float("nan"), len(yk)
    col_mean = np.nanmean(Xk, axis=0)
    col_keep = ~np.isnan(col_mean)
    Xk = Xk[:, col_keep]
    if Xk.size == 0 or Xk.shape[1] == 0:
        return float("nan"), len(yk)
    col_mean = col_mean[col_keep]
    idx = np.where(np.isnan(Xk))
    Xk[idx] = np.take(col_mean, idx[1])
    preds = np.full_like(yk, np.nan, dtype=float)
    for tr, te in LeaveOneOut().split(Xk):
        m = Ridge(alpha=alpha).fit(Xk[tr], yk[tr])
        preds[te] = m.predict(Xk[te])
    ss_res = float(np.sum((yk - preds) ** 2))
    ss_tot = float(np.sum((yk - yk.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, len(yk)


def main() -> None:
    dom = pd.read_csv(DOMAIN_CSV)
    dom["model"] = dom["model"].str.lower().str.strip()
    dom = dom.set_index("model")
    aggregate_cols = [c for c, _ in DOMAINS] + \
                     ["bigtom", "balanced_copa", "ethics",
                      "bigbench_strategy", "stepgame"]
    aggregate_cols = [c for c in aggregate_cols if c in dom.columns]

    cog = json.loads(COG_JSON.read_text()) if COG_JSON.exists() else {}

    rows = []
    for bench, label in DOMAINS:
        if bench not in dom.columns:
            continue
        # Kitchen-sink predictors: every other aggregate available
        ks_cols = [c for c in aggregate_cols if c != bench]
        X = dom[ks_cols].to_numpy(float)
        y = dom[bench].to_numpy(float)
        ks_r2, n = loo_r2(X, y)

        cog_r2 = cog.get(label, {}).get("r2_cv")
        delta = (cog_r2 - ks_r2) if (cog_r2 is not None and not math.isnan(ks_r2)) else None

        rows.append({
            "domain": label,
            "n": n,
            "kitchen_sink_features": len(ks_cols),
            "kitchen_sink_r2_cv": f"{ks_r2:+.3f}" if not math.isnan(ks_r2) else "NA",
            "cognitive_r2_cv": f"{cog_r2:+.3f}" if cog_r2 is not None else "NA",
            "delta_cog_minus_ks": f"{delta:+.3f}" if delta is not None else "NA",
        })

    if rows:
        with DST.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {DST}")
        wins = sum(1 for r in rows if r["delta_cog_minus_ks"] not in ("NA",) and float(r["delta_cog_minus_ks"]) > 0)
        print(f"  cognitive wins on {wins}/{len(rows)} domains under kitchen-sink baseline")


if __name__ == "__main__":
    main()
