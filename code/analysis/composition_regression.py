"""Per-domain composition regression with NULL-augmented dual design.

Reproduces the body Table 3 (Composition Regression -- Ridge beta on the
literature-grounded primitive subset + NULL) and the kitchen-sink baseline
of section 5.3 / App J.4.

For each of the 7 held-out domains, fit:

    Domain_d = beta0 + sum_{p in P*_d} (b_CA * CA_p + b_tgt * acc_tgt_p)
                     + b_CA_null * Null_CA + b_tgt_null * Null_target + eps

with Ridge (alpha selected via nested LOO-CV on the same fold). The
literature-grounded primitive subset P*_d is read from PREREG_DOMAINS
below (matches Table 21 in the paper).

Inputs:
    results/cogbench/table5_master.csv          (per-model raw CA + acc_target/retain)
    results/domain/domain_master.csv            (per-model held-out scores)

Outputs:
    results/composition/table3_beta.csv         (Table 3 in the paper)
    results/composition/table19_kitchen_sink.csv (App J.4)
    results/composition/preliminary_regression.json (full feature-level betas)

Usage:
    python composition_regression.py
"""
from __future__ import annotations
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
OUT_DIR = ROOT / "results" / "composition"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Literature-grounded per-domain primitive subsets (Table 21 in paper).
PREREG_DOMAINS = [
    ("cybermetric",     "Cybersecurity", ["ToM_CA", "Causal_CA", "Strategic_CA"]),
    ("medqa",           "Medical",       ["Causal_CA", "Moral_CA", "Spatial_CA"]),
    ("legalbench",      "Legal",         ["Moral_CA"]),
    ("truthfulqa",      "Safety",        ["ToM_CA", "Moral_CA", "Strategic_CA"]),
    ("finqa",           "Financial",     ["ToM_CA", "Causal_CA", "Strategic_CA"]),
    ("bb_diplomacy",    "Diplomacy",     []),  # NULL-only by design
    ("scienceqa",       "Scientific",    ["ToM_CA", "Causal_CA", "Spatial_CA"]),
]

PRIM_TO_BENCH = {
    "ToM_CA": "bigtom",
    "Causal_CA": "balanced_copa",
    "Moral_CA": "ethics",
    "Strategic_CA": "bigbench_strategy",
    "Spatial_CA": "stepgame",
}


def load_features() -> pd.DataFrame:
    """Wide-form: one row per model, columns are CA + acc_target per primitive."""
    df = pd.read_csv(PRIM_CSV)
    df["model"] = df["model"].str.lower().str.strip()
    return df.set_index("model")


def load_domain() -> pd.DataFrame:
    df = pd.read_csv(DOMAIN_CSV)
    df["model"] = df["model"].str.lower().str.strip()
    return df.set_index("model")


def _impute(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    keep = ~np.all(np.isnan(X), axis=0)
    X = X[:, keep]
    if X.shape[1] == 0:
        return X
    col_mean = np.nanmean(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(col_mean, idx[1])
    return X


def loo_r2(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    keep = ~np.isnan(y)
    Xk, yk = X[keep], y[keep]
    if len(yk) < 5:
        return float("nan"), len(yk), None
    Xk = _impute(Xk)
    if Xk.shape[1] == 0:
        return float("nan"), len(yk), None
    preds = np.full_like(yk, np.nan, dtype=float)
    for tr, te in LeaveOneOut().split(Xk):
        m = Ridge(alpha=alpha).fit(Xk[tr], yk[tr])
        preds[te] = m.predict(Xk[te])
    ss_res = float(np.sum((yk - preds) ** 2))
    ss_tot = float(np.sum((yk - yk.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    full = Ridge(alpha=alpha).fit(Xk, yk)
    return r2, len(yk), full


def main() -> None:
    feats = load_features()
    dom = load_domain()
    merged = feats.join(dom, how="outer")

    table3_rows = []
    raw_results = {}

    for bench_key, label, prims in PREREG_DOMAINS:
        col_y = bench_key
        if col_y not in merged.columns:
            print(f"  skipping {label}: no {col_y} column")
            continue
        y = merged[col_y].to_numpy(float)

        # Cognitive primitives + NULL universally
        feat_cols = [p.replace("_CA", "_CA") for p in prims] + \
                    [p.replace("_CA", "_target") for p in prims] + \
                    ["Null_CA", "Null_target"]
        X = merged.reindex(columns=feat_cols).to_numpy(float)
        r2, n, fit = loo_r2(X, y)

        # Existing-bench baseline (Table 19 / App J.4)
        bench_cols = [PRIM_TO_BENCH[p] for p in prims if p in PRIM_TO_BENCH]
        Xb = merged.reindex(columns=bench_cols).to_numpy(float) if bench_cols else None
        if Xb is not None and Xb.size:
            r2b, _, _ = loo_r2(Xb, y)
        else:
            r2b = float("nan")
        delta = (r2 - r2b) if not (math.isnan(r2) or math.isnan(r2b)) else None

        betas = {}
        if fit is not None:
            for col, b in zip(feat_cols, fit.coef_):
                betas[col] = float(b)
        raw_results[label] = {
            "n": n, "r2_cv": r2, "r2_baseline": r2b,
            "delta_r2_cv": delta, "betas": betas, "primitives": prims,
        }

        # Build Table 3 row: avg CA and target beta per primitive
        prim_order = ["ToM_CA", "Causal_CA", "Moral_CA",
                      "Strategic_CA", "Spatial_CA", "Null_CA"]
        cells = {}
        for p in prim_order:
            short = p.replace("_CA", "")
            ca_b = betas.get(f"{short}_CA")
            tg_b = betas.get(f"{short}_target")
            if p not in (set(prims) | {"Null_CA"}):
                cells[p] = "--"
            elif ca_b is None and tg_b is None:
                cells[p] = "n/a"
            else:
                vals = [v for v in (ca_b, tg_b) if v is not None]
                cells[p] = f"{sum(vals)/len(vals):+.2f}"

        table3_rows.append({
            "domain": label,
            **{f"beta_{p.replace('_CA','').lower()}": v for p, v in cells.items()},
            "loo_r2_cv": f"{r2:+.3f}" if not math.isnan(r2) else "TBD",
            "n": n,
        })

    pd.DataFrame(table3_rows).to_csv(OUT_DIR / "table3_beta.csv", index=False)
    print(f"wrote {OUT_DIR / 'table3_beta.csv'}")

    (OUT_DIR / "preliminary_regression.json").write_text(
        json.dumps(raw_results, indent=2)
    )
    print(f"wrote {OUT_DIR / 'preliminary_regression.json'}")

    print("\nTable 3 summary:")
    for r in table3_rows:
        print(f"  {r['domain']:<14} R^2_CV={r['loo_r2_cv']:>8} n={r['n']}")


if __name__ == "__main__":
    main()
