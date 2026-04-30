"""Per-primitive eta^2(family) with bootstrap 95% CIs.

Reads `results/ceiling_compression/ca_all_rows.csv`; writes
`results/ceiling_compression/ca_eta2_family_ci.csv` with one row per
primitive containing the one-way and family|logN partial-eta^2 estimates
plus their bootstrap confidence intervals.

This script reproduces the inline table in section 4.3.2 of the paper.

Specifications:
  - one-way: eta^2 = SS_family / SS_total
  - controlling for log N: SS(family | logN) / SS_total (Type-II contrast)
  - B = 2000 bootstrap resamples per primitive
  - SPL-CC and CORE-MATH are excluded (sample corpora; not analysed)

Usage:
    python eta2_family_ci.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

EXCLUDED = {"Math", "Proof"}
B = 2000
SEED = 0

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "ceiling_compression" / "ca_all_rows.csv"
DST = ROOT / "results" / "ceiling_compression" / "ca_eta2_family_ci.csv"


def eta2_oneway(d: pd.DataFrame) -> float:
    grand = d["ca"].mean()
    ss_total = float(((d["ca"] - grand) ** 2).sum())
    if ss_total == 0:
        return float("nan")
    ss_between = 0.0
    for _, g in d.groupby("family"):
        ss_between += len(g) * (g["ca"].mean() - grand) ** 2
    return ss_between / ss_total


def eta2_family_given_logN(d: pd.DataFrame) -> float:
    if len(d) < 5 or d["family"].nunique() < 2:
        return float("nan")
    y = d["ca"].to_numpy(float)
    grand = y.mean()
    ss_total = float(((y - grand) ** 2).sum())
    if ss_total == 0:
        return float("nan")
    fam = pd.get_dummies(d["family"], drop_first=True).astype(float).to_numpy()
    logn = d["log_params"].to_numpy(float).reshape(-1, 1)
    one = np.ones((len(d), 1))

    def ss_res(X: np.ndarray) -> float:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        return float((r * r).sum())

    return (ss_res(np.hstack([one, logn])) - ss_res(np.hstack([one, logn, fam]))) / ss_total


def boot_ci(d: pd.DataFrame, fn, rng: np.random.Generator) -> tuple[float, float]:
    n = len(d)
    arr = d.reset_index(drop=True)
    vals = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        sub = arr.iloc[idx]
        if sub["family"].nunique() < 2:
            vals[i] = np.nan
            continue
        try:
            vals[i] = fn(sub)
        except Exception:
            vals[i] = np.nan
    vals = vals[~np.isnan(vals)]
    if len(vals) < 100:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    if not SRC.exists():
        print(f"missing: {SRC}", file=sys.stderr)
        return 1
    df = pd.read_csv(SRC)
    df = df[~df["primitive"].isin(EXCLUDED)].copy()
    rng = np.random.default_rng(SEED)
    rows = []
    for prim, d in df.groupby("primitive"):
        e1 = eta2_oneway(d)
        lo1, hi1 = boot_ci(d, eta2_oneway, rng)
        e2 = eta2_family_given_logN(d)
        lo2, hi2 = boot_ci(d, eta2_family_given_logN, rng)
        rows.append({
            "primitive": prim,
            "n_obs": len(d),
            "n_families": d["family"].nunique(),
            "eta2_oneway": e1,
            "eta2_oneway_ci_lo": lo1,
            "eta2_oneway_ci_hi": hi1,
            "eta2_given_logN": e2,
            "eta2_given_logN_ci_lo": lo2,
            "eta2_given_logN_ci_hi": hi2,
        })
    out = pd.DataFrame(rows).sort_values("primitive").reset_index(drop=True)
    out.to_csv(DST, index=False, float_format="%.4f")
    print(out.to_string(index=False))
    print(f"\nwrote {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
