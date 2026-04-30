"""Per-(family, primitive) ceiling-compression decomposition.

Decomposes each multi-scale (family, primitive) cell into target- and
retain-slope per decade of parameters. Reproduces the "13/34 true-compression
cells" claim in section 4.3.2 of the paper.

A cell is `true compression` iff:
    target_slope > 0  AND  retain_slope >= target_slope  AND  CA_slope <= 0
i.e. the larger model's accuracy goes up but the gap (CA) stays flat or
shrinks because retain rose at least as fast as target.

Inputs:
    results/ceiling_compression/ca_all_rows.csv

Outputs:
    results/ceiling_compression/family_primitive_modes.csv (one row per cell)
    Console summary: counts of true-compression / growth / pseudo-compression /
    mixed across the 34 multi-scale cells.

Usage:
    python ceiling_compression.py
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "ceiling_compression" / "ca_all_rows.csv"
DST = ROOT / "results" / "ceiling_compression" / "family_primitive_modes.csv"

EXCLUDED_PRIMITIVES = {"Math", "Proof"}


def slope(xs: np.ndarray, ys: np.ndarray) -> float:
    """OLS slope of y on x; returns nan if <2 distinct x values."""
    if len(xs) < 2 or len(set(xs)) < 2:
        return float("nan")
    A = np.vstack([xs, np.ones_like(xs)]).T
    m, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(m)


def classify(t_slope: float, r_slope: float, ca_slope: float) -> str:
    if any(np.isnan(s) for s in (t_slope, r_slope, ca_slope)):
        return "insufficient"
    if t_slope > 0 and r_slope >= t_slope and ca_slope <= 0:
        return "true_compression"
    if t_slope > 0 and ca_slope > 0:
        return "growth"
    if t_slope <= 0 and r_slope <= 0:
        return "pseudo_compression"
    return "mixed"


def main() -> None:
    df = pd.read_csv(SRC)
    df = df[~df["primitive"].isin(EXCLUDED_PRIMITIVES)].copy()
    df["log_params"] = pd.to_numeric(df["log_params"], errors="coerce")
    df = df.dropna(subset=["log_params", "ca", "acc_target", "acc_retain"])

    rows = []
    counts: defaultdict = defaultdict(int)
    for (fam, prim), g in df.groupby(["family", "primitive"]):
        if len(g) < 2 or g["log_params"].nunique() < 2:
            continue
        x = g["log_params"].to_numpy(float)
        t_s = slope(x, g["acc_target"].to_numpy(float))
        r_s = slope(x, g["acc_retain"].to_numpy(float))
        c_s = slope(x, g["ca"].to_numpy(float))
        mode = classify(t_s, r_s, c_s)
        counts[mode] += 1
        rows.append({
            "family": fam,
            "primitive": prim,
            "n_models": len(g),
            "n_distinct_sizes": g["log_params"].nunique(),
            "target_slope": t_s,
            "retain_slope": r_s,
            "ca_slope": c_s,
            "mode": mode,
        })

    out = pd.DataFrame(rows).sort_values(["primitive", "family"]).reset_index(drop=True)
    out.to_csv(DST, index=False, float_format="%.4f")

    total = len(out)
    print(f"Multi-scale (family, primitive) cells: {total}")
    for mode in ("true_compression", "growth", "pseudo_compression", "mixed", "insufficient"):
        n = counts.get(mode, 0)
        if n:
            print(f"  {mode:<22}: {n}/{total} ({100*n/total:.0f}%)")
    print(f"\nwrote {DST}")
    if "true_compression" in counts:
        print("\nTrue-compression cells by primitive:")
        tc = out[out["mode"] == "true_compression"]
        for prim, sub in tc.groupby("primitive"):
            fams = ", ".join(sorted(sub["family"]))
            print(f"  {prim:<10} ({len(sub)} cells): {fams}")


if __name__ == "__main__":
    main()
