"""Apply per-corpus judge-bias correction to raw CA values.

Reads `results/ceiling_compression/ca_all_rows.csv`; writes the per-model
wide table of raw and bias-corrected CA to
`results/cogbench/table5_master_corrected.csv`. The corrections come from
the calibration audit (App G Table 15, also in `calibration/calibration_summary.csv`):

    ToM:        delta = +0.140  ->  CA_corrected = CA_raw - 0.140
    Causal:     delta = +0.160  ->  CA_corrected = CA_raw - 0.160
    Moral:      delta = +0.040  ->  no correction (|delta| < 5 pp)
    Strategic:  delta = +0.220  ->  CA_corrected = CA_raw - 0.220
    Spatial:    auto-graded     ->  no correction
    Null:       delta = +0.700  ->  diagnostic; corrected reported alongside raw

Because the correction is a constant per-corpus shift, cross-model rankings
on every primitive are preserved exactly (Spearman rho = 1.000) and Ridge
regression slopes on CA are mathematically invariant to the shift.

Usage:
    python bias_correction.py
"""
from __future__ import annotations
import csv
from pathlib import Path

DELTA = {
    "ToM":       0.140,
    "Causal":    0.160,
    "Moral":     0.000,
    "Strategic": 0.220,
    "Spatial":   0.000,
    "Null":      0.700,
}

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "ceiling_compression" / "ca_all_rows.csv"
DST_LONG = ROOT / "results" / "cogbench" / "ca_corrected_long.csv"
DST_WIDE = ROOT / "results" / "cogbench" / "table5_master_corrected.csv"


def main() -> None:
    rows = list(csv.DictReader(SRC.open()))

    with DST_LONG.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "family", "size_b", "arch", "tier",
                    "corpus", "primitive", "ca_raw", "delta", "ca_corrected"])
        for r in rows:
            if not r["ca"].strip():
                continue
            prim = r["primitive"]
            if prim not in DELTA:
                continue
            raw = float(r["ca"])
            d = DELTA[prim]
            w.writerow([r["model"], r["family"], r["size_b"], r["arch"],
                        r["tier"], r["corpus"], prim,
                        f"{raw:.4f}", f"{d:.3f}", f"{raw - d:.4f}"])
    print(f"wrote {DST_LONG}")

    data: dict[str, dict[str, float | str]] = {}
    for r in rows:
        if not r["ca"].strip():
            continue
        p = r["primitive"]
        if p not in DELTA:
            continue
        m = r["model"]
        data.setdefault(m, {"family": r["family"], "size_b": r["size_b"]})
        raw = float(r["ca"])
        data[m][f"{p}_raw"] = raw
        data[m][f"{p}_corrected"] = raw - DELTA[p]

    prims = ["ToM", "Causal", "Moral", "Strategic", "Spatial", "Null"]
    headers = ["model", "family", "size_b"] + \
              [f"{p}_raw" for p in prims] + \
              [f"{p}_corrected" for p in prims]
    with DST_WIDE.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, restval="")
        w.writeheader()
        for m in sorted(data):
            row = {"model": m, "family": data[m]["family"], "size_b": data[m]["size_b"]}
            for p in prims:
                for sfx in ("raw", "corrected"):
                    k = f"{p}_{sfx}"
                    if k in data[m]:
                        row[k] = f"{data[m][k]:.4f}"
            w.writerow(row)
    print(f"wrote {DST_WIDE}")


if __name__ == "__main__":
    main()
