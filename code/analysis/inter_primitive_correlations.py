"""Inter-primitive correlation matrix + Moral leave-one-out sensitivity.

Reproduces the inter-primitive correlation matrix (Table 22 in the paper)
and the Moral floor LOO sensitivity (App D.8) from
`results/ceiling_compression/ca_all_rows.csv`.

Outputs:
    results/cogbench/inter_primitive_correlations.csv
    results/cogbench/moral_loo_sensitivity.csv

The Moral cells in the body matrix are reportable only when the zoo-wide
Moral CA standard deviation exceeds 0.10 (committed reporting rule). At
n=39, sd(Moral) is currently 0.084, narrowly below threshold; this script
reports the underlying values so a future reader can verify.

Usage:
    python inter_primitive_correlations.py
"""
from __future__ import annotations
import csv
import statistics
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "ceiling_compression" / "ca_all_rows.csv"
DST_MATRIX = ROOT / "results" / "cogbench" / "inter_primitive_correlations.csv"
DST_LOO = ROOT / "results" / "cogbench" / "moral_loo_sensitivity.csv"

PRIMITIVES = ["ToM", "Causal", "Moral", "Strategic", "Spatial", "Null"]


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    return num / (dx2 * dy2) ** 0.5 if dx2 * dy2 > 0 else 0.0


def main() -> None:
    data: dict[str, dict[str, float]] = {}
    with SRC.open() as f:
        for r in csv.DictReader(f):
            if not r["ca"].strip():
                continue
            p = r["primitive"]
            if p in PRIMITIVES:
                data.setdefault(r["model"], {})[p] = float(r["ca"])

    # Pairwise correlation matrix
    with DST_MATRIX.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["primitive_a", "primitive_b", "n", "pearson_r"])
        for a, b in combinations(PRIMITIVES, 2):
            pairs = [(d[a], d[b]) for d in data.values() if a in d and b in d]
            if len(pairs) < 3:
                continue
            xs, ys = zip(*pairs)
            r = pearson(xs, ys)
            w.writerow([a, b, len(pairs), f"{r:+.4f}"])
    print(f"wrote {DST_MATRIX}")

    # Moral CA distribution + Moral x Strategic LOO
    moral = [d["Moral"] for d in data.values() if "Moral" in d]
    n_m = len(moral)
    sd_m = statistics.stdev(moral) if n_m >= 2 else float("nan")
    mean_m = statistics.mean(moral) if n_m else float("nan")
    print(f"\nMoral CA: n={n_m} mean={mean_m:+.4f} sd={sd_m:.4f}")
    print(f"Threshold (sd > 0.10): {'PASS' if sd_m > 0.10 else 'FAIL (cells withheld)'}")

    pairs = [(m, d["Moral"], d.get("Strategic"))
             for m, d in data.items() if "Moral" in d]
    pairs = [(m, a, b) for m, a, b in pairs if b is not None]
    n = len(pairs)
    xs = [p[1] for p in pairs]
    ys = [p[2] for p in pairs]
    r_full = pearson(xs, ys)

    with DST_LOO.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dropped_model", "n", "moral_strategic_r"])
        w.writerow(["(none)", n, f"{r_full:+.4f}"])
        for i, (m, _, _) in enumerate(pairs):
            xs_i = xs[:i] + xs[i + 1:]
            ys_i = ys[:i] + ys[i + 1:]
            r_i = pearson(xs_i, ys_i)
            w.writerow([m, n - 1, f"{r_i:+.4f}"])
    print(f"wrote {DST_LOO}")
    print(f"Moral x Strategic full r = {r_full:+.4f} (n={n})")


if __name__ == "__main__":
    main()
