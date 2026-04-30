"""App E.2 NULL-CC orthogonality diagnostic.

For each cognitive primitive, train a probe at the same layer ell as the
NULL-CC probe; compute cosine similarity between the cognitive probe
direction w^cog_ell and the NULL probe direction w^null_ell.

Expected behaviour:
    - cosine(w_cog, w_null) ~ 0 across primitives at the layer where the
      cognitive probe is most accurate -- confirms that cognitive probes
      capture reasoning content, not stylistic register.
    - On small models (<= 3B), the angle may shrink (Linguistic Ceiling
      from section 4.3.3); on large models (>= 27B) it should approach
      0 as the model decouples style from logic.

Inputs:
    Pre-trained probe weight files from `train_probe.py` saved at
        results/probes/{model}/{corpus}_layer{ell}.npz

Outputs:
    results/probes/{model}/orthogonality_layer{ell}.csv with columns
        (cognitive_primitive, cosine_to_null, dot, w_cog_norm, w_null_norm)

Usage:
    python null_orthogonality_check.py --model <hf-id-or-name> --layer 16
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    args = ap.parse_args()

    probe_dir = ROOT / "results" / "probes" / Path(args.model).name
    null_path = probe_dir / f"null_cc_layer{args.layer:02d}.npz"
    if not null_path.exists():
        raise FileNotFoundError(f"Missing NULL-CC probe at layer {args.layer}: "
                                f"{null_path}. Run train_probe.py --corpus null_cc first.")

    null = np.load(null_path)
    w_null = null["w"]
    n_norm = float(np.linalg.norm(w_null))

    rows = []
    for corpus in ("tom_cc", "ctr_cc", "mor_cc", "str_cc", "stp_cc"):
        cog_path = probe_dir / f"{corpus}_layer{args.layer:02d}.npz"
        if not cog_path.exists():
            continue
        cog = np.load(cog_path)
        w_cog = cog["w"]
        c_norm = float(np.linalg.norm(w_cog))
        if c_norm == 0 or n_norm == 0:
            cos = float("nan")
        else:
            cos = float(np.dot(w_cog, w_null) / (c_norm * n_norm))
        rows.append({
            "cognitive_primitive": corpus.upper().replace("_CC", ""),
            "cosine_to_null": f"{cos:+.4f}",
            "dot": f"{float(np.dot(w_cog, w_null)):+.4f}",
            "w_cog_norm": f"{c_norm:.4f}",
            "w_null_norm": f"{n_norm:.4f}",
        })

    if not rows:
        print("No cognitive probes found. Run train_probe.py for each corpus first.")
        return

    dst = probe_dir / f"orthogonality_layer{args.layer:02d}.csv"
    with dst.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {dst}")
    for r in rows:
        print(f"  {r['cognitive_primitive']:<10}  cos = {r['cosine_to_null']}")


if __name__ == "__main__":
    main()
