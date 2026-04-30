"""Ceiling compression — systematic regression of CA on log(params), controlling for family.

Replaces the single-family Gemma-2 ToM (2B → 27B) anecdote with zoo-wide evidence:
pooled OLS, family-fixed-effect OLS, and mixed-effects with random slopes by family,
per primitive. Also produces a per-family CA trajectory plot.

Reads per-model cogbench JSONs at evaluation/results/cogbench/*.json and extracts
`results[corpus].contrastive_accuracy` with metadata (family, size_b, arch).

Outputs (under evaluation/results/ceiling_compression/):
  - ca_all_rows.csv              : flat (model, primitive, size_b, family, ca)
  - ca_regression_summary.csv    : per-primitive pooled/FE/mixed slopes + CIs
  - ca_regression_summary.tex    : LaTeX fragment for the appendix
  - fig_ceiling_by_family.pdf    : CA vs log(size_b) colored by family, per primitive

Usage:
    python -m evaluation.scripts.plots.ceiling_compression_regression
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import apply_style, savefig, RESULTS, PALETTE  # type: ignore

OUT_DIR = RESULTS / "ceiling_compression"

CORPUS_TO_PRIMITIVE = {
    "ctr_cc": "Causal",
    "mor_cc": "Moral",
    "str_cc": "Strategic",
    "tom_cc": "ToM",
    "spl_cc": "Math",
    "stp_cc": "Spatial",
    "null_cc": "Null",
}

# Primitives included in the regression and figures.
# spl_cc ("Math") is excluded: retain=0 by design (see insight_spl_cc_retain_zero.md),
#   so "CA" is degenerate and a CA-vs-scale regression conflates floor-rising-capability
#   with the ceiling-compression mechanism the analysis is designed to test.
# core_math ("Proof") is excluded: demoted from the primitive set
#   (see feedback_no_core_math_backfill.md, 2026-04-27).
EXCLUDED_PRIMITIVES = {"Math", "Proof"}


def load_ca_rows() -> pd.DataFrame:
    src = RESULTS / "cogbench"
    rows = []
    for f in sorted(src.glob("*.json")):
        if "percategory" in f.name:
            continue
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(j, dict) or "results" not in j:
            continue
        base = {
            "model": j.get("model"),
            "family": j.get("family"),
            "size_b": j.get("size_b"),
            "arch": j.get("arch"),
            "tier": j.get("tier"),
        }
        for corpus, r in (j.get("results") or {}).items():
            if not isinstance(r, dict) or r.get("status") != "ok":
                continue
            ca = r.get("contrastive_accuracy")
            if ca is None:
                continue
            primitive = r.get("primitive") or CORPUS_TO_PRIMITIVE.get(corpus)
            rows.append({
                **base,
                "corpus": corpus,
                "primitive": primitive,
                "ca": float(ca),
                "acc_target": r.get("acc_target"),
                "acc_retain": r.get("acc_retain"),
            })
    df = pd.DataFrame(rows)
    df["size_b"] = pd.to_numeric(df["size_b"], errors="coerce")
    df["log_params"] = np.log10(df["size_b"] * 1e9)
    df = df.dropna(subset=["size_b", "ca", "family", "primitive"])
    return df.reset_index(drop=True)


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, r2)."""
    if len(y) < 2 or np.allclose(x, x[0]):
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def _bootstrap_slope_ci(y: np.ndarray, x: np.ndarray, n_boot: int = 1000,
                       seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    if n < 3:
        return float("nan"), float("nan")
    slopes = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if np.allclose(xb, xb[0]):
            slopes[i] = np.nan
            continue
        slopes[i] = np.polyfit(xb, yb, 1)[0]
    slopes = slopes[~np.isnan(slopes)]
    if len(slopes) < 10:
        return float("nan"), float("nan")
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return float(lo), float(hi)


def _family_fe_slope(df_p: pd.DataFrame) -> tuple[float, float]:
    """OLS CA ~ log_params + family_dummies. Returns (slope_on_log_params, r2)."""
    # Keep only families with ≥2 scale points
    counts = df_p.groupby("family")["size_b"].nunique()
    keep = counts[counts >= 2].index
    d = df_p[df_p["family"].isin(keep)]
    if len(d) < 4 or d["family"].nunique() < 2:
        return float("nan"), float("nan")
    fam_dummies = pd.get_dummies(d["family"], drop_first=True).astype(float)
    X = np.column_stack([
        np.ones(len(d)),
        d["log_params"].to_numpy(dtype=float),
        fam_dummies.to_numpy(),
    ])
    y = d["ca"].to_numpy(dtype=float)
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(beta[1]), float(r2)


def _mixed_model_slope(df_p: pd.DataFrame) -> tuple[float, float, int]:
    """Random-slope mixed-effects: ca ~ log_params + (log_params | family).

    Requires statsmodels. Returns (fixed_slope, slope_se, n_families_used).
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return float("nan"), float("nan"), 0
    counts = df_p.groupby("family")["size_b"].nunique()
    keep = counts[counts >= 2].index
    d = df_p[df_p["family"].isin(keep)].copy()
    if d["family"].nunique() < 3 or len(d) < 6:
        return float("nan"), float("nan"), int(d["family"].nunique())
    try:
        md = smf.mixedlm("ca ~ log_params", d, groups=d["family"],
                         re_formula="~log_params")
        res = md.fit(method="lbfgs", reml=False, disp=False)
        slope = float(res.fe_params.get("log_params", np.nan))
        se = float(res.bse.get("log_params", np.nan))
        return slope, se, int(d["family"].nunique())
    except Exception:
        return float("nan"), float("nan"), int(d["family"].nunique())


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    df = df[~df["primitive"].isin(EXCLUDED_PRIMITIVES)]
    for prim, d in df.groupby("primitive"):
        x = d["log_params"].to_numpy(dtype=float)
        y = d["ca"].to_numpy(dtype=float)
        slope, intercept, r2 = _ols(y, x)
        ci_lo, ci_hi = _bootstrap_slope_ci(y, x)
        fe_slope, fe_r2 = _family_fe_slope(d)
        mm_slope, mm_se, n_fam = _mixed_model_slope(d)
        # families with ≥2 scale points
        counts = d.groupby("family")["size_b"].nunique()
        n_fam_multi = int((counts >= 2).sum())
        out.append({
            "primitive": prim,
            "n_obs": len(d),
            "n_models": d["model"].nunique(),
            "n_families": d["family"].nunique(),
            "n_families_multi_scale": n_fam_multi,
            "pooled_slope": slope,
            "pooled_intercept": intercept,
            "pooled_r2": r2,
            "pooled_slope_ci_lo": ci_lo,
            "pooled_slope_ci_hi": ci_hi,
            "family_fe_slope": fe_slope,
            "family_fe_r2": fe_r2,
            "mixed_slope": mm_slope,
            "mixed_slope_se": mm_se,
            "mixed_n_families": n_fam,
        })
    return pd.DataFrame(out).sort_values("primitive").reset_index(drop=True)


def _families_palette(families: list[str]) -> dict[str, str]:
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
              "#56B4E9", "#A65628", "#999999", "#F0E442", "#222222"]
    return {f: colors[i % len(colors)] for i, f in enumerate(sorted(families))}


def plot(df: pd.DataFrame, summary: pd.DataFrame, out_pdf: Path) -> None:
    apply_style()
    df = df[~df["primitive"].isin(EXCLUDED_PRIMITIVES)]
    primitives = [p for p in ["ToM", "Causal", "Moral", "Strategic",
                              "Spatial", "Null"] if p in df["primitive"].unique()]
    fams = sorted(df["family"].unique())
    fam_colors = _families_palette(fams)

    n = len(primitives)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.0 * nrows),
                              sharey=False)
    axes = np.atleast_1d(axes).flatten()

    for ax, prim in zip(axes, primitives):
        d = df[df["primitive"] == prim]
        for fam, dd in d.groupby("family"):
            dd = dd.sort_values("size_b")
            c = fam_colors[fam]
            # connect same-family points with a line
            if dd["size_b"].nunique() >= 2:
                ax.plot(dd["size_b"], dd["ca"], color=c, linewidth=0.9,
                        alpha=0.75, zorder=2)
            ax.scatter(dd["size_b"], dd["ca"], color=c, s=16, zorder=3,
                       label=fam)
        # pooled regression line
        row = summary[summary["primitive"] == prim]
        if not row.empty and not np.isnan(row["pooled_slope"].iloc[0]):
            s = float(row["pooled_slope"].iloc[0])
            b = float(row["pooled_intercept"].iloc[0])
            xs = np.linspace(d["log_params"].min(), d["log_params"].max(), 20)
            ys = s * xs + b
            # convert back to size_b for plotting on log-x
            ax.plot(10 ** xs / 1e9, ys, color="black", linewidth=0.9,
                    linestyle="--", alpha=0.7, zorder=4,
                    label=f"pooled β={s:+.3f}")
        ax.set_xscale("log")
        ax.axhline(0, color="#888888", linewidth=0.4, linestyle=":")
        ax.set_title(prim, pad=2)
        ax.set_xlabel("params (B)")
        ax.set_ylabel("CA")
        ax.grid(True, linewidth=0.3, alpha=0.6)

    for ax in axes[len(primitives):]:
        ax.axis("off")

    # single legend for family colors, outside last used axis
    handles = [plt.Line2D([0], [0], marker="o", color=fam_colors[f],
                           linestyle="-", markersize=4, label=f) for f in fams]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(fams)),
               frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("CA vs scale, by family — ceiling compression is not zoo-universal",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"Wrote {out_pdf} and {out_pdf.with_suffix('.png')}")


def write_tex(summary: pd.DataFrame, out_tex: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Primitive & $n$ & families ($\geq$2 scales) & "
        r"pooled $\beta_{\log N}$ & 95\% CI & "
        r"family-FE $\beta$ & mixed $\beta$ (SE) & $R^2_{\text{pool}}$ \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        def f(v, fmt="{:+.3f}"):
            return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)
        ci = "--" if np.isnan(r["pooled_slope_ci_lo"]) else (
            f"[{r['pooled_slope_ci_lo']:+.3f}, {r['pooled_slope_ci_hi']:+.3f}]")
        mixed = "--" if np.isnan(r["mixed_slope"]) else (
            f"{r['mixed_slope']:+.3f} ({r['mixed_slope_se']:.3f})")
        lines.append(
            f"{r['primitive']} & {int(r['n_obs'])} & {int(r['n_families_multi_scale'])} & "
            f"{f(r['pooled_slope'])} & {ci} & {f(r['family_fe_slope'])} & {mixed} & "
            f"{f(r['pooled_r2'], '{:.3f}')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_tex}")


def main():
    df = load_ca_rows()
    if df.empty:
        raise SystemExit("No CA rows found under evaluation/results/cogbench/*.json")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows_csv = OUT_DIR / "ca_all_rows.csv"
    df.to_csv(all_rows_csv, index=False)
    print(f"Wrote {all_rows_csv}  ({len(df)} rows)")

    summary = analyze(df)
    summary_csv = OUT_DIR / "ca_regression_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv}")
    print(summary.to_string(index=False))

    write_tex(summary, OUT_DIR / "ca_regression_summary.tex")
    plot(df, summary, OUT_DIR / "fig_ceiling_by_family.pdf")


if __name__ == "__main__":
    main()