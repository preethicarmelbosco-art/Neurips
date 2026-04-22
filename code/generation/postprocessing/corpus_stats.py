"""Corpus-level statistics for the CogBench datasheet and paper.

Produces a single JSON report plus a Markdown summary covering, per corpus:
  - record count (bench = train+holdout)
  - category / difficulty cross-tab
  - target/retain length distribution (chars + whitespace tokens)
  - mean pair-distance (|target| - |retain|) in characters
  - optional embedding similarity summary if pre-computed JSON exists

Usage:
    python -m ContrastiveData.scripts.corpus_stats
"""

import json
import statistics
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
BENCH_DIR = DATA_DIR / "bench"
STATS_DIR = DATA_DIR / "stats"

CORPORA = {
    "spl_cc": {
        "name": "SPL-CC",
        "primitive": "Scientific / Applied Math",
        "file": "spl_cc_bench.jsonl",
        "target": "target",
        "retain": "retain",
        "category": "seed_topic",
        "difficulty": None,
        "gate": "regex",
    },
    "ctr_cc": {
        "name": "CTR-CC",
        "primitive": "Causal",
        "file": "ctr_cc_bench.jsonl",
        "target": "target_text",
        "retain": "retain_text",
        "category": "category",
        "difficulty": "difficulty",
        "gate": "regex",
    },
    "tom_cc": {
        "name": "ToM-CC",
        "primitive": "Theory of Mind",
        "file": "tom_cc_bench.jsonl",
        "target": "target",
        "retain": "retain",
        "category": "category",
        "difficulty": "difficulty",
        "gate": "judge",
    },
    "str_cc": {
        "name": "STR-CC",
        "primitive": "Strategic",
        "file": "str_cc_bench.jsonl",
        "target": "target",
        "retain": "retain",
        "category": "category",
        "difficulty": "difficulty",
        "gate": "judge",
    },
    "mor_cc": {
        "name": "MOR-CC",
        "primitive": "Moral",
        "file": "mor_cc_bench.jsonl",
        "target": "target",
        "retain": "retain",
        "category": "category",
        "difficulty": "difficulty",
        "gate": "judge",
    },
    "stp_cc": {
        "name": "STP-CC",
        "primitive": "Spatial",
        "file": "stp_cc_bench.jsonl",
        "target": "target",
        "retain": "retain",
        "category": "category",
        "difficulty": "difficulty",
        "gate": "regex",
    },
    "core_math": {
        "name": "CORE-MATH",
        "primitive": "Proof",
        "file": "core_math_bench.jsonl",
        "target": "target_proof",
        "retain": "retain_intuition",
        "category": "category",
        "difficulty": "difficulty",
        "gate": "regex",
    },
    "null_cc": {
        "name": "NULL-CC",
        "primitive": "Stylistic (null control)",
        "file": "null_cc_bench.jsonl",
        "target": "target_formal",
        "retain": "retain_informal",
        "category": "category",
        "difficulty": "complexity",
        "gate": "regex",
    },
}


def summarize(values):
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 1),
        "median": int(statistics.median(values)),
        "p10": int(statistics.quantiles(values, n=10)[0]) if len(values) >= 10 else min(values),
        "p90": int(statistics.quantiles(values, n=10)[8]) if len(values) >= 10 else max(values),
        "min": min(values),
        "max": max(values),
    }


def scan_corpus(cfg):
    path = BENCH_DIR / cfg["file"]
    if not path.exists():
        return {"error": f"missing file: {path}"}

    target_chars, retain_chars = [], []
    target_words, retain_words = [], []
    category_x_diff = {}

    n = 0
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            tgt = rec.get(cfg["target"], "") or ""
            ret = rec.get(cfg["retain"], "") or ""
            cat = rec.get(cfg["category"], "unknown")
            diff = rec.get(cfg["difficulty"], "all") if cfg["difficulty"] else "all"

            target_chars.append(len(tgt))
            retain_chars.append(len(ret))
            target_words.append(len(tgt.split()))
            retain_words.append(len(ret.split()))

            key = (cat, diff)
            category_x_diff[key] = category_x_diff.get(key, 0) + 1
            n += 1

    length_delta_chars = [t - r for t, r in zip(target_chars, retain_chars)]

    return {
        "n_pairs": n,
        "gate": cfg["gate"],
        "primitive": cfg["primitive"],
        "n_categories": len({k[0] for k in category_x_diff.keys()}),
        "n_strata": len(category_x_diff),
        "target_chars": summarize(target_chars),
        "retain_chars": summarize(retain_chars),
        "target_words": summarize(target_words),
        "retain_words": summarize(retain_words),
        "length_delta_chars": summarize(length_delta_chars),
        "category_counts": {
            cat: sum(v for (c, _), v in category_x_diff.items() if c == cat)
            for cat in sorted({k[0] for k in category_x_diff})
        },
        "difficulty_counts": {
            diff: sum(v for (_, d), v in category_x_diff.items() if d == diff)
            for diff in sorted({k[1] for k in category_x_diff})
        },
    }


def load_embed_similarity(corpus_key):
    path = DATA_DIR / f"{corpus_key}_embed_similarity.json"
    if not path.exists():
        path = DATA_DIR / "embed_similarity_summary.json"
        if not path.exists():
            return None
        try:
            data = json.load(open(path))
            return data.get(corpus_key, {})
        except Exception:
            return None
    try:
        return json.load(open(path))
    except Exception:
        return None


def main():
    report = {}
    for key, cfg in CORPORA.items():
        stats = scan_corpus(cfg)
        stats["name"] = cfg["name"]
        emb = load_embed_similarity(key)
        if emb is not None:
            stats["embed_similarity"] = emb
        report[key] = stats

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = STATS_DIR / "corpus_stats.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_json}")

    md_lines = [
        "# Corpus Statistics",
        "",
        "Computed from `bench/*_bench.jsonl` (train + holdout union).",
        "",
        "## Summary",
        "",
        "| Corpus | Primitive | Gate | Pairs | Cats | Target chars (p50) | Retain chars (p50) | Δchars (p50) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, s in report.items():
        md_lines.append(
            f"| {s['name']} | {s['primitive']} | {s['gate']} | {s['n_pairs']} | "
            f"{s['n_categories']} | {s['target_chars']['median']} | "
            f"{s['retain_chars']['median']} | {s['length_delta_chars']['median']} |"
        )

    md_lines += ["", "## Per-corpus detail", ""]
    for key, s in report.items():
        md_lines += [
            f"### {s['name']} ({s['primitive']})",
            "",
            f"- Pairs: {s['n_pairs']}; gate: {s['gate']}; strata: {s['n_strata']} "
            f"({s['n_categories']} categories × difficulties).",
            f"- Target length (chars): mean {s['target_chars']['mean']}, "
            f"p10 {s['target_chars']['p10']}, p50 {s['target_chars']['median']}, "
            f"p90 {s['target_chars']['p90']}.",
            f"- Retain length (chars): mean {s['retain_chars']['mean']}, "
            f"p10 {s['retain_chars']['p10']}, p50 {s['retain_chars']['median']}, "
            f"p90 {s['retain_chars']['p90']}.",
            f"- Δ length (target − retain, chars): mean {s['length_delta_chars']['mean']}, "
            f"p50 {s['length_delta_chars']['median']}.",
        ]
        emb = s.get("embed_similarity")
        if emb and isinstance(emb, dict):
            for k in ("mean", "median", "cosine_mean", "cosine_median"):
                if k in emb:
                    md_lines.append(f"- Embedding similarity ({k}): {emb[k]:.3f}")
        md_lines.append("")
        md_lines.append("**Categories:**")
        md_lines.append("")
        md_lines.append("| Category | Pairs |")
        md_lines.append("|---|---|")
        for cat, c in sorted(s["category_counts"].items(), key=lambda x: -x[1]):
            md_lines.append(f"| {cat} | {c} |")
        md_lines.append("")
        if list(s["difficulty_counts"].keys()) != ["all"]:
            md_lines.append("**Difficulty / complexity:**")
            md_lines.append("")
            md_lines.append("| Level | Pairs |")
            md_lines.append("|---|---|")
            for d, c in sorted(s["difficulty_counts"].items(), key=lambda x: -x[1]):
                md_lines.append(f"| {d} | {c} |")
            md_lines.append("")

    out_md = STATS_DIR / "corpus_stats.md"
    with open(out_md, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
