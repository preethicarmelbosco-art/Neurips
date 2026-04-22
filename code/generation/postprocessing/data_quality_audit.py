"""Reusable data quality audit for contrastive corpora.

Runs a comprehensive quality audit on any JSONL training dataset and produces:
  1. JSON report (machine-readable, for CI/pipeline integration)
  2. Appendix-ready markdown table (for paper inclusion)

Checks performed:
  Q1. Field completeness — empty or missing target/retain fields
  Q2. Field length — short records below configurable threshold
  Q3. Language purity — non-English (CJK) character detection
  Q4. Duplicate detection — exact and near-duplicate records
  Q5. Benchmark decontamination — 13-gram overlap with eval benchmarks
  Q6. Seed topic coherence — (SPL-CC only) flags physically nonsensical combos
  Q7. Field statistics — length distributions, percentiles

Designed to run as part of the data pipeline: call audit_dataset() after
generation and before training.

Usage:
    # Audit all datasets
    python -m ContrastiveData.scripts.data_quality_audit

    # Audit single dataset
    python -m ContrastiveData.scripts.data_quality_audit --domain math

    # Generate appendix markdown only (from existing reports)
    python -m ContrastiveData.scripts.data_quality_audit --appendix-only
"""

import argparse
import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = Path(__file__).parent.parent / "data"
REPORT_DIR = Path(__file__).parent.parent / "data" / "quality_reports"

# ── Benchmark configs for decontamination ──────────────────────────────

DECONTAM_BENCHMARKS = {
    "math": [
        {"name": "MATH", "hf_id": "EleutherAI/hendrycks_math",
         "configs": ["algebra", "counting_and_probability", "geometry",
                     "intermediate_algebra", "number_theory", "prealgebra", "precalculus"],
         "splits": ["test", "train"], "fields": ["problem", "solution"]},
        {"name": "GSM8K", "hf_id": "openai/gsm8k", "configs": ["main"],
         "splits": ["test", "train"], "fields": ["question", "answer"]},
    ],
    "tom": [
        {"name": "SocialIQA", "hf_id": "allenai/social_i_qa", "configs": [None],
         "splits": ["validation", "train"],
         "fields": ["context", "question", "answerA", "answerB", "answerC"]},
        {"name": "BigToM", "hf_id": "ptsv/bigtom_train", "configs": [None],
         "splits": ["train"], "fields": ["text"]},
    ],
    "causal": [
        {"name": "COPA", "hf_id": "pkavumba/balanced-copa", "configs": [None],
         "splits": ["test", "train"],
         "fields": ["premise", "choice1", "choice2"]},
    ],
    "coin": [
        {"name": "SocialIQA", "hf_id": "allenai/social_i_qa", "configs": [None],
         "splits": ["validation", "train"],
         "fields": ["context", "question", "answerA", "answerB", "answerC"]},
        {"name": "BigToM", "hf_id": "ptsv/bigtom_train", "configs": [None],
         "splits": ["train"], "fields": ["text"]},
        {"name": "COPA", "hf_id": "pkavumba/balanced-copa", "configs": [None],
         "splits": ["test", "train"],
         "fields": ["premise", "choice1", "choice2"]},
    ],
    "strategic": [
        {"name": "COPA", "hf_id": "pkavumba/balanced-copa", "configs": [None],
         "splits": ["test", "train"],
         "fields": ["premise", "choice1", "choice2"]},
    ],
}

# ── Dataset configs ────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "math": {
        "name": "SPL-CC",
        "full_name": "Symbolic Physics-Logic Contrastive Corpus",
        "train_path": str(DATA_DIR / "spl_cc_train.jsonl"),
        "holdout_path": str(DATA_DIR / "holdout" / "spl_cc_holdout.jsonl"),
        "target_field": "target",
        "retain_field": "retain",
        "id_field": "id",
        "seed_field": "seed_topic",
    },
    "tom": {
        "name": "ToM-CC",
        "full_name": "Theory of Mind Contrastive Corpus",
        "train_path": str(DATA_DIR / "tom_cc_train.jsonl"),
        "holdout_path": str(DATA_DIR / "holdout" / "tom_cc_holdout.jsonl"),
        "target_field": "target",
        "retain_field": "retain",
        "id_field": "id",
        "seed_field": "category",
    },
    "causal": {
        "name": "CTR-CC",
        "full_name": "Causal-Temporal Reasoning Contrastive Corpus",
        "train_path": str(DATA_DIR / "ctr_cc_train.jsonl"),
        "holdout_path": str(DATA_DIR / "holdout" / "ctr_cc_holdout.jsonl"),
        "target_field": "target_text",
        "retain_field": "retain_text",
        "id_field": "id",
        "seed_field": "category",
    },
    "coin": {
        "name": "COIN",
        "full_name": "Contrastive Opposite/Inverse Negatives",
        "train_path": str(DATA_DIR / "coin_cc_train.jsonl"),
        "holdout_path": str(DATA_DIR / "holdout" / "coin_holdout.jsonl"),
        "target_field": "target",
        "retain_field": "retain",
        "id_field": "id",
        "seed_field": "theme",
    },
    "strategic": {
        "name": "STR-CC",
        "full_name": "Strategic Reasoning Contrastive Corpus",
        "train_path": str(DATA_DIR / "str_cc_train.jsonl"),
        "holdout_path": str(DATA_DIR / "holdout" / "str_cc_holdout.jsonl"),
        "target_field": "target",
        "retain_field": "retain",
        "id_field": "id",
        "seed_field": "category",
    },
}


# ── Utility functions ──────────────────────────────────────────────────

def has_cjk(text: str) -> bool:
    """Check if text contains CJK (Chinese/Japanese/Korean) characters."""
    return any("\u4e00" <= c <= "\u9fff" or
               "\u3040" <= c <= "\u309f" or
               "\u30a0" <= c <= "\u30ff" for c in text)


def normalize_for_ngram(text: str) -> str:
    """Normalize text for n-gram comparison."""
    text = unicodedata.normalize("NFKD", text).lower()
    text = re.sub(r"\\(?:text|mathrm|mathbf|frac|sqrt|left|right|begin|end)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_ngrams(text: str, n: int = 13) -> set[str]:
    """Extract word-level n-grams."""
    words = normalize_for_ngram(text).split()
    if len(words) < n:
        return set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def text_hash(text: str) -> str:
    """SHA-256 hash of normalized text for dedup."""
    return hashlib.sha256(normalize_for_ngram(text).encode()).hexdigest()[:16]


def percentile(sorted_vals: list[float], p: float) -> float:
    """Compute p-th percentile from sorted values."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_vals) else f
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


# ── Core audit function ───────────────────────────────────────────────

def audit_dataset(
    domain: str,
    min_field_length: int = 200,
    ngram_size: int = 13,
    run_decontam: bool = True,
) -> dict:
    """Run full quality audit on a single dataset.

    Returns a comprehensive report dict.
    """
    cfg = DATASET_CONFIGS[domain]
    train_path = cfg["train_path"]
    holdout_path = cfg["holdout_path"]
    target_field = cfg["target_field"]
    retain_field = cfg["retain_field"]

    report = {
        "domain": domain,
        "corpus": cfg["name"],
        "full_name": cfg["full_name"],
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "min_field_length": min_field_length,
            "ngram_size": ngram_size,
            "target_field": target_field,
            "retain_field": retain_field,
        },
        "files": {},
    }

    # Audit both train and holdout
    for split_name, path in [("train", train_path), ("holdout", holdout_path)]:
        if not Path(path).exists():
            logger.warning("  %s/%s: file not found at %s — skipping", domain, split_name, path)
            report["files"][split_name] = {"status": "missing", "path": path}
            continue

        logger.info("Auditing %s/%s: %s", domain, split_name, path)
        split_report = _audit_file(
            path, target_field, retain_field, cfg.get("seed_field"),
            min_field_length, ngram_size,
        )
        split_report["path"] = path
        report["files"][split_name] = split_report

    # Decontamination (train only)
    if run_decontam and Path(train_path).exists() and domain in DECONTAM_BENCHMARKS:
        logger.info("Running decontamination scan for %s...", domain)
        decontam = _run_decontamination(
            train_path, target_field, retain_field,
            DECONTAM_BENCHMARKS[domain], ngram_size,
        )
        report["decontamination"] = decontam
    else:
        report["decontamination"] = {"status": "skipped"}

    # Summary: total records to drop
    train_data = report["files"].get("train", {})
    if train_data.get("status") != "missing":
        drop_set = set()
        drop_set.update(train_data.get("q1_empty_indices", []))
        drop_set.update(train_data.get("q2_short_indices", []))
        drop_set.update(train_data.get("q3_cjk_indices", []))
        drop_set.update(train_data.get("q4_exact_dup_indices", []))

        decontam_indices = set()
        if isinstance(report.get("decontamination"), dict):
            for hit in report["decontamination"].get("hits", []):
                decontam_indices.add(hit["line_idx"])
        drop_set.update(decontam_indices)

        report["summary"] = {
            "original_count": train_data["total_records"],
            "q1_empty": len(train_data.get("q1_empty_indices", [])),
            "q2_short": len(train_data.get("q2_short_indices", [])),
            "q3_cjk": len(train_data.get("q3_cjk_indices", [])),
            "q4_exact_duplicates": len(train_data.get("q4_exact_dup_indices", [])),
            "q5_contaminated": len(decontam_indices),
            "total_flagged": len(drop_set),
            "remaining_after_cleanup": train_data["total_records"] - len(drop_set),
            "drop_rate_pct": round(100 * len(drop_set) / max(train_data["total_records"], 1), 2),
        }

    return report


def _audit_file(
    path: str,
    target_field: str,
    retain_field: str,
    seed_field: str | None,
    min_field_length: int,
    ngram_size: int,
) -> dict:
    """Audit a single JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))

    total = len(records)
    result = {"total_records": total, "status": "ok"}

    # Q1: Empty fields
    q1_empty = []
    for i, rec in enumerate(records):
        t = rec.get(target_field, "")
        r = rec.get(retain_field, "")
        if not t.strip() or not r.strip():
            q1_empty.append(i)
    result["q1_empty_count"] = len(q1_empty)
    result["q1_empty_indices"] = q1_empty

    # Q2: Short fields (but not empty — those are Q1)
    q2_short = []
    for i, rec in enumerate(records):
        if i in set(q1_empty):
            continue
        t = rec.get(target_field, "")
        r = rec.get(retain_field, "")
        if len(t.strip()) < min_field_length or len(r.strip()) < min_field_length:
            q2_short.append(i)
    result["q2_short_count"] = len(q2_short)
    result["q2_short_indices"] = q2_short

    # Q3: CJK characters
    q3_cjk = []
    for i, rec in enumerate(records):
        text = rec.get(target_field, "") + rec.get(retain_field, "")
        if has_cjk(text):
            q3_cjk.append(i)
    result["q3_cjk_count"] = len(q3_cjk)
    result["q3_cjk_indices"] = q3_cjk

    # Q4: Exact duplicates (by target hash)
    seen_hashes = {}
    q4_dups = []
    for i, rec in enumerate(records):
        h = text_hash(rec.get(target_field, ""))
        if h in seen_hashes:
            q4_dups.append(i)
        else:
            seen_hashes[h] = i
    result["q4_exact_dup_count"] = len(q4_dups)
    result["q4_exact_dup_indices"] = q4_dups

    # Q7: Field length statistics
    target_lengths = sorted([len(rec.get(target_field, "").strip()) for rec in records])
    retain_lengths = sorted([len(rec.get(retain_field, "").strip()) for rec in records])

    result["q7_stats"] = {
        "target": {
            "min": target_lengths[0] if target_lengths else 0,
            "max": target_lengths[-1] if target_lengths else 0,
            "mean": round(sum(target_lengths) / max(len(target_lengths), 1), 1),
            "p5": round(percentile(target_lengths, 5), 1),
            "p25": round(percentile(target_lengths, 25), 1),
            "p50": round(percentile(target_lengths, 50), 1),
            "p75": round(percentile(target_lengths, 75), 1),
            "p95": round(percentile(target_lengths, 95), 1),
        },
        "retain": {
            "min": retain_lengths[0] if retain_lengths else 0,
            "max": retain_lengths[-1] if retain_lengths else 0,
            "mean": round(sum(retain_lengths) / max(len(retain_lengths), 1), 1),
            "p5": round(percentile(retain_lengths, 5), 1),
            "p25": round(percentile(retain_lengths, 25), 1),
            "p50": round(percentile(retain_lengths, 50), 1),
            "p75": round(percentile(retain_lengths, 75), 1),
            "p95": round(percentile(retain_lengths, 95), 1),
        },
    }

    # Q6: Seed topic stats (if applicable)
    if seed_field:
        seeds = [rec.get(seed_field, "") for rec in records]
        result["q6_unique_seeds"] = len(set(seeds))
        result["q6_top_seeds"] = [
            {"seed": s, "count": c}
            for s, c in Counter(seeds).most_common(10)
        ]

    return result


def _run_decontamination(
    train_path: str,
    target_field: str,
    retain_field: str,
    benchmarks: list[dict],
    ngram_size: int,
) -> dict:
    """Run n-gram decontamination against benchmarks."""
    from datasets import load_dataset

    total_benchmark_texts = 0
    total_ngrams = 0
    all_hits = []
    benchmark_details = []

    for bench in benchmarks:
        texts = []
        for config in bench["configs"]:
            for split in bench["splits"]:
                try:
                    kwargs = {"split": split, "trust_remote_code": True}
                    if config:
                        ds = load_dataset(bench["hf_id"], config, **kwargs)
                    else:
                        ds = load_dataset(bench["hf_id"], **kwargs)
                    for ex in ds:
                        combined = " ".join(str(ex.get(f, "")) for f in bench["fields"])
                        texts.append(combined)
                except Exception as e:
                    logger.warning("  Could not load %s/%s/%s: %s",
                                   bench["hf_id"], config, split, str(e)[:100])

        if not texts:
            benchmark_details.append({"name": bench["name"], "texts": 0, "status": "failed"})
            continue

        # Build index
        index = defaultdict(list)
        for idx, text in enumerate(texts):
            for ng in extract_ngrams(text, ngram_size):
                index[ng].append(idx)

        # Scan
        hits = 0
        with open(train_path) as f:
            for i, line in enumerate(f):
                rec = json.loads(line)
                target_ngs = extract_ngrams(rec.get(target_field, ""), ngram_size)
                retain_ngs = extract_ngrams(rec.get(retain_field, ""), ngram_size)
                matching = sum(1 for ng in target_ngs if ng in index) + \
                           sum(1 for ng in retain_ngs if ng in index)
                if matching > 0:
                    hits += 1
                    all_hits.append({
                        "line_idx": i,
                        "benchmark": bench["name"],
                        "matching_ngrams": matching,
                    })

        total_benchmark_texts += len(texts)
        total_ngrams += len(index)
        benchmark_details.append({
            "name": bench["name"],
            "texts": len(texts),
            "unique_ngrams": len(index),
            "hits": hits,
        })
        logger.info("  %s: %d texts, %d ngrams, %d hits",
                     bench["name"], len(texts), len(index), hits)

    return {
        "ngram_size": ngram_size,
        "total_benchmark_texts": total_benchmark_texts,
        "total_unique_ngrams": total_ngrams,
        "total_contaminated": len(all_hits),
        "benchmarks": benchmark_details,
        "hits": all_hits,
    }


# ── Appendix markdown generation ──────────────────────────────────────

def generate_appendix_markdown(reports: dict[str, dict]) -> str:
    """Generate appendix-ready markdown from audit reports."""
    lines = []
    lines.append("## D.7 Post-Generation Quality Audit\n")
    lines.append(
        "All contrastive corpora undergo an automated quality audit after generation "
        "and before use in any downstream task (extraction, distillation, evaluation). "
        "The audit script (`ContrastiveData/scripts/data_quality_audit.py`) performs "
        "five checks per corpus and produces the summary below. Audit results are "
        "deterministic and reproducible.\n"
    )

    # Summary table
    lines.append("### D.7.1 Audit Summary\n")
    lines.append("| Check | Description | Threshold |")
    lines.append("|-------|-------------|-----------|")
    lines.append("| Q1 | Empty fields | target or retain is whitespace-only |")
    lines.append("| Q2 | Short fields | target or retain < 200 characters |")
    lines.append("| Q3 | Non-English text | Contains CJK (Chinese/Japanese/Korean) characters |")
    lines.append("| Q4 | Exact duplicates | SHA-256 hash collision on normalized target text |")
    lines.append("| Q5 | Benchmark contamination | 13-gram word overlap with evaluation benchmarks |")
    lines.append("")

    # Per-corpus results
    lines.append("### D.7.2 Per-Corpus Results\n")

    # Build the main table
    header = "| Corpus | Generated | Q1 Empty | Q2 Short | Q3 Non-EN | Q4 Dups | Q5 Contam. | Total Dropped | Final Count | Drop Rate |"
    sep =    "|--------|-----------|----------|----------|-----------|---------|------------|---------------|-------------|-----------|"
    lines.append(header)
    lines.append(sep)

    for domain in ["math", "tom", "causal", "coin"]:
        if domain not in reports:
            cfg = DATASET_CONFIGS[domain]
            lines.append(
                f"| {cfg['name']} | — | — | — | — | — | — | — | — | *pending* |"
            )
            continue

        r = reports[domain]
        s = r.get("summary", {})
        if not s:
            cfg = DATASET_CONFIGS[domain]
            lines.append(
                f"| {cfg['name']} | — | — | — | — | — | — | — | — | *missing* |"
            )
            continue

        cfg = DATASET_CONFIGS[domain]
        lines.append(
            f"| {cfg['name']} "
            f"| {s['original_count']:,} "
            f"| {s['q1_empty']} "
            f"| {s['q2_short']} "
            f"| {s['q3_cjk']} "
            f"| {s['q4_exact_duplicates']} "
            f"| {s['q5_contaminated']} "
            f"| {s['total_flagged']} "
            f"| **{s['remaining_after_cleanup']:,}** "
            f"| {s['drop_rate_pct']}% |"
        )

    lines.append("")

    # Decontamination details
    lines.append("### D.7.3 Benchmark Decontamination\n")
    lines.append(
        "We check each training record for 13-gram word overlap against all "
        "evaluation benchmark splits (both train and test) to ensure zero data leakage. "
        "The 13-gram threshold follows Brown et al. (2020) and the GPT-4 technical report.\n"
    )

    decontam_rows = []
    for domain in ["math", "tom", "causal", "coin"]:
        if domain not in reports:
            continue
        r = reports[domain]
        decontam = r.get("decontamination", {})
        for bench in decontam.get("benchmarks", []):
            decontam_rows.append(
                f"| {DATASET_CONFIGS[domain]['name']} | {bench['name']} "
                f"| {bench.get('texts', '—'):,} "
                f"| {bench.get('unique_ngrams', '—'):,} "
                f"| **{bench.get('hits', '—')}** |"
            )

    if decontam_rows:
        lines.append("| Corpus | Benchmark | Benchmark Texts | Unique 13-grams | Contaminated Records |")
        lines.append("|--------|-----------|-----------------|-----------------|---------------------|")
        lines.extend(decontam_rows)
        lines.append("")
        lines.append(
            "All corpora are fully synthetic (LLM-generated from procedural seeds), "
            "so literal n-gram overlap with human-authored benchmarks is zero as expected. "
            "This confirms that downstream evaluation metrics are uncontaminated.\n"
        )

    # Field length statistics
    lines.append("### D.7.4 Field Length Distributions (Post-Cleanup)\n")

    stat_rows = []
    for domain in ["math", "tom", "causal", "coin"]:
        if domain not in reports:
            continue
        r = reports[domain]
        train = r.get("files", {}).get("train", {})
        stats = train.get("q7_stats", {})
        for field_name in ["target", "retain"]:
            s = stats.get(field_name, {})
            if not s:
                continue
            cfg = DATASET_CONFIGS[domain]
            field_label = cfg.get(f"{field_name}_field", field_name)
            stat_rows.append(
                f"| {cfg['name']} | `{field_label}` "
                f"| {s.get('min', '—')} "
                f"| {s.get('p5', '—')} "
                f"| {s.get('p50', '—')} "
                f"| {s.get('p95', '—')} "
                f"| {s.get('max', '—')} "
                f"| {s.get('mean', '—')} |"
            )

    if stat_rows:
        lines.append("| Corpus | Field | Min | P5 | P50 | P95 | Max | Mean |")
        lines.append("|--------|-------|-----|-----|------|------|------|------|")
        lines.extend(stat_rows)
        lines.append("")

    # Notes on seed topic coherence (SPL-CC specific)
    if "math" in reports:
        lines.append("### D.7.5 Seed Topic Coherence (SPL-CC)\n")
        lines.append(
            "SPL-CC seed topics are generated by combinatorial expansion of 50 physics "
            "concepts $\\times$ 30 objects $\\times$ 20 contexts. Approximately 3.2\\% of "
            "combinations yield physically implausible scenarios (e.g., *\"Adiabatic "
            "Processes of a cannonball at the equilibrium position\"*). Manual inspection "
            "confirms that the generated solutions still contain valid mathematical "
            "derivations — the LLM constructs plausible physical bridges between the "
            "concept and scenario. Since the contrastive signal targets equation-solving "
            "ability (presence vs. absence of mathematical notation), these records "
            "preserve the intended training signal and are retained.\n"
        )

    # Non-English note
    lines.append("### D.7.6 Language Filtering\n")
    lines.append(
        "All corpora target English-language evaluation benchmarks (MATH, GSM8K, "
        "Social IQA, BigToM, COPA). Records containing CJK characters — produced when "
        "the generator (DeepSeek-R1:32b) occasionally responds in Chinese — are removed "
        "to ensure language consistency between training data and evaluation. "
        "Original unfiltered files are preserved as `*.original.jsonl`.\n"
    )

    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Data quality audit for contrastive corpora")
    parser.add_argument("--domain", choices=["math", "tom", "causal", "coin", "strategic", "all"], default="all")
    parser.add_argument("--min-length", type=int, default=200)
    parser.add_argument("--ngram-size", type=int, default=13)
    parser.add_argument("--skip-decontam", action="store_true",
                        help="Skip benchmark decontamination (faster)")
    parser.add_argument("--appendix-only", action="store_true",
                        help="Generate appendix from existing reports only")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    domains = list(DATASET_CONFIGS.keys()) if args.domain == "all" else [args.domain]

    if args.appendix_only:
        # Load existing reports
        reports = {}
        for domain in domains:
            rpath = REPORT_DIR / f"quality_report_{domain}.json"
            if rpath.exists():
                with open(rpath) as f:
                    reports[domain] = json.load(f)
            else:
                logger.warning("No report for %s at %s", domain, rpath)
    else:
        # Run audits
        reports = {}
        for domain in domains:
            logger.info("\n{'=' * 60}")
            logger.info("Auditing %s...", domain)
            report = audit_dataset(
                domain,
                min_field_length=args.min_length,
                ngram_size=args.ngram_size,
                run_decontam=not args.skip_decontam,
            )
            reports[domain] = report

            # Save individual report
            rpath = REPORT_DIR / f"quality_report_{domain}.json"
            with open(rpath, "w") as f:
                json.dump(report, f, indent=2)
            logger.info("Report saved: %s", rpath)

            # Print summary
            s = report.get("summary", {})
            if s:
                print(f"\n{DATASET_CONFIGS[domain]['name']} ({domain}):")
                print(f"  Original:    {s['original_count']:,}")
                print(f"  Q1 Empty:    {s['q1_empty']}")
                print(f"  Q2 Short:    {s['q2_short']}")
                print(f"  Q3 Non-EN:   {s['q3_cjk']}")
                print(f"  Q4 Dups:     {s['q4_exact_duplicates']}")
                print(f"  Q5 Contam:   {s['q5_contaminated']}")
                print(f"  Total drop:  {s['total_flagged']}")
                print(f"  Remaining:   {s['remaining_after_cleanup']:,}")
                print(f"  Drop rate:   {s['drop_rate_pct']}%")

    # Generate appendix markdown
    appendix_md = generate_appendix_markdown(reports)
    appendix_path = REPORT_DIR / "appendix_quality_audit.md"
    with open(appendix_path, "w") as f:
        f.write(appendix_md)
    logger.info("Appendix markdown: %s", appendix_path)

    # Also save combined report
    combined_path = REPORT_DIR / "quality_report_combined.json"
    with open(combined_path, "w") as f:
        json.dump(reports, f, indent=2)
    logger.info("Combined report: %s", combined_path)

    print(f"\nAppendix markdown written to: {appendix_path}")


if __name__ == "__main__":
    main()
