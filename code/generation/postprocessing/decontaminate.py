"""Decontamination scanner for SPL-CC, ToM-CC, and CTR-CC training data.

Checks training data against evaluation benchmarks using n-gram overlap
(standard decontamination per Brown et al., 2020; GPT-4 tech report).

Strategy: Flag contaminated records and replace them with fresh generations
from the same seed topic (SPL) or scenario category (ToM/CTR).

Benchmarks checked:
  - SPL-CC (math):   MATH (hendrycks/competition_math), GSM8K (openai/gsm8k)
  - ToM-CC (tom):    Social IQA (allenai/social_i_qa), BigToM (ptsv/bigtom_train)
  - CTR-CC (causal): CLadder (causalNLP/cladder), COPA (super_glue copa)

Usage:
    python -m ContrastiveData.scripts.decontaminate --scan          # scan only
    python -m ContrastiveData.scripts.decontaminate --scan --drop   # scan + write cleaned files
    python -m ContrastiveData.scripts.decontaminate --scan --report # scan + save detailed report
    python -m ContrastiveData.scripts.decontaminate --scan --domain causal
"""

import argparse
import json
import logging
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = Path(__file__).parent.parent / "data"
REPORT_DIR = Path(__file__).parent.parent / "data" / "decontamination"

# --- Text normalization ---

def normalize_text(text: str) -> str:
    """Normalize text for n-gram comparison.

    Lowercase, collapse whitespace, strip punctuation, normalize unicode.
    Keeps digits and math operators as they're meaningful for MATH/GSM8K.
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    # Remove LaTeX commands but keep content
    text = re.sub(r"\\(?:text|mathrm|mathbf|frac|sqrt|left|right|begin|end)\b", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_ngrams(text: str, n: int = 13) -> set[str]:
    """Extract character-level n-grams from normalized text."""
    words = normalize_text(text).split()
    if len(words) < n:
        return set()
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


# --- Benchmark loaders ---

def load_math_benchmark() -> list[str]:
    """Load MATH benchmark problems (all splits)."""
    from datasets import load_dataset
    texts = []
    try:
        ds = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)
        for ex in ds:
            texts.append(ex.get("problem", "") + " " + ex.get("solution", ""))
    except Exception as e:
        logger.warning("Could not load hendrycks/competition_math: %s", e)

    try:
        ds = load_dataset("hendrycks/competition_math", split="train", trust_remote_code=True)
        for ex in ds:
            texts.append(ex.get("problem", "") + " " + ex.get("solution", ""))
    except Exception as e:
        logger.warning("Could not load MATH train split: %s", e)

    logger.info("  MATH benchmark: %d texts", len(texts))
    return texts


def load_gsm8k_benchmark() -> list[str]:
    """Load GSM8K benchmark problems."""
    from datasets import load_dataset
    texts = []
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
        for ex in ds:
            texts.append(ex.get("question", "") + " " + ex.get("answer", ""))
    except Exception as e:
        logger.warning("Could not load gsm8k test: %s", e)

    try:
        ds = load_dataset("openai/gsm8k", "main", split="train", trust_remote_code=True)
        for ex in ds:
            texts.append(ex.get("question", "") + " " + ex.get("answer", ""))
    except Exception as e:
        logger.warning("Could not load gsm8k train: %s", e)

    logger.info("  GSM8K benchmark: %d texts", len(texts))
    return texts


def load_social_iqa_benchmark() -> list[str]:
    """Load Social IQA benchmark."""
    from datasets import load_dataset
    texts = []
    try:
        ds = load_dataset("allenai/social_i_qa", split="validation", trust_remote_code=True)
        for ex in ds:
            context = ex.get("context", "")
            question = ex.get("question", "")
            answers = " ".join([ex.get(f"answer{c}", "") for c in "ABC"])
            texts.append(f"{context} {question} {answers}")
    except Exception as e:
        logger.warning("Could not load social_i_qa: %s", e)

    try:
        ds = load_dataset("allenai/social_i_qa", split="train", trust_remote_code=True)
        for ex in ds:
            context = ex.get("context", "")
            question = ex.get("question", "")
            answers = " ".join([ex.get(f"answer{c}", "") for c in "ABC"])
            texts.append(f"{context} {question} {answers}")
    except Exception as e:
        logger.warning("Could not load social_i_qa train: %s", e)

    logger.info("  Social IQA benchmark: %d texts", len(texts))
    return texts


def load_bigtom_benchmark() -> list[str]:
    """Load BigToM benchmark."""
    from datasets import load_dataset
    texts = []
    try:
        ds = load_dataset("ptsv/bigtom_train", split="train", trust_remote_code=True)
        for ex in ds:
            texts.append(ex.get("text", ""))
    except Exception as e:
        logger.warning("Could not load bigtom: %s", e)

    logger.info("  BigToM benchmark: %d texts", len(texts))
    return texts


def load_cladder_benchmark() -> list[str]:
    """Load CLadder benchmark (NeurIPS 2023) — causal reasoning over Pearl's hierarchy."""
    from datasets import load_dataset
    texts = []
    try:
        ds = load_dataset("causalNLP/cladder", split="test", trust_remote_code=True)
        for ex in ds:
            q = ex.get("question", "")
            a = ex.get("answer", "")
            story = ex.get("story", "")
            texts.append(f"{story} {q} {a}".strip())
    except Exception as e:
        logger.warning("Could not load causalNLP/cladder test: %s", e)

    try:
        ds = load_dataset("causalNLP/cladder", split="train", trust_remote_code=True)
        for ex in ds:
            q = ex.get("question", "")
            a = ex.get("answer", "")
            story = ex.get("story", "")
            texts.append(f"{story} {q} {a}".strip())
    except Exception as e:
        logger.warning("Could not load causalNLP/cladder train: %s", e)

    logger.info("  CLadder benchmark: %d texts", len(texts))
    return texts


def load_copa_benchmark() -> list[str]:
    """Load COPA benchmark (SuperGLUE) — causal commonsense reasoning."""
    from datasets import load_dataset
    texts = []
    try:
        ds = load_dataset("super_glue", "copa", split="validation", trust_remote_code=True)
        for ex in ds:
            premise = ex.get("premise", "")
            c1 = ex.get("choice1", "")
            c2 = ex.get("choice2", "")
            question = ex.get("question", "")
            texts.append(f"{premise} {question} {c1} {c2}".strip())
    except Exception as e:
        logger.warning("Could not load super_glue copa validation: %s", e)

    try:
        ds = load_dataset("super_glue", "copa", split="train", trust_remote_code=True)
        for ex in ds:
            premise = ex.get("premise", "")
            c1 = ex.get("choice1", "")
            c2 = ex.get("choice2", "")
            question = ex.get("question", "")
            texts.append(f"{premise} {question} {c1} {c2}".strip())
    except Exception as e:
        logger.warning("Could not load super_glue copa train: %s", e)

    logger.info("  COPA benchmark: %d texts", len(texts))
    return texts


# --- Build n-gram index ---

def build_benchmark_ngram_index(
    benchmark_texts: list[str],
    n: int = 13,
) -> dict[str, list[int]]:
    """Build inverted index: ngram -> list of benchmark indices that contain it."""
    index = defaultdict(list)
    for idx, text in enumerate(benchmark_texts):
        for ngram in extract_ngrams(text, n):
            index[ngram].append(idx)
    return index


# --- Scanning ---

def scan_training_data(
    train_path: str,
    target_field: str,
    retain_field: str,
    benchmark_ngram_index: dict[str, list[int]],
    benchmark_name: str,
    n: int = 13,
) -> list[dict]:
    """Scan training data for benchmark contamination.

    Returns list of contaminated records with details.
    """
    contaminated = []
    total = 0

    with open(train_path) as f:
        for line_idx, line in enumerate(f):
            total += 1
            rec = json.loads(line)
            target = rec.get(target_field, "")
            retain = rec.get(retain_field, "")

            # Check both target and retain fields
            target_ngrams = extract_ngrams(target, n)
            retain_ngrams = extract_ngrams(retain, n)

            target_hits = set()
            retain_hits = set()

            for ng in target_ngrams:
                if ng in benchmark_ngram_index:
                    target_hits.add(ng)
            for ng in retain_ngrams:
                if ng in benchmark_ngram_index:
                    retain_hits.add(ng)

            if target_hits or retain_hits:
                # Compute overlap ratio
                total_ngrams = len(target_ngrams) + len(retain_ngrams)
                total_hits = len(target_hits) + len(retain_hits)
                overlap_ratio = total_hits / max(total_ngrams, 1)

                contaminated.append({
                    "line_idx": line_idx,
                    "record_id": rec.get("id", f"line_{line_idx}"),
                    "seed_topic": rec.get("seed_topic", rec.get("category", "")),
                    "benchmark": benchmark_name,
                    "target_hit_count": len(target_hits),
                    "retain_hit_count": len(retain_hits),
                    "total_hit_count": total_hits,
                    "overlap_ratio": round(overlap_ratio, 6),
                    "sample_matching_ngrams": list(target_hits | retain_hits)[:5],
                })

    logger.info("  %s: %d/%d records contaminated (%.2f%%)",
                benchmark_name, len(contaminated), total,
                100 * len(contaminated) / max(total, 1))
    return contaminated


def scan_domain(
    domain: str,
    train_path: str,
    target_field: str,
    retain_field: str,
    ngram_size: int = 13,
) -> list[dict]:
    """Full decontamination scan for one domain."""
    logger.info("Scanning %s (%s)...", domain, train_path)

    all_contaminated = []

    if domain == "math":
        # Check against MATH + GSM8K
        for loader, name in [(load_math_benchmark, "MATH"), (load_gsm8k_benchmark, "GSM8K")]:
            texts = loader()
            if not texts:
                continue
            index = build_benchmark_ngram_index(texts, ngram_size)
            logger.info("  %s ngram index: %d unique %d-grams", name, len(index), ngram_size)
            hits = scan_training_data(train_path, target_field, retain_field, index, name, ngram_size)
            all_contaminated.extend(hits)

    elif domain == "tom":
        for loader, name in [(load_social_iqa_benchmark, "SocialIQA"), (load_bigtom_benchmark, "BigToM")]:
            texts = loader()
            if not texts:
                continue
            index = build_benchmark_ngram_index(texts, ngram_size)
            logger.info("  %s ngram index: %d unique %d-grams", name, len(index), ngram_size)
            hits = scan_training_data(train_path, target_field, retain_field, index, name, ngram_size)
            all_contaminated.extend(hits)

    elif domain == "causal":
        for loader, name in [(load_cladder_benchmark, "CLadder"), (load_copa_benchmark, "COPA")]:
            texts = loader()
            if not texts:
                continue
            index = build_benchmark_ngram_index(texts, ngram_size)
            logger.info("  %s ngram index: %d unique %d-grams", name, len(index), ngram_size)
            hits = scan_training_data(train_path, target_field, retain_field, index, name, ngram_size)
            all_contaminated.extend(hits)

    elif domain == "coin":
        # COIN spans all 5 cognitive primitives — check against all relevant benchmarks
        for loader, name in [
            (load_social_iqa_benchmark, "SocialIQA"),
            (load_bigtom_benchmark, "BigToM"),
            (load_cladder_benchmark, "CLadder"),
            (load_copa_benchmark, "COPA"),
        ]:
            texts = loader()
            if not texts:
                continue
            index = build_benchmark_ngram_index(texts, ngram_size)
            logger.info("  %s ngram index: %d unique %d-grams", name, len(index), ngram_size)
            hits = scan_training_data(train_path, target_field, retain_field, index, name, ngram_size)
            all_contaminated.extend(hits)

    elif domain == "strategic":
        # STR-CC: strategic reasoning — no standard strategic reasoning benchmark,
        # check against COPA (commonsense causal) as nearest proxy
        for loader, name in [(load_copa_benchmark, "COPA")]:
            texts = loader()
            if not texts:
                continue
            index = build_benchmark_ngram_index(texts, ngram_size)
            logger.info("  %s ngram index: %d unique %d-grams", name, len(index), ngram_size)
            hits = scan_training_data(train_path, target_field, retain_field, index, name, ngram_size)
            all_contaminated.extend(hits)

    elif domain == "moral":
        # MOR-CC: moral reasoning — check against COPA + SocialIQA as proxies
        for loader, name in [(load_copa_benchmark, "COPA"), (load_social_iqa_benchmark, "SocialIQA")]:
            texts = loader()
            if not texts:
                continue
            index = build_benchmark_ngram_index(texts, ngram_size)
            logger.info("  %s ngram index: %d unique %d-grams", name, len(index), ngram_size)
            hits = scan_training_data(train_path, target_field, retain_field, index, name, ngram_size)
            all_contaminated.extend(hits)

    elif domain == "core_math":
        # CORE-MATH: formal math proofs — check against MATH + GSM8K
        for loader, name in [(load_math_benchmark, "MATH"), (load_gsm8k_benchmark, "GSM8K")]:
            texts = loader()
            if not texts:
                continue
            index = build_benchmark_ngram_index(texts, ngram_size)
            logger.info("  %s ngram index: %d unique %d-grams", name, len(index), ngram_size)
            hits = scan_training_data(train_path, target_field, retain_field, index, name, ngram_size)
            all_contaminated.extend(hits)

    elif domain == "null_cc":
        # NULL-CC: formal/informal language — check against COPA as proxy
        for loader, name in [(load_copa_benchmark, "COPA")]:
            texts = loader()
            if not texts:
                continue
            index = build_benchmark_ngram_index(texts, ngram_size)
            logger.info("  %s ngram index: %d unique %d-grams", name, len(index), ngram_size)
            hits = scan_training_data(train_path, target_field, retain_field, index, name, ngram_size)
            all_contaminated.extend(hits)

    elif domain == "spatial":
        # STP-CC: spatial-temporal — check against COPA as proxy
        for loader, name in [(load_copa_benchmark, "COPA")]:
            texts = loader()
            if not texts:
                continue
            index = build_benchmark_ngram_index(texts, ngram_size)
            logger.info("  %s ngram index: %d unique %d-grams", name, len(index), ngram_size)
            hits = scan_training_data(train_path, target_field, retain_field, index, name, ngram_size)
            all_contaminated.extend(hits)

    # Deduplicate by line_idx (a record might hit multiple benchmarks)
    by_line = {}
    for hit in all_contaminated:
        idx = hit["line_idx"]
        if idx not in by_line:
            by_line[idx] = hit
        else:
            # Merge: combine hit counts
            existing = by_line[idx]
            existing["benchmark"] += f"+{hit['benchmark']}"
            existing["total_hit_count"] += hit["total_hit_count"]
            existing["target_hit_count"] += hit["target_hit_count"]
            existing["retain_hit_count"] += hit["retain_hit_count"]
            existing["sample_matching_ngrams"].extend(hit["sample_matching_ngrams"][:3])

    deduped = sorted(by_line.values(), key=lambda x: x["overlap_ratio"], reverse=True)
    logger.info("  %s total: %d unique contaminated records", domain, len(deduped))
    return deduped


def write_cleaned_file(
    input_path: str,
    output_path: str,
    contaminated_indices: set[int],
    bad_quality_indices: set[int] | None = None,
) -> tuple[int, int]:
    """Write cleaned JSONL file with contaminated + bad records removed.

    Returns (kept, dropped) counts.
    """
    drop_indices = contaminated_indices | (bad_quality_indices or set())
    kept = 0
    dropped = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for i, line in enumerate(fin):
            if i in drop_indices:
                dropped += 1
            else:
                fout.write(line)
                kept += 1

    logger.info("  Wrote %s: kept=%d, dropped=%d", output_path, kept, dropped)
    return kept, dropped


# --- Quality filter helpers ---

def find_bad_quality_records(
    train_path: str,
    target_field: str,
    retain_field: str,
    min_length: int = 200,
) -> set[int]:
    """Find records with short/empty target or retain fields."""
    bad = set()
    with open(train_path) as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            target = rec.get(target_field, "")
            retain = rec.get(retain_field, "")
            if len(target.strip()) < min_length or len(retain.strip()) < min_length:
                bad.add(i)
    return bad


def main():
    parser = argparse.ArgumentParser(description="Decontamination scanner")
    parser.add_argument("--scan", action="store_true", help="Run contamination scan")
    parser.add_argument("--drop", action="store_true", help="Write cleaned files (drop contaminated + bad quality)")
    parser.add_argument("--report", action="store_true", help="Save detailed JSON report")
    parser.add_argument("--ngram-size", type=int, default=13, help="N-gram size for overlap (default: 13)")
    parser.add_argument("--domain", choices=["math", "tom", "causal", "coin", "strategic", "moral", "core_math", "null_cc", "spatial", "all"], default="all")
    parser.add_argument("--min-length", type=int, default=200, help="Min field length for quality filter")
    args = parser.parse_args()

    if not args.scan:
        parser.print_help()
        return

    results = {}

    # Domain configs matching seeds.py
    domain_configs = {
        "math": {
            "train_path": str(DATA_DIR / "bench" / "spl_cc_bench.jsonl"),
            "target_field": "target",
            "retain_field": "retain",
        },
        "tom": {
            "train_path": str(DATA_DIR / "bench" / "tom_cc_bench.jsonl"),
            "target_field": "target",
            "retain_field": "retain",
        },
        "causal": {
            "train_path": str(DATA_DIR / "bench" / "ctr_cc_bench.jsonl"),
            "target_field": "target_text",
            "retain_field": "retain_text",
        },
        "coin": {
            "train_path": str(DATA_DIR / "COIN" / "coin_pairs.jsonl"),
            "target_field": "target",
            "retain_field": "retain",
        },
        "strategic": {
            "train_path": str(DATA_DIR / "bench" / "str_cc_bench.jsonl"),
            "target_field": "target",
            "retain_field": "retain",
        },
        "moral": {
            "train_path": str(DATA_DIR / "bench" / "mor_cc_bench.jsonl"),
            "target_field": "target",
            "retain_field": "retain",
        },
        "core_math": {
            "train_path": str(DATA_DIR / "bench" / "core_math_bench.jsonl"),
            "target_field": "target_proof",
            "retain_field": "retain_intuition",
        },
        "null_cc": {
            "train_path": str(DATA_DIR / "bench" / "null_cc_bench.jsonl"),
            "target_field": "target_formal",
            "retain_field": "retain_informal",
        },
        "spatial": {
            "train_path": str(DATA_DIR / "bench" / "stp_cc_bench.jsonl"),
            "target_field": "target",
            "retain_field": "retain",
        },
    }

    domains = list(domain_configs.keys()) if args.domain == "all" else [args.domain]

    for domain in domains:
        cfg = domain_configs[domain]
        train_path = cfg["train_path"]

        if not Path(train_path).exists():
            logger.warning("Training data not found: %s — skipping", train_path)
            continue

        # Quality filter
        bad_quality = find_bad_quality_records(
            train_path, cfg["target_field"], cfg["retain_field"],
            min_length=args.min_length,
        )
        logger.info("%s: %d bad quality records (field < %d chars)",
                     domain, len(bad_quality), args.min_length)

        # Contamination scan
        contaminated = scan_domain(
            domain, train_path,
            cfg["target_field"], cfg["retain_field"],
            ngram_size=args.ngram_size,
        )
        contaminated_indices = {h["line_idx"] for h in contaminated}

        overlap = contaminated_indices & bad_quality
        total_drop = contaminated_indices | bad_quality

        # Count original
        with open(train_path) as f:
            n_original = sum(1 for _ in f)

        results[domain] = {
            "original_count": n_original,
            "bad_quality_count": len(bad_quality),
            "contaminated_count": len(contaminated),
            "overlap_bad_and_contaminated": len(overlap),
            "total_to_drop": len(total_drop),
            "remaining_after_drop": n_original - len(total_drop),
            "contaminated_details": contaminated,
            "bad_quality_indices": sorted(bad_quality),
        }

        logger.info("\n%s Summary:", domain.upper())
        logger.info("  Original: %d", n_original)
        logger.info("  Bad quality: %d", len(bad_quality))
        logger.info("  Contaminated: %d", len(contaminated))
        logger.info("  Overlap (bad+contam): %d", len(overlap))
        logger.info("  Total to drop: %d", len(total_drop))
        logger.info("  Remaining: %d", n_original - len(total_drop))

        # Write cleaned file
        if args.drop:
            clean_path = str(Path(train_path).with_suffix(".clean.jsonl"))
            write_cleaned_file(train_path, clean_path, contaminated_indices, bad_quality)

    # Save report (merge with existing so per-domain runs don't clobber prior scans)
    if args.report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / "decontamination_report.json"

        existing = {}
        if report_path.exists():
            try:
                with open(report_path) as f:
                    existing = json.load(f)
            except Exception as e:
                logger.warning("Could not read existing report, will overwrite: %s", e)

        # Make report JSON-serializable (remove set references)
        serializable = dict(existing)
        for domain, data in results.items():
            serializable[domain] = {
                k: v for k, v in data.items()
                if k != "bad_quality_indices" or isinstance(v, list)
            }
        with open(report_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info("Report saved to %s", report_path)

    # Print summary table
    print("\n" + "=" * 60)
    print("DECONTAMINATION SUMMARY")
    print("=" * 60)
    for domain, data in results.items():
        print(f"\n{domain.upper()} ({domain_configs[domain]['train_path']})")
        print(f"  Original records:     {data['original_count']}")
        print(f"  Bad quality:          {data['bad_quality_count']}")
        print(f"  Contaminated:         {data['contaminated_count']}")
        print(f"  Total to drop:        {data['total_to_drop']}")
        print(f"  Remaining:            {data['remaining_after_drop']}")
        if data["contaminated_details"]:
            print(f"  Top contamination by overlap ratio:")
            for hit in data["contaminated_details"][:10]:
                print(f"    line {hit['line_idx']}: {hit['benchmark']} "
                      f"overlap={hit['overlap_ratio']:.4f} "
                      f"hits={hit['total_hit_count']} "
                      f"topic={hit['seed_topic'][:50]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
