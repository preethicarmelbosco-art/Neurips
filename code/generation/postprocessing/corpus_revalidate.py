"""Re-validate contrastive pairs through 3-judge consensus gate.

Supports MOR-CC, NULL-CC, CORE-MATH corpora.

Usage:
    python -m scripts.corpus_revalidate --corpus mor --input data/mor_cc_pairs.jsonl
    python -m scripts.corpus_revalidate --corpus null_cc --input data/train/null_cc_train.jsonl data/holdout/null_cc_holdout.jsonl
    python -m scripts.corpus_revalidate --corpus core_math --input data/train/core_math_train.jsonl data/holdout/core_math_holdout.jsonl
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

load_dotenv(PROJECT_ROOT / ".env")

from corpora.judge_panel import JudgePanel, load_judge_configs_from_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# --- Corpus-specific config ---

CORPUS_CONFIG = {
    "mor": {
        "imports": ("ContrastiveData.src.mor_models", "MORJudgeVerdict"),
        "prompts": ("ContrastiveData.src.mor_validator", "MOR_JUDGE_SYSTEM_PROMPT", "MOR_JUDGE_USER_PROMPT"),
        "verdict_field": "has_moral_evaluation",
        "evidence_field": "flagged_phrases",
        "retain_key": "retain",
        "target_key": "target",
        "prefix": "mor_cc",
    },
    "null_cc": {
        "imports": ("ContrastiveData.src.null_cc_models", "NullCCJudgeVerdict"),
        "prompts": ("ContrastiveData.src.null_cc_validator", "NULL_CC_JUDGE_SYSTEM_PROMPT", "NULL_CC_JUDGE_USER_PROMPT"),
        "verdict_field": "has_informal_language",
        "evidence_field": "flagged_phrases",
        "judge_text_key": "target_formal",  # judge validates formal text for informal leakage
        "retain_key": "retain_informal",
        "target_key": "target_formal",
        "prefix": "null_cc",
    },
    "core_math": {
        "imports": ("ContrastiveData.src.core_math_models", "CoreMathJudgeVerdict"),
        "prompts": ("ContrastiveData.src.core_math_validator", "CORE_MATH_JUDGE_SYSTEM_PROMPT", "CORE_MATH_JUDGE_USER_PROMPT"),
        "verdict_field": "has_formal_math",
        "evidence_field": "flagged_phrases",
        "retain_key": "retain_intuition",
        "target_key": "target_proof",
        "prefix": "core_math",
    },
}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def has_cjk(text: str) -> bool:
    """Check if text contains CJK (Chinese/Japanese/Korean) characters."""
    return any("\u4e00" <= c <= "\u9fff" or
               "\u3040" <= c <= "\u309f" or
               "\u30a0" <= c <= "\u30ff" for c in text)


def normalize_for_dedup(text: str) -> str:
    """Normalize text for duplicate detection."""
    text = unicodedata.normalize("NFKD", text).lower()
    text = " ".join(text.split())
    return text


def quality_filter(records: list[dict], target_key: str, retain_key: str) -> tuple[list[dict], dict]:
    """Post-judge quality filter: CJK removal and exact dedup.

    Returns: (cleaned_records, quality_stats)
    """
    cjk_dropped = []
    seen_hashes = {}
    dup_dropped = []
    clean = []

    for i, rec in enumerate(records):
        target = rec.get(target_key, "")
        retain = rec.get(retain_key, "")

        # CJK check
        if has_cjk(target) or has_cjk(retain):
            cjk_dropped.append(rec)
            continue

        # Exact dedup on normalized target hash
        h = hashlib.sha256(normalize_for_dedup(target).encode()).hexdigest()
        if h in seen_hashes:
            dup_dropped.append(rec)
            continue
        seen_hashes[h] = i

        clean.append(rec)

    stats = {
        "cjk_dropped": len(cjk_dropped),
        "duplicates_dropped": len(dup_dropped),
    }
    return clean, stats


async def revalidate(
    pairs: list[dict],
    judge: JudgePanel,
    retain_key: str,
    max_concurrent: int = 3,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Re-validate pairs through judge panel.

    Returns: (accepted, rejected, verdict_log)
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    accepted = []
    rejected = []
    verdict_log = []

    async def process_one(pair: dict) -> None:
        async with semaphore:
            retain_text = pair.get(retain_key, "")
            pair_id = pair.get("id", "unknown")
            category = pair.get("category", "unknown")

            try:
                flagged, reason, evidence = await judge.vote(retain_text)
            except Exception as e:
                logger.warning("Judge error for %s: %s — accepting by default", pair_id, e)
                accepted.append(pair)
                verdict_log.append({
                    "pair_id": pair_id,
                    "category": category,
                    "flagged": False,
                    "reason": f"ERROR: {e}",
                })
                return

            log_entry = {
                "pair_id": pair_id,
                "category": category,
                "flagged": flagged,
                "reason": reason,
                "evidence": evidence[:5] if evidence else [],
            }
            verdict_log.append(log_entry)

            if flagged:
                pair_with_verdict = {**pair, "_revalidation_reason": reason}
                rejected.append(pair_with_verdict)
            else:
                accepted.append(pair)

    tasks = [process_one(p) for p in pairs]

    done = 0
    batch_size = 10
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        await asyncio.gather(*batch)
        done += len(batch)
        if done % 50 == 0 or done == len(tasks):
            logger.info(
                "Progress: %d/%d | Accepted: %d | Rejected: %d",
                done, len(tasks), len(accepted), len(rejected),
            )

    return accepted, rejected, verdict_log


def compute_table2b_stats(verdict_log: list[dict]) -> dict:
    total = len(verdict_log)
    if total == 0:
        return {}

    all_clean = 0
    one_dissent = 0
    majority_reject = 0
    judge_dissents = Counter()

    for entry in verdict_log:
        reason = entry.get("reason", "")
        if "ERROR" in reason:
            continue

        true_count = reason.count("=TRUE")
        false_count = reason.count("=FALSE")

        if true_count == 0:
            all_clean += 1
        elif true_count == 1:
            one_dissent += 1
            for part in reason.split("[")[1].split("]")[0].split(",") if "[" in reason else []:
                part = part.strip()
                if "=TRUE" in part:
                    judge_name = part.split("=")[0].strip()
                    judge_dissents[judge_name] += 1
        else:
            majority_reject += 1

    return {
        "total_pairs_judged": total,
        "3_of_3_agree_clean": all_clean,
        "2_of_3_agree_clean_1_dissent": one_dissent,
        "majority_reject_2plus_flag": majority_reject,
        "rejection_rate": majority_reject / total if total > 0 else 0,
        "single_dissent_rate": one_dissent / total if total > 0 else 0,
        "per_judge_dissent_counts": dict(judge_dissents),
    }


def compute_category_stats(verdict_log: list[dict]) -> dict:
    cat_total = defaultdict(int)
    cat_rejected = defaultdict(int)

    for entry in verdict_log:
        cat = entry.get("category", "unknown")
        cat_total[cat] += 1
        if entry.get("flagged"):
            cat_rejected[cat] += 1

    return {
        cat: {
            "total": cat_total[cat],
            "rejected": cat_rejected[cat],
            "rate": cat_rejected[cat] / cat_total[cat] if cat_total[cat] > 0 else 0,
        }
        for cat in sorted(cat_total.keys())
    }


def main():
    parser = argparse.ArgumentParser(description="Re-validate corpus pairs through consensus gate")
    parser.add_argument("--corpus", type=str, required=True,
                        choices=list(CORPUS_CONFIG.keys()),
                        help="Corpus to revalidate")
    parser.add_argument("--input", type=str, nargs="+", required=True,
                        help="Path(s) to input JSONL files (multiple files concatenated)")
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="Limit number of pairs to process (0 = all)")
    parser.add_argument("--max-concurrent", type=int, default=3,
                        help="Max concurrent judge requests")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as first input)")
    args = parser.parse_args()

    cfg = CORPUS_CONFIG[args.corpus]
    prefix = cfg["prefix"]

    # Dynamic imports
    import importlib
    mod_path, cls_name = cfg["imports"]
    model_mod = importlib.import_module(mod_path)
    response_model = getattr(model_mod, cls_name)

    prompt_mod_path, sys_name, usr_name = cfg["prompts"]
    prompt_mod = importlib.import_module(prompt_mod_path)
    system_prompt = getattr(prompt_mod, sys_name)
    user_prompt = getattr(prompt_mod, usr_name)

    # Load all input files
    pairs = []
    for inp in args.input:
        input_path = Path(inp)
        if not input_path.is_absolute():
            input_path = PROJECT_ROOT / input_path
        if not input_path.exists():
            logger.error("Input file not found: %s", input_path)
            sys.exit(1)
        loaded = load_jsonl(input_path)
        logger.info("Loaded %d pairs from %s", len(loaded), input_path)
        pairs.extend(loaded)

    logger.info("Total pairs loaded: %d", len(pairs))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    else:
        output_dir = Path(args.input[0]).parent
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]
        logger.info("Limited to first %d pairs", len(pairs))

    # Set up judge panel
    judge_configs = load_judge_configs_from_env(prefix="JUDGE")
    if not judge_configs:
        logger.error("No JUDGE configs found in environment. Set JUDGE1_MODEL etc.")
        sys.exit(1)

    logger.info("Judge panel: %s", [c.label for c in judge_configs])

    judge = JudgePanel(
        configs=judge_configs,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt,
        response_model=response_model,
        verdict_field=cfg["verdict_field"],
        evidence_field=cfg["evidence_field"],
    )

    # Run revalidation — use judge_text_key if specified, else retain_key
    judge_text_key = cfg.get("judge_text_key", cfg["retain_key"])
    accepted, rejected, verdict_log = asyncio.run(
        revalidate(pairs, judge, judge_text_key, max_concurrent=args.max_concurrent)
    )

    # Post-judge quality filter: CJK + dedup
    target_key = cfg.get("target_key", "target")
    retain_key = cfg["retain_key"]
    cleaned, qf_stats = quality_filter(accepted, target_key, retain_key)
    logger.info(
        "Quality filter: CJK dropped %d, duplicates dropped %d",
        qf_stats["cjk_dropped"], qf_stats["duplicates_dropped"],
    )

    # Write outputs
    accepted_path = output_dir / f"{prefix}_accepted.jsonl"
    rejected_path = output_dir / f"{prefix}_rejected.jsonl"
    verdicts_path = output_dir / f"{prefix}_revalidation_verdicts.jsonl"
    stats_path = output_dir / f"{prefix}_revalidation_stats.json"

    write_jsonl(accepted_path, cleaned)
    write_jsonl(rejected_path, rejected)
    write_jsonl(verdicts_path, verdict_log)

    # Compute and save stats
    table2b = compute_table2b_stats(verdict_log)
    category_stats = compute_category_stats(verdict_log)

    stats = {
        "corpus": args.corpus,
        "input_files": args.input,
        "total_input": len(pairs),
        "judge_accepted": len(accepted),
        "judge_rejected": len(rejected),
        "quality_filter": qf_stats,
        "final_accepted": len(cleaned),
        "table2b": table2b,
        "per_category": category_stats,
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("=== Revalidation Complete ===")
    logger.info("Input:         %d pairs", len(pairs))
    logger.info("Judge accept:  %d (%.1f%%)", len(accepted), 100 * len(accepted) / len(pairs))
    logger.info("Judge reject:  %d (%.1f%%)", len(rejected), 100 * len(rejected) / len(pairs))
    logger.info("CJK dropped:   %d", qf_stats["cjk_dropped"])
    logger.info("Dups dropped:  %d", qf_stats["duplicates_dropped"])
    logger.info("Final clean:   %d (%.1f%%)", len(cleaned), 100 * len(cleaned) / len(pairs))
    logger.info("Outputs:")
    logger.info("  Accepted:  %s", accepted_path)
    logger.info("  Rejected:  %s", rejected_path)
    logger.info("  Verdicts:  %s", verdicts_path)
    logger.info("  Stats:     %s", stats_path)

    if table2b:
        logger.info("=== Table 2b Stats ===")
        logger.info("  3/3 agree clean:    %d", table2b.get("3_of_3_agree_clean", 0))
        logger.info("  2/3 clean (1 diss): %d", table2b.get("2_of_3_agree_clean_1_dissent", 0))
        logger.info("  Majority reject:    %d", table2b.get("majority_reject_2plus_flag", 0))
        logger.info("  Rejection rate:     %.1f%%", 100 * table2b.get("rejection_rate", 0))
        if table2b.get("per_judge_dissent_counts"):
            logger.info("  Per-judge dissents: %s", table2b["per_judge_dissent_counts"])

    if category_stats:
        sorted_cats = sorted(category_stats.items(), key=lambda x: x[1]["rate"], reverse=True)
        logger.info("=== Per-Category Rejection Rates (top 5) ===")
        for cat, s in sorted_cats[:5]:
            logger.info("  %s: %d/%d (%.1f%%)", cat, s["rejected"], s["total"], 100 * s["rate"])


if __name__ == "__main__":
    main()
