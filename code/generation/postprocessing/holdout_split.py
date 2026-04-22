"""Split a completed corpus JSONL into train + holdout sets.

Produces stratified holdout samples (by category × difficulty where available)
to ensure uniform coverage across all cells.

Usage:
    python -m scripts.holdout_split --corpus tom_cc --holdout-size 1000
    python -m scripts.holdout_split --corpus spl_cc --holdout-size 1000
    python -m scripts.holdout_split --all --holdout-size 1000
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
HOLDOUT_DIR = DATA_DIR / "holdout"

# Corpus configs: input filename and stratification key(s)
CORPUS_CONFIGS = {
    "spl_cc": {
        "input": "SPLCCDataset.jsonl",
        "stratify_keys": None,  # No category/difficulty — pure random
    },
    "tom_cc": {
        "input": "tom_contrastive_pairs.jsonl",
        "stratify_keys": ("category", "difficulty"),
    },
    "ctr_cc": {
        # CTR has inline holdout in writer, but this script can re-split if needed.
        # CTR uses target_text/retain_text instead of target/retain.
        "input": "ctr_cc_pairs.jsonl",
        "stratify_keys": ("category", "difficulty"),
    },
    "mor_cc": {
        "input": "mor_cc_pairs.jsonl",
        "stratify_keys": ("category", "difficulty"),
    },
    "str_cc": {
        "input": "str_cc_pairs.jsonl",
        "stratify_keys": ("category", "difficulty"),
    },
    "stp_cc": {
        "input": "stp_cc_pairs.jsonl",
        "stratify_keys": ("category", "difficulty"),
    },
    "coin_cc": {
        "input": "coin_pairs.jsonl",
        "stratify_keys": ("theme", "category", "difficulty"),
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


def stratified_split(
    records: list[dict],
    holdout_size: int,
    stratify_keys: tuple[str, ...] | None,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """Split records into (train, holdout) with stratification."""

    if stratify_keys is None:
        # Pure random split
        shuffled = list(records)
        rng.shuffle(shuffled)
        holdout_size = min(holdout_size, len(shuffled))
        return shuffled[holdout_size:], shuffled[:holdout_size]

    # Group by stratification cell
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for rec in records:
        key = tuple(rec.get(k, "unknown") for k in stratify_keys)
        cells[key].append(rec)

    # Calculate per-cell holdout count (proportional, at least 1 per cell)
    n_cells = len(cells)
    if n_cells == 0:
        return records, []

    base_per_cell = holdout_size // n_cells
    remainder = holdout_size % n_cells

    # Sort cells for deterministic ordering
    sorted_keys = sorted(cells.keys())

    train_all = []
    holdout_all = []

    for i, key in enumerate(sorted_keys):
        cell_records = cells[key]
        rng.shuffle(cell_records)

        # Extra 1 for the first `remainder` cells
        n_hold = base_per_cell + (1 if i < remainder else 0)
        n_hold = min(n_hold, len(cell_records))

        holdout_all.extend(cell_records[:n_hold])
        train_all.extend(cell_records[n_hold:])

    return train_all, holdout_all


def split_corpus(corpus_name: str, holdout_size: int, seed: int = 42):
    """Split a single corpus."""
    cfg = CORPUS_CONFIGS.get(corpus_name)
    if cfg is None:
        print(f"Unknown corpus: {corpus_name}. Available: {list(CORPUS_CONFIGS.keys())}")
        return False

    input_path = DATA_DIR / cfg["input"]
    if not input_path.exists():
        print(f"  SKIP {corpus_name}: {input_path} not found")
        return False

    records = load_jsonl(input_path)
    total = len(records)

    if total < holdout_size + 100:
        print(f"  SKIP {corpus_name}: only {total} records (need {holdout_size} holdout + train)")
        return False

    rng = random.Random(seed)
    train, holdout = stratified_split(records, holdout_size, cfg["stratify_keys"], rng)

    holdout_path = HOLDOUT_DIR / f"{corpus_name}_holdout.jsonl"
    train_path = DATA_DIR / f"{corpus_name}_train.jsonl"

    write_jsonl(holdout_path, holdout)
    write_jsonl(train_path, train)

    print(f"  {corpus_name}: {total} total → {len(train)} train + {len(holdout)} holdout")

    # Print stratification summary if applicable
    if cfg["stratify_keys"]:
        cell_counts = defaultdict(int)
        for rec in holdout:
            key = tuple(rec.get(k, "?") for k in cfg["stratify_keys"])
            cell_counts[key] += 1
        print(f"    Holdout cells: {len(cell_counts)} | "
              f"min={min(cell_counts.values())} max={max(cell_counts.values())} "
              f"per cell")

    return True


def main():
    parser = argparse.ArgumentParser(description="Split corpus into train + holdout")
    parser.add_argument("--corpus", type=str, help="Corpus name (e.g., tom_cc)")
    parser.add_argument("--all", action="store_true", help="Split all available corpora")
    parser.add_argument("--holdout-size", type=int, default=1000,
                        help="Number of holdout pairs (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not args.corpus and not args.all:
        parser.print_help()
        sys.exit(1)

    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        print(f"Splitting all corpora (holdout={args.holdout_size}, seed={args.seed}):")
        for name in CORPUS_CONFIGS:
            split_corpus(name, args.holdout_size, args.seed)
    else:
        split_corpus(args.corpus, args.holdout_size, args.seed)


if __name__ == "__main__":
    main()
