#!/usr/bin/env python3
"""Minimal CogBench data loader — pure Python, no torch, no HF hub.

Streams records from the local CogBench release layout:

    data/
      train/<corpus>_train.jsonl
      bench/<corpus>_bench.jsonl
      holdout/<corpus>_holdout.jsonl
      coin/<DOMAIN>_COIN/<domain>_coin_{train,bench,holdout}.jsonl

Usage as a script:

    python load_cogbench.py --corpus spl_cc --split bench --limit 3
    python load_cogbench.py --list

Usage as a library:

    from load_cogbench import load_split, list_corpora
    for record in load_split("tom_cc", "bench", limit=10):
        print(record["target"])
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "data"

PRIMARY_CORPORA = (
    "spl_cc", "tom_cc", "ctr_cc", "mor_cc",
    "str_cc", "stp_cc", "null_cc", "core_math",
)
COIN_DOMAINS = ("CAU_COIN", "MOR_COIN", "STP_COIN", "STR_COIN", "TOM_COIN")
SPLITS = ("train", "bench", "holdout")


def _resolve_path(corpus: str, split: str) -> Path:
    if corpus in PRIMARY_CORPORA:
        return DATA_ROOT / split / f"{corpus}_{split}.jsonl"
    if corpus in COIN_DOMAINS:
        return DATA_ROOT / "coin" / corpus / f"{corpus.lower()}_{split}.jsonl"
    if corpus == "coin":
        raise ValueError(
            "Use one of the five COIN sub-corpora: "
            f"{', '.join(COIN_DOMAINS)}"
        )
    raise ValueError(f"Unknown corpus {corpus!r}. Try --list.")


def load_split(corpus: str, split: str, limit: int | None = None) -> Iterator[dict]:
    """Yield records from one (corpus, split). Streams — memory O(1)."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}.")
    path = _resolve_path(corpus, split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run this script from the release root, "
            "or pass --root to point at the `data/` parent."
        )
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            yield json.loads(line)


def list_corpora() -> list[tuple[str, str, Path, int]]:
    """Enumerate every (corpus, split, path, line_count) available on disk."""
    out: list[tuple[str, str, Path, int]] = []
    for corpus in PRIMARY_CORPORA:
        for split in SPLITS:
            path = _resolve_path(corpus, split)
            if path.exists():
                out.append((corpus, split, path, _count_lines(path)))
    for domain in COIN_DOMAINS:
        for split in SPLITS:
            path = _resolve_path(domain, split)
            if path.exists():
                out.append((domain, split, path, _count_lines(path)))
    return out


def _count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def _pretty(record: dict) -> str:
    return json.dumps(record, indent=2, ensure_ascii=False)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Minimal CogBench data loader")
    parser.add_argument("--corpus", type=str,
                        help=f"One of {', '.join(PRIMARY_CORPORA)} "
                             f"or a COIN domain {', '.join(COIN_DOMAINS)}")
    parser.add_argument("--split", type=str, choices=SPLITS, default="bench")
    parser.add_argument("--limit", type=int, default=3,
                        help="Max records to print (default: 3)")
    parser.add_argument("--list", action="store_true",
                        help="List every (corpus, split) file with line counts")
    parser.add_argument("--root", type=Path, default=None,
                        help="Override the release root (parent of data/)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    global DATA_ROOT
    if args.root is not None:
        DATA_ROOT = args.root / "data"

    if args.list:
        rows = list_corpora()
        if not rows:
            print(f"No corpora found under {DATA_ROOT}.")
            return 1
        width = max(len(c) for c, *_ in rows)
        for corpus, split, path, n in rows:
            print(f"  {corpus:<{width}}  {split:<7}  {n:>7,}  {path}")
        print(f"\n{sum(n for _, _, _, n in rows):,} records across {len(rows)} files.")
        return 0

    if not args.corpus:
        parser.error("Provide --corpus or --list")

    try:
        records = list(load_split(args.corpus, args.split, limit=args.limit))
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for i, rec in enumerate(records):
        print(f"─── {args.corpus}/{args.split}[{i}] ─────────────────────")
        print(_pretty(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())