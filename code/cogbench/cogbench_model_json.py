"""Single chokepoint for writing per-model CogBench JSONs.

INVARIANT: no write may reduce the set of corpora already on disk.

All code paths that persist `cogbench/<model>.json` — per-corpus saves,
end-of-run saves, grader merges, and cross-node sync pulls — MUST call
`safe_write_model_json` (or `safe_merge_tree` for bulk merge) so that a
partial-corpus payload can never wipe a more-complete on-disk version.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def load_existing(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _pick_corpus(existing: dict, incoming: dict) -> dict:
    """Prefer newer generation_timestamp; fall back to keeping existing."""
    te = (existing or {}).get("generation_timestamp") or ""
    ti = (incoming or {}).get("generation_timestamp") or ""
    return incoming if ti > te else existing


def merge_results(existing: dict, incoming: dict) -> dict:
    merged = dict(existing or {})
    for corpus, data in (incoming or {}).items():
        if corpus in merged:
            merged[corpus] = _pick_corpus(merged[corpus], data)
        else:
            merged[corpus] = data
    return merged


def atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


_META_KEYS = ("size_b", "family", "arch", "tier", "hf_id", "hardware")


def safe_write_model_json(
    path: Path,
    model_key: str,
    new_results: dict,
    model_meta: dict | None = None,
) -> dict:
    """Write per-model JSON while preserving existing corpora.

    Reads the existing JSON at `path`, unions its `results` with `new_results`
    (per-corpus; newer `generation_timestamp` wins), and atomically writes the
    merged payload (tmp + fsync + rename). Metadata is taken from `model_meta`
    when provided, else carried forward from the existing file.

    Returns the merged payload dict.
    """
    existing = load_existing(path)
    merged_results = merge_results(existing.get("results") or {}, new_results or {})
    payload: dict = {"model": model_key}
    if model_meta:
        for k in _META_KEYS:
            if k in model_meta and model_meta[k] is not None:
                payload[k] = model_meta[k]
    for k in _META_KEYS:
        if k not in payload and k in existing:
            payload[k] = existing[k]
    payload["results"] = merged_results
    atomic_write_json(path, payload)
    return payload


def safe_merge_tree(src_dir: Path, dst_dir: Path) -> dict:
    """Merge every `<model>.json` in `src_dir` into the matching file in `dst_dir`.

    Intended for sync pipelines: rsync to `<dst>/.incoming/`, then call this
    with src=`<dst>/.incoming/` and dst=`<dst>/` so that partial-corpus source
    files never wipe a more-complete destination.

    Returns `{model: sorted(corpora)}` after merge.
    """
    report: dict[str, list[str]] = {}
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.glob("*.json")):
        if src.name.startswith("percategory_") or src.name.startswith("table"):
            continue
        try:
            src_payload = json.loads(src.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(src_payload, dict):
            continue
        model_key = src_payload.get("model") or src.stem
        dst_path = dst_dir / src.name
        meta = {k: src_payload[k] for k in _META_KEYS if k in src_payload}
        merged = safe_write_model_json(
            dst_path, model_key,
            src_payload.get("results") or {},
            model_meta=meta,
        )
        report[model_key] = sorted(merged.get("results", {}).keys())
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    mt = sub.add_parser("merge-tree", help="Merge src_dir/*.json into dst_dir/*.json")
    mt.add_argument("src")
    mt.add_argument("dst")
    args = ap.parse_args()

    if args.command == "merge-tree":
        report = safe_merge_tree(Path(args.src), Path(args.dst))
        if not report:
            print("  (no files merged)")
        for model, corpora in sorted(report.items()):
            print(f"  {model}: corpora={corpora}")
