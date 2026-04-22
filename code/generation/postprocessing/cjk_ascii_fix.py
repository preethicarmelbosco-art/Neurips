"""In-place ASCII normalization of fullwidth / CJK punctuation in corpus files.

Fixes DeepSeek-style-drift punctuation (。？，＜ etc.) in the target/retain text
fields of the 8 primary corpora across train/bench/holdout splits.

Row counts and line numbers are preserved (each line is edited in place), so any
downstream files that reference line numbers remain valid.

Run:
    python -m evaluation.scripts.cjk_ascii_fix                # all 8 corpora, all splits
    python -m evaluation.scripts.cjk_ascii_fix --dry-run      # report only, no writes
    python -m evaluation.scripts.cjk_ascii_fix --split bench  # one split
"""

import argparse
import json
import re
import shutil
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs.seeds import DOMAINS, _ALL_DOMAINS as PRIMARY_DOMAINS

# Explicit CJK punctuation map (U+3000 block — no ASCII offset rule applies).
# Do NOT touch smart/curly quotes (U+2018/2019/201C/201D) — those are ordinary
# English typographic punctuation and not in scope for the CJK cleanup.
_EXPLICIT_MAP = {
    "\u3000": " ",   # ideographic space
    "\u3001": ",",   # 、 ideographic comma
    "\u3002": ".",   # 。 ideographic full stop
    "\u3003": '"',   # 〃 ditto
}

_FULLWIDTH_RE = re.compile(r"[\uff01-\uff5e]")           # fullwidth ASCII
_CJK_PUNCT_RE = re.compile(r"[\u3000-\u303f\uff00\uff5f-\uffef]")  # other CJK punct
_IDEOGRAPH_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]"
)


# Known multi-char CJK word leaks from DeepSeek-generated text.
# These must be replaced BEFORE the generic ideograph regex drops them.
_MULTICHAR_LEAKS = {
    "호텔": "hotel",  # Korean hotel (STP-CC row 7700)
}


def normalize_text(text: str) -> tuple[str, dict]:
    """Return (normalized_text, counts). Counts = how many of each class were changed."""
    counts = {"fullwidth_ascii": 0, "explicit_punct": 0, "other_cjk_punct": 0, "ideograph": 0, "multichar_leak": 0}

    # 0) Multi-char CJK word leaks (replace with English equivalent)
    for src, dst in _MULTICHAR_LEAKS.items():
        if src in text:
            counts["multichar_leak"] += text.count(src)
            text = text.replace(src, dst)

    # 1) Fullwidth ASCII U+FF01–FF5E -> subtract 0xFEE0
    def _sub_fw(m):
        counts["fullwidth_ascii"] += 1
        return chr(ord(m.group(0)) - 0xFEE0)
    text = _FULLWIDTH_RE.sub(_sub_fw, text)

    # 2) Explicit map (U+3000 block + smart quotes)
    out_chars = []
    for ch in text:
        if ch in _EXPLICIT_MAP:
            out_chars.append(_EXPLICIT_MAP[ch])
            counts["explicit_punct"] += 1
        else:
            out_chars.append(ch)
    text = "".join(out_chars)

    # 3) Any remaining CJK punctuation (outside the explicit map) — drop
    def _sub_other(m):
        counts["other_cjk_punct"] += 1
        return ""
    text = _CJK_PUNCT_RE.sub(_sub_other, text)

    # 4) Actual ideographs / kana / hangul — drop (these should not appear; track if they do)
    def _sub_id(m):
        counts["ideograph"] += 1
        return ""
    text = _IDEOGRAPH_RE.sub(_sub_id, text)

    return text, counts


_SPLIT_KEY = {"train": "train_data", "bench": "bench_data", "holdout": "holdout_data"}


def patch_file(path: Path, target_field: str, retain_field: str, dry_run: bool) -> dict:
    """Normalize target/retain fields line-by-line. Row count preserved."""
    result = {
        "path": str(path),
        "rows": 0,
        "changed_rows": 0,
        "counts": {"fullwidth_ascii": 0, "explicit_punct": 0, "other_cjk_punct": 0, "ideograph": 0, "multichar_leak": 0},
    }
    if not path.exists():
        result["status"] = "missing"
        return result

    tmp = path.with_suffix(path.suffix + ".tmp")
    backup = path.with_suffix(path.suffix + ".pre_cjk_ascii.bak")

    with open(path, "r", encoding="utf-8") as f_in, open(tmp, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.rstrip("\n")
            if not line:
                f_out.write("\n")
                continue
            result["rows"] += 1
            obj = json.loads(line)
            changed = False
            for field in (target_field, retain_field):
                if field in obj and isinstance(obj[field], str):
                    new_text, counts = normalize_text(obj[field])
                    if new_text != obj[field]:
                        obj[field] = new_text
                        changed = True
                        for k, v in counts.items():
                            result["counts"][k] += v
            if changed:
                result["changed_rows"] += 1
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if dry_run:
        tmp.unlink()
        result["status"] = "dry-run"
        return result

    # Only back up and swap if we actually changed something
    if result["changed_rows"] == 0:
        tmp.unlink()
        result["status"] = "unchanged"
        return result

    shutil.copy2(path, backup)
    tmp.replace(path)
    result["status"] = "patched"
    result["backup"] = str(backup)
    return result


def main():
    ap = argparse.ArgumentParser(description="ASCII-normalize fullwidth/CJK punctuation in corpus files.")
    ap.add_argument("--split", choices=["train", "bench", "holdout", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true", help="Report only; no file writes.")
    args = ap.parse_args()

    splits = ["train", "bench", "holdout"] if args.split == "all" else [args.split]

    report = {"timestamp": datetime.utcnow().isoformat(), "dry_run": args.dry_run, "files": []}

    print(f"{'File':<60} {'rows':>6} {'changed':>8} {'fw':>5} {'exp':>5} {'oth':>5} {'ide':>5} {'mul':>5} status")
    print("-" * 130)

    for dom in PRIMARY_DOMAINS:
        cfg = DOMAINS[dom]
        for split in splits:
            key = _SPLIT_KEY[split]
            if key not in cfg:
                continue
            path = Path(cfg[key])
            r = patch_file(path, cfg["target_field"], cfg["retain_field"], args.dry_run)
            report["files"].append(r)
            c = r["counts"]
            status = r.get("status", "?")
            rel = path.relative_to(path.parents[2]) if len(path.parents) >= 3 else path.name
            print(f"{str(rel):<60} {r['rows']:>6} {r['changed_rows']:>8} "
                  f"{c['fullwidth_ascii']:>5} {c['explicit_punct']:>5} "
                  f"{c['other_cjk_punct']:>5} {c['ideograph']:>5} {c['multichar_leak']:>5} {status}")

    # Save report
    out_dir = Path(__file__).resolve().parent.parent / "results" / "embedding_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("cjk_ascii_fix_dry_run.json" if args.dry_run else "cjk_ascii_fix_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
