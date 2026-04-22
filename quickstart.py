#!/usr/bin/env python3
"""CogBench quickstart — score any HF causal-LM on one CogBench corpus.

Design:
  * Single file. One import path (transformers). No judge panel, no vLLM,
    no cluster logic.
  * Auto-graded only: we use the auto-gradable corpora (spl_cc for numeric,
    stp_cc for location, null_cc as a length heuristic). Judge-graded
    corpora are skipped unless the user explicitly asks; CogBench's full
    harness lives in `code/cogbench/cogbench_eval.py`.
  * `--cpu-tiny` flag uses `distilgpt2` on CPU so the script produces a
    number on a laptop in under 5 minutes, with no GPU and no gated
    checkpoint.

Example (GPU, 1 corpus, 100 pairs):

    python quickstart.py --model meta-llama/Llama-3.2-1B --corpus spl_cc --limit 100

Example (CPU laptop, tiny model):

    python quickstart.py --cpu-tiny --corpus spl_cc --limit 20

Output (stdout):

    corpus=spl_cc  n=100  target_acc=0.48  retain_acc=0.39  CA=0.09  wall=42.7s

Exit 0 on success; non-zero on configuration or runtime errors.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from load_cogbench import load_split  # noqa: E402


# ── Graders ──────────────────────────────────────────────────────────────────
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _numeric_match(prediction: str, reference: str, tol: float = 1e-3) -> bool:
    """True if `prediction` contains a number matching the last number in `reference`."""
    ref_nums = _NUM_RE.findall(reference)
    if not ref_nums:
        return False
    try:
        target = float(ref_nums[-1])
    except ValueError:
        return False
    for m in _NUM_RE.findall(prediction):
        try:
            if abs(float(m) - target) <= max(tol, tol * abs(target)):
                return True
        except ValueError:
            continue
    return False


def _location_match(prediction: str, reference: str) -> bool:
    """True if prediction echoes the final location-like token from reference."""
    tail = reference.strip().split()[-5:]
    if not tail:
        return False
    needle = " ".join(tail).lower()
    return needle in prediction.lower()


def _length_style_match(prediction: str, reference: str) -> bool:
    """Heuristic for the NULL-CC style control — declares a match on non-empty output.

    NULL-CC is a negative control; we only care that generation ran, not that
    it matches. We return True for both target and retain so CA ≈ 0 by
    construction and the caller can sanity-check the pipeline.
    """
    return len(prediction.strip()) > 0


AUTO_GRADERS: dict[str, Callable[[str, str], bool]] = {
    "spl_cc": _numeric_match,
    "stp_cc": _location_match,
    "null_cc": _length_style_match,
}


# ── Prompt templates (minimal — match CogBench eval defaults at a high level) ─
PROMPT_TEMPLATES: dict[str, tuple[str, str]] = {
    # (target_question, retain_question). `{text}` is the source passage.
    "spl_cc":    ("Given this physics description, what is the final numeric answer?\n\n{text}\n\nFinal answer: ",
                  "Given this physics description, what is the final numeric answer?\n\n{text}\n\nFinal answer: "),
    "stp_cc":    ("Given this description, where is the object?\n\n{text}\n\nLocation: ",
                  "Given this description, where is the object?\n\n{text}\n\nLocation: "),
    "null_cc":   ("Summarise this passage in one sentence:\n\n{text}\n\nSummary: ",
                  "Summarise this passage in one sentence:\n\n{text}\n\nSummary: "),
}


# ── Runner ───────────────────────────────────────────────────────────────────
def _load_model(model_id: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return model, tok


def _generate(model, tok, prompt: str, device: str, max_new_tokens: int) -> str:
    import torch
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    gen = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


def _field(record: dict, candidates: Iterable[str]) -> str | None:
    for key in candidates:
        if key in record:
            return record[key]
    return None


def run(
    model_id: str,
    corpus: str,
    split: str = "bench",
    limit: int = 100,
    device: str = "cuda",
    max_new_tokens: int = 40,
) -> dict:
    if corpus not in AUTO_GRADERS:
        raise ValueError(
            f"quickstart supports only auto-graded corpora "
            f"({', '.join(AUTO_GRADERS)}); for judge-graded corpora use "
            "code/cogbench/cogbench_eval.py."
        )
    grader = AUTO_GRADERS[corpus]
    target_tpl, retain_tpl = PROMPT_TEMPLATES[corpus]

    model, tok = _load_model(model_id, device)

    tgt_hits = ret_hits = n = 0
    t0 = time.time()
    for rec in load_split(corpus, split, limit=limit):
        target_text = _field(rec, ("target", "target_text", "target_proof", "target_formal"))
        retain_text = _field(rec, ("retain", "retain_text", "retain_intuition", "retain_informal"))
        if not target_text or not retain_text:
            continue
        tgt_pred = _generate(model, tok, target_tpl.format(text=target_text), device, max_new_tokens)
        ret_pred = _generate(model, tok, retain_tpl.format(text=retain_text), device, max_new_tokens)
        tgt_hits += int(grader(tgt_pred, target_text))
        ret_hits += int(grader(ret_pred, retain_text))
        n += 1

    wall = time.time() - t0
    target_acc = tgt_hits / n if n else 0.0
    retain_acc = ret_hits / n if n else 0.0
    return {
        "model": model_id,
        "corpus": corpus,
        "split": split,
        "n": n,
        "target_acc": round(target_acc, 4),
        "retain_acc": round(retain_acc, 4),
        "CA": round(target_acc - retain_acc, 4),
        "wall_s": round(wall, 1),
        "device": device,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CogBench quickstart — single-model CA on one corpus")
    parser.add_argument("--model", type=str,
                        help="HuggingFace model id (ignored if --cpu-tiny is set)")
    parser.add_argument("--corpus", type=str, default="spl_cc",
                        choices=sorted(AUTO_GRADERS.keys()),
                        help="Auto-graded corpus (default: spl_cc)")
    parser.add_argument("--split", type=str, choices=("train", "bench", "holdout"),
                        default="bench")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max pairs to score (default: 100)")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cuda, cpu). Default: auto-detect.")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--cpu-tiny", action="store_true",
                        help="Use distilgpt2 on CPU — produces a number in <5 min on a laptop")
    parser.add_argument("--json", action="store_true",
                        help="Emit a single JSON line instead of a human row")
    args = parser.parse_args()

    if args.cpu_tiny:
        model_id = "distilgpt2"
        device = "cpu"
    else:
        if not args.model:
            parser.error("Provide --model <hf_id>, or use --cpu-tiny for a no-GPU smoke test.")
        model_id = args.model
        if args.device:
            device = args.device
        else:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                print("error: PyTorch not installed. pip install torch transformers", file=sys.stderr)
                return 2

    try:
        result = run(
            model_id=model_id,
            corpus=args.corpus,
            split=args.split,
            limit=args.limit,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result))
    else:
        print(
            f"corpus={result['corpus']}  n={result['n']}  "
            f"target_acc={result['target_acc']}  retain_acc={result['retain_acc']}  "
            f"CA={result['CA']}  wall={result['wall_s']}s  device={result['device']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
