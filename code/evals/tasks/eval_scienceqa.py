"""Custom ScienceQA evaluator for scientific reasoning benchmark.

ScienceQA (Lu et al., 2022) is a multimodal science question answering
benchmark. We evaluate text-only questions (no image required) using
log-likelihood multiple-choice scoring.

Dataset: derek-thomas/ScienceQA on HuggingFace.
Format: question + choices (list) + answer (int index).
Filter: image column is None (text-only subset).

Maps to domain hypothesis: Causal + Spatial + Math (Table 10 Row 7).

Usage:
    python -m evaluation.scripts.eval_scienceqa --model llama-3-8b
    python -m evaluation.scripts.eval_scienceqa --model llama-3-8b --max-examples 500
"""

import argparse
import json
import logging
import time
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.model_loader import load_student
from utils.metrics import EvalResult, save_results

LOG_DIR = Path(__file__).parent.parent / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("eval_scienceqa")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(LOG_DIR / "eval_scienceqa.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(_fh)

try:
    from evaluation.scripts.cogbench_eval import COGBENCH_MODELS, load_model
except ImportError:
    COGBENCH_MODELS = None

RESULTS_DIR = Path(__file__).parent.parent / "results" / "scienceqa"


def compute_choice_log_prob(model, tokenizer, prompt, choice_text, device):
    """Compute length-normalized log prob of a choice given a prompt."""
    full = prompt + " " + choice_text
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(full, return_tensors="pt").input_ids.to(device)

    p_len = prompt_ids.shape[1]
    c_len = full_ids.shape[1] - p_len
    if c_len <= 0:
        return float("-inf")

    with torch.no_grad():
        logits = model(full_ids).logits

    shift_logits = logits[:, p_len - 1:-1, :]
    shift_labels = full_ids[:, p_len:]
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_lp.sum().item() / c_len


def run_heldout(model, tokenizer, model_key: str, adapter_path: str | None = None,
                max_examples: int = 500) -> dict:
    """Evaluate a live (model, tokenizer) on ScienceQA without load/unload."""
    print("\nLoading ScienceQA dataset...")
    try:
        dataset = load_dataset("derek-thomas/ScienceQA", split="test",
                               trust_remote_code=True)
    except Exception:
        try:
            dataset = load_dataset("lucasmccabe-lmi/ScienceQA", split="test",
                                   trust_remote_code=True)
        except Exception:
            dataset = load_dataset("derek-thomas/ScienceQA", split="validation",
                                   trust_remote_code=True)

    print(f"Loaded {len(dataset)} total examples")

    # Filter to text-only questions (no image required)
    dataset = dataset.filter(lambda x: x.get("image") is None or x["image"] == "")
    print(f"After text-only filter: {len(dataset)} examples")

    if max_examples > 0 and len(dataset) > max_examples:
        dataset = dataset.select(range(max_examples))

    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total = 0
    by_subject = {}

    for ex in tqdm(dataset, desc="ScienceQA"):
        question = ex["question"]
        choices = ex["choices"]
        gold = ex["answer"]  # int index into choices
        subject = ex.get("subject", "unknown")
        hint = ex.get("hint", "")

        if not choices or not question:
            continue

        # Build prompt with optional hint/context
        prompt = ""
        if hint:
            prompt += f"Context: {hint}\n\n"
        prompt += f"Question: {question}\n\nChoices:\n"
        for i, c in enumerate(choices):
            prompt += f"  ({chr(65 + i)}) {c}\n"
        prompt += "\nAnswer:"

        # Score each choice
        best_idx = 0
        best_score = float("-inf")
        for i, c in enumerate(choices):
            score = compute_choice_log_prob(
                model, tokenizer, prompt, f" ({chr(65 + i)}) {c}", device
            )
            if score > best_score:
                best_score = score
                best_idx = i

        is_correct = best_idx == gold
        if is_correct:
            correct += 1
        total += 1

        # Track by subject
        if subject not in by_subject:
            by_subject[subject] = {"correct": 0, "total": 0}
        by_subject[subject]["total"] += 1
        if is_correct:
            by_subject[subject]["correct"] += 1

    accuracy = correct / total if total > 0 else 0

    logger.info("SCIENCEQA_RESULT model=%s accuracy=%.4f correct=%d total=%d adapter=%s",
                model_key, accuracy, correct, total, adapter_path or "none")
    for subj in sorted(by_subject.keys()):
        s = by_subject[subj]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0
        logger.info("  SCIENCEQA_SUBJECT subject=%s accuracy=%.4f n=%d", subj, acc, s["total"])

    print(f"\nScienceQA Results for {model_key}:")
    print(f"  Overall: {correct}/{total} = {accuracy:.4f}")
    for subj in sorted(by_subject.keys()):
        s = by_subject[subj]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0
        print(f"  {subj}: {s['correct']}/{s['total']} = {acc:.4f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adapter_label = Path(adapter_path).name if adapter_path else "base"

    result = EvalResult(
        model=model_key, task="scienceqa", accuracy=accuracy,
        n_samples=total, experiment="lora" if adapter_path else "zero_shot",
        driver_path=adapter_path or "",
    )
    save_results([result], str(RESULTS_DIR / "scienceqa_results.csv"))

    model_meta = {}
    if COGBENCH_MODELS and model_key in COGBENCH_MODELS:
        m = COGBENCH_MODELS[model_key]
        model_meta = {"size_b": m.get("size_b"), "family": m.get("family"),
                      "arch": m.get("arch"), "tier": m.get("tier")}

    detail = {
        "model": model_key, "adapter": adapter_path,
        "accuracy": accuracy, "correct": correct, "total": total,
        "text_only": True,
        "by_subject": {k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0}
                       for k, v in by_subject.items()},
        "model_metadata": model_meta,
        "hypothesis": ["Causal", "Spatial", "Math"],
        "paper_table": "Table 10 Row 7",
    }
    with open(RESULTS_DIR / f"scienceqa_{model_key}_{adapter_label}.json", "w") as f:
        json.dump(detail, f, indent=2)

    return detail


def evaluate_scienceqa(model_key: str, adapter_path: str | None = None,
                       max_examples: int = 500,
                       use_cogbench_loader: bool = False):
    """Standalone entrypoint: load model, run heldout, unload."""
    if use_cogbench_loader and COGBENCH_MODELS and model_key in COGBENCH_MODELS:
        model, tokenizer = load_model(model_key)
    else:
        model, tokenizer, _ = load_student(model_key)
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
    try:
        detail = run_heldout(model, tokenizer, model_key,
                             adapter_path=adapter_path, max_examples=max_examples)
    finally:
        del model
        torch.cuda.empty_cache()
    return detail["accuracy"]


def main():
    parser = argparse.ArgumentParser(description="ScienceQA evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--cogbench-loader", action="store_true")
    args = parser.parse_args()
    evaluate_scienceqa(args.model, args.adapter, args.max_examples,
                       args.cogbench_loader)


if __name__ == "__main__":
    main()
