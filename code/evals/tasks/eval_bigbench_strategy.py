"""Custom BigBench Strategic QA evaluator for strategic reasoning benchmark.

BigBench Strategic QA tests whether models can identify optimal strategies
in game-theoretic and decision-making scenarios. Each example presents a
scenario with multiple choice answers about the best strategic move.

This is the PRIMARY benchmark for the "strategic" domain (STR-CC corpus)
in the CogBench evaluation pipeline.

Evaluation: log-likelihood multiple-choice scoring over candidate answers.

Dataset: tasksource/bigbench (strategic_qa) on HuggingFace
Fallback: google/bigbench (strategic_qa)
Fallback: ContrastiveData/data/bigbench_strategy.jsonl

Usage:
    python -m evaluation.scripts.eval_bigbench_strategy --model llama-3-8b
    python -m evaluation.scripts.eval_bigbench_strategy --model llama-3-8b --adapter path/to/lora
    python -m evaluation.scripts.eval_bigbench_strategy --model all
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
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from _logprob_helpers import choice_logprob
from configs.seeds import STUDENT_MODELS
from utils.model_loader import load_student
from utils.metrics import EvalResult, save_results

LOG_DIR = Path(__file__).parent.parent / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("eval_bigbench_strategy")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(LOG_DIR / "eval_bigbench_strategy.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(_fh)

try:
    from evaluation.scripts.cogbench_eval import COGBENCH_MODELS, load_model
except ImportError:
    COGBENCH_MODELS = None

RESULTS_DIR = Path(__file__).parent.parent / "results" / "bigbench_strategy"

CHOICE_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def compute_log_prob(model, tokenizer, prompt: str, completion: str, device: str) -> float:
    """Mean per-token log P(completion | prompt).

    Thin wrapper around shared :func:`choice_logprob` — tokenises prompt and
    continuation separately, strips BOS, applies chat template for instruct
    models, and handles the space-prefix contract via the helper.
    """
    return choice_logprob(model, tokenizer, prompt, completion, device, reduce="byte_mean")


def _extract_fields(example):
    """Extract scenario, choices, and correct answer from a BigBench example.

    Handles both naming conventions:
      - inputs / input  for the scenario text
      - multiple_choice_targets / choices  for candidate answers
      - multiple_choice_scores  carries the correct-answer mask (preferred)
      - targets / target  for free-form answer text (fallback)
    """
    scenario = (example.get("inputs")
                or example.get("input")
                or example.get("question", ""))

    choices = (example.get("multiple_choice_targets")
               or example.get("multiple_choice_options")
               or example.get("choices")
               or example.get("options"))
    if choices is None:
        choices = []

    if isinstance(choices, str):
        try:
            choices = json.loads(choices)
        except (json.JSONDecodeError, ValueError):
            choices = [c.strip() for c in choices.split(",") if c.strip()]

    # Preferred path: BigBench provides multiple_choice_scores with a 1-hot
    # mask on multiple_choice_targets. The free-form `targets` field is a
    # long explanatory essay (e.g., "No. St. Vitus's Dance, also called...")
    # which never matches the option text — using it gave 0% accuracy.
    target = ""
    scores = example.get("multiple_choice_scores")
    if isinstance(scores, list) and scores and choices:
        try:
            idx = max(range(len(scores)), key=lambda i: scores[i])
            if 0 <= idx < len(choices):
                target = str(choices[idx]).strip()
        except (TypeError, ValueError):
            target = ""

    if not target:
        raw = example.get("targets", example.get("target", example.get("answer", "")))
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        target = str(raw).strip()

    return scenario, choices, target


def _build_prompt(scenario: str, choices: list[str]) -> str:
    """Build the MC prompt with lettered choices."""
    lines = ["Read the following scenario and choose the best strategic response.", ""]
    lines.append(f"Scenario: {scenario}")
    lines.append("")
    for i, choice in enumerate(choices):
        letter = CHOICE_LETTERS[i] if i < len(CHOICE_LETTERS) else str(i + 1)
        lines.append(f"{letter}) {choice}")
    lines.append("")
    lines.append("The best answer is:")
    return "\n".join(lines)


def run_heldout(model, tokenizer, model_key: str, adapter_path: str | None = None,
                max_examples: int = 500) -> dict:
    """Evaluate a live (model, tokenizer) on BigBench Strategic QA without load/unload."""
    print(f"\nLoading BigBench Strategic QA dataset...")
    dataset = None

    # Primary: tasksource/bigbench
    try:
        dataset = load_dataset("tasksource/bigbench", "strategyqa",
                               split="validation", trust_remote_code=True)
        print(f"  Loaded from tasksource/bigbench: {len(dataset)} examples")
    except Exception as e:
        logger.debug("tasksource/bigbench load failed: %s", e)

    # Fallback: google/bigbench
    if dataset is None:
        try:
            dataset = load_dataset("google/bigbench", "strategyqa",
                                   split="validation", trust_remote_code=True)
            print(f"  Loaded from google/bigbench: {len(dataset)} examples")
        except Exception as e:
            logger.debug("google/bigbench load failed: %s", e)

    # Third fallback: local JSONL
    if dataset is None:
        local_path = (Path(__file__).parent.parent.parent.parent
                      / "ContrastiveData" / "data" / "bigbench_strategy.jsonl")
        if local_path.exists():
            print(f"  Loading from local: {local_path}")
            records = []
            with open(local_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            dataset = records
            print(f"  Loaded {len(dataset)} examples from local JSONL")
        else:
            print(f"  Could not load BigBench Strategic QA from any source.")
            print(f"  Local path not found: {local_path}")
            return {"accuracy": 0.0, "correct": 0, "total": 0, "by_num_choices": {}}

    # Cap to max_examples
    if len(dataset) > max_examples:
        if hasattr(dataset, 'select'):
            dataset = dataset.select(range(max_examples))
        else:
            dataset = dataset[:max_examples]

    print(f"Evaluating on {len(dataset)} examples")

    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total = 0
    by_num_choices = {}  # accuracy breakdown by number of choices

    for example in tqdm(dataset, desc="BigBench Strategic QA"):
        scenario, choices, target = _extract_fields(example)

        if not scenario or not choices or not target:
            continue

        prompt = _build_prompt(scenario, choices)
        num_choices = len(choices)

        # Score each candidate answer
        best_idx = -1
        best_score = float("-inf")
        for i, choice in enumerate(choices):
            letter = CHOICE_LETTERS[i] if i < len(CHOICE_LETTERS) else str(i + 1)
            # Score both the letter and the full answer text
            score = compute_log_prob(model, tokenizer, prompt,
                                     f" {letter}) {choice}", device)
            if score > best_score:
                best_score = score
                best_idx = i

        # Determine correctness: match by text or by letter
        predicted = choices[best_idx] if best_idx >= 0 else ""
        predicted_letter = CHOICE_LETTERS[best_idx] if best_idx >= 0 and best_idx < len(CHOICE_LETTERS) else ""

        # Target could be the answer text, a letter, or an index
        is_correct = False
        target_lower = target.lower().strip().rstrip(")")
        if predicted.lower().strip() == target_lower:
            is_correct = True
        elif predicted_letter.lower() == target_lower:
            is_correct = True
        elif target_lower.isdigit() and int(target_lower) == best_idx:
            is_correct = True
        else:
            # Check if target text matches the choice at best_idx
            for ci, c in enumerate(choices):
                if c.lower().strip() == target_lower:
                    if ci == best_idx:
                        is_correct = True
                    break

        if is_correct:
            correct += 1
        total += 1

        # Track by number of choices
        nc_key = str(num_choices)
        if nc_key not in by_num_choices:
            by_num_choices[nc_key] = {"correct": 0, "total": 0}
        by_num_choices[nc_key]["total"] += 1
        if is_correct:
            by_num_choices[nc_key]["correct"] += 1

    accuracy = correct / total if total > 0 else 0

    logger.info("BIGBENCH_STRATEGY_RESULT model=%s accuracy=%.4f correct=%d total=%d adapter=%s",
                model_key, accuracy, correct, total, adapter_path or "none")
    for nc in sorted(by_num_choices.keys(), key=lambda x: int(x) if x.isdigit() else 99):
        c = by_num_choices[nc]
        acc = c["correct"] / c["total"] if c["total"] > 0 else 0
        logger.info("  BIGBENCH_STRATEGY_BYCHOICES n=%s accuracy=%.4f count=%d", nc, acc, c["total"])

    print(f"\nBigBench Strategic QA Results for {model_key}:")
    print(f"  Overall: {correct}/{total} = {accuracy:.4f}")
    for nc in sorted(by_num_choices.keys(), key=lambda x: int(x) if x.isdigit() else 99):
        c = by_num_choices[nc]
        acc = c["correct"] / c["total"] if c["total"] > 0 else 0
        print(f"  {nc}-choice: {c['correct']}/{c['total']} = {acc:.4f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adapter_label = Path(adapter_path).name if adapter_path else "base"
    experiment = "lora" if adapter_path else "zero_shot"

    result = EvalResult(
        model=model_key,
        task="bigbench_strategy",
        accuracy=accuracy,
        n_samples=total,
        experiment=experiment,
        driver_path=adapter_path or "",
    )
    save_results([result], str(RESULTS_DIR / "bigbench_strategy_results.csv"))

    detail = {
        "model": model_key,
        "adapter": adapter_path,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "by_num_choices": {k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0}
                          for k, v in by_num_choices.items()},
    }
    detail_path = RESULTS_DIR / f"bigbench_strategy_{model_key}_{adapter_label}.json"
    with open(detail_path, "w") as f:
        json.dump(detail, f, indent=2)

    return detail


def evaluate_bigbench_strategy(model_key: str, adapter_path: str | None = None,
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
            print(f"  Loaded adapter: {adapter_path}")
    try:
        detail = run_heldout(model, tokenizer, model_key,
                             adapter_path=adapter_path, max_examples=max_examples)
    finally:
        del model
        torch.cuda.empty_cache()
    return detail["accuracy"]


def main():
    parser = argparse.ArgumentParser(description="BigBench Strategic QA evaluation")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(STUDENT_MODELS.keys()) + ["all"])
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA/Q-DoRA adapter")
    parser.add_argument("--max-examples", type=int, default=500,
                        help="Max examples to evaluate (default 500)")
    parser.add_argument("--cogbench-loader", action="store_true",
                        help="Use CogBench model zoo loader (NF4 for large models)")
    args = parser.parse_args()

    if args.model == "all":
        for model_key in STUDENT_MODELS:
            evaluate_bigbench_strategy(model_key, args.adapter,
                                       args.max_examples, args.cogbench_loader)
    else:
        evaluate_bigbench_strategy(args.model, args.adapter,
                                   args.max_examples, args.cogbench_loader)


if __name__ == "__main__":
    main()
