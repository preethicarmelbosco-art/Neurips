"""Custom BigBench Diplomacy Recognition evaluator for strategic domain held-out benchmark.

Diplomacy Recognition (BigBench) tests models' ability to recognize deceptive vs
truthful messages in the context of the board game Diplomacy. Each example presents
a message from one player to another and asks whether it is truthful or deceptive.
This tests social-strategic reasoning -- understanding intent behind communication.

Evaluation approach: Log-likelihood binary classification. For each example, score
two completions ("truthful" / "deceptive") and pick the higher log-prob.

This is the HELD-OUT benchmark for the "strategic" domain (STR-CC corpus).

Dataset: tasksource/bigbench (diplomacy_recognition config), with fallbacks.

Usage:
    python -m evaluation.scripts.eval_bigbench_diplomacy --model llama-3-8b
    python -m evaluation.scripts.eval_bigbench_diplomacy --model llama-3-8b --adapter path/to/lora
    python -m evaluation.scripts.eval_bigbench_diplomacy --model all
"""

import argparse
import json
import logging
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
from configs.seeds import STUDENT_MODELS
from utils.model_loader import load_student
from utils.metrics import EvalResult, save_results
from _logprob_helpers import choice_logprob

LOG_DIR = Path(__file__).parent.parent / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("eval_bigbench_diplomacy")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(LOG_DIR / "eval_bigbench_diplomacy.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(_fh)

try:
    from evaluation.scripts.cogbench_eval import COGBENCH_MODELS, load_model
except ImportError:
    COGBENCH_MODELS = None

RESULTS_DIR = Path(__file__).parent.parent / "results" / "bigbench_diplomacy"

# Single-letter answer format: both "T" and "D" tokenise to a single token on
# essentially every tokeniser, and as isolated capital letters have balanced
# training-data priors — which eliminates both the tokeniser-length bias and
# the lexical-frequency bias that made the word-level ("truthful"/"deceptive")
# prompt label-collapse even after tokenisation + chat-template fixes.
PROMPT_TEMPLATE = """In the board game Diplomacy, players send messages that may be truthful or deceptive. Read the following message and answer with a single letter.

Message: {input}

Is this message truthful (T) or deceptive (D)?

Answer ("T" or "D"):"""
# Candidate → gold label mapping.
CANDIDATES = {"T": "truthful", "D": "deceptive"}


def compute_log_prob(model, tokenizer, prompt: str, completion: str, device: str) -> float:
    """Thin wrapper — delegates to shared :func:`choice_logprob`."""
    return choice_logprob(model, tokenizer, prompt, completion, device, reduce="byte_mean")


def load_diplomacy_dataset(max_examples: int = 500):
    """Load Diplomacy deception benchmark, local-first.

    BIG-bench has no `diplomacy_recognition` task (phantom reference); this eval
    uses Peskoff et al. 2020 (ACL) "It Takes Two" Diplomacy deception corpus,
    flattened to per-message JSONL with BigBench-compatible schema
    ({inputs, targets}). HF fallbacks retained in case someone re-hosts it.
    """
    # Source 1 (primary): local JSONL — works offline, guaranteed schema
    _repo_root = Path(__file__).resolve().parent.parent.parent.parent
    local_path = _repo_root / "ContrastiveData" / "data" / "bigbench_diplomacy.jsonl"
    if local_path.exists():
        examples = []
        with open(local_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        if max_examples > 0 and len(examples) > max_examples:
            examples = examples[:max_examples]
        logger.info("Loaded %d examples from local JSONL: %s", len(examples), local_path)
        return examples, str(local_path)

    # Source 2: tasksource/bigbench (not currently hosted — kept for future)
    for split in ["validation", "test", "train"]:
        try:
            ds = load_dataset("tasksource/bigbench", "diplomacy_recognition",
                              split=split, trust_remote_code=True)
            if len(ds) > 0:
                if max_examples > 0 and len(ds) > max_examples:
                    ds = ds.select(range(max_examples))
                logger.info("Loaded %d examples from tasksource/bigbench diplomacy_recognition/%s",
                            len(ds), split)
                return ds, f"tasksource/bigbench/diplomacy_recognition/{split}"
        except Exception:
            continue

    # Source 3: google/bigbench (requires `bigbench` pypi module — not on CDC)
    for split in ["validation", "test", "train"]:
        try:
            ds = load_dataset("google/bigbench", "diplomacy_recognition",
                              split=split, trust_remote_code=True)
            if len(ds) > 0:
                if max_examples > 0 and len(ds) > max_examples:
                    ds = ds.select(range(max_examples))
                logger.info("Loaded %d examples from google/bigbench diplomacy_recognition/%s",
                            len(ds), split)
                return ds, f"google/bigbench/diplomacy_recognition/{split}"
        except Exception:
            continue

    return None, None


def extract_label(example: dict) -> str | None:
    """Extract the gold label from a BigBench example.

    Handles both naming conventions (inputs/input, targets/multiple_choice_targets).
    Returns normalized label: 'truthful' or 'deceptive', or None if unparseable.
    """
    # Try targets field (list or string)
    targets = example.get("targets", example.get("target", None))
    if targets is not None:
        if isinstance(targets, list) and len(targets) > 0:
            gold = str(targets[0]).strip().lower()
        else:
            gold = str(targets).strip().lower()

        # Normalize to truthful/deceptive
        if gold in ("true", "truthful", "yes"):
            return "truthful"
        if gold in ("false", "deceptive", "no"):
            return "deceptive"

    # Try multiple_choice_scores to find the correct answer
    mc_targets = example.get("multiple_choice_targets", None)
    mc_scores = example.get("multiple_choice_scores", None)
    if mc_targets and mc_scores and isinstance(mc_targets, list) and isinstance(mc_scores, list):
        max_score = max(mc_scores)
        for i, score in enumerate(mc_scores):
            if score == max_score:
                label = str(mc_targets[i]).strip().lower()
                if label in ("true", "truthful", "yes"):
                    return "truthful"
                if label in ("false", "deceptive", "no"):
                    return "deceptive"
                break

    return None


def extract_input(example: dict) -> str:
    """Extract the input text from a BigBench example."""
    return example.get("inputs", example.get("input", ""))


def run_heldout(model, tokenizer, model_key: str, adapter_path: str | None = None,
                max_examples: int = 500) -> dict | None:
    """Evaluate a live (model, tokenizer) on BB-Diplomacy without load/unload."""
    print(f"\nLoading BigBench Diplomacy Recognition dataset...")
    dataset, source = load_diplomacy_dataset(max_examples)
    if dataset is None:
        logger.error("Could not load BigBench diplomacy_recognition from any source.")
        print("ERROR: Could not load BigBench diplomacy_recognition dataset.")
        return None

    n_examples = len(dataset)
    print(f"Loaded {n_examples} examples from {source}")
    logger.info("Dataset loaded: %d examples from %s", n_examples, source)

    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total = 0
    by_label = {
        "truthful": {"correct": 0, "total": 0},
        "deceptive": {"correct": 0, "total": 0},
    }

    for ex in tqdm(dataset, desc="Diplomacy Recognition eval"):
        if isinstance(ex, dict):
            input_text = extract_input(ex)
            gold_label = extract_label(ex)
        else:
            # HuggingFace dataset row
            input_text = extract_input(ex)
            gold_label = extract_label(ex)

        if not input_text or gold_label is None:
            continue

        user_prompt = PROMPT_TEMPLATE.format(input=input_text)
        # Helper handles chat-template + format-string contract internally.
        # Pass the bare candidate letter ("T" / "D"); both tokenise to a single
        # token on every tokeniser and have near-equal training-data priors.
        score_T = choice_logprob(model, tokenizer, user_prompt, "T", device, reduce="byte_mean")
        score_D = choice_logprob(model, tokenizer, user_prompt, "D", device, reduce="byte_mean")
        predicted = CANDIDATES["T"] if score_T > score_D else CANDIDATES["D"]
        is_correct = predicted == gold_label

        if is_correct:
            correct += 1
            by_label[gold_label]["correct"] += 1
        by_label[gold_label]["total"] += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"\nBigBench Diplomacy Recognition Results for {model_key}:")
    print(f"  Overall: {correct}/{total} = {accuracy:.4f}")
    for label, counts in by_label.items():
        t = counts["total"]
        c = counts["correct"]
        print(f"  {label}: {c}/{t} = {c / t:.4f}" if t > 0 else f"  {label}: N/A")

    logger.info("BIGBENCH_DIPLOMACY_RESULT model=%s accuracy=%.4f correct=%d total=%d "
                "adapter=%s by_label=%s",
                model_key, accuracy, correct, total, adapter_path or "none",
                json.dumps({k: v.get("accuracy", v["correct"] / v["total"] if v["total"] > 0 else 0)
                            for k, v in by_label.items()}))

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adapter_label = Path(adapter_path).name if adapter_path else "base"
    experiment = "lora" if adapter_path else "zero_shot"

    result = EvalResult(
        model=model_key,
        task="bigbench_diplomacy",
        accuracy=accuracy,
        n_samples=total,
        experiment=experiment,
        driver_path=adapter_path or "",
    )
    save_results([result], str(RESULTS_DIR / "bigbench_diplomacy_results.csv"))

    detail = {
        "model": model_key,
        "adapter": adapter_path,
        "overall_accuracy": accuracy,
        "total": total,
        "correct": correct,
        "by_label": {
            k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0}
            for k, v in by_label.items()
        },
    }
    detail_path = RESULTS_DIR / f"bigbench_diplomacy_{model_key}_{adapter_label}.json"
    with open(detail_path, "w") as f:
        json.dump(detail, f, indent=2)

    return detail


def evaluate_bigbench_diplomacy(model_key: str, adapter_path: str | None = None,
                                max_examples: int = 500,
                                use_cogbench_loader: bool = False):
    """Standalone entrypoint: load model, run heldout, unload."""
    print(f"Loading model: {model_key}...")
    if use_cogbench_loader and COGBENCH_MODELS and model_key in COGBENCH_MODELS:
        model, tokenizer = load_model(model_key)
    else:
        model, tokenizer, _ = load_student(model_key)
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            print(f"Loaded adapter: {adapter_path}")
    try:
        detail = run_heldout(model, tokenizer, model_key,
                             adapter_path=adapter_path, max_examples=max_examples)
    finally:
        del model
        torch.cuda.empty_cache()
    return detail["overall_accuracy"] if detail else None


def main():
    parser = argparse.ArgumentParser(description="BigBench Diplomacy Recognition evaluation")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(STUDENT_MODELS.keys()) + ["all"])
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA/Q-DoRA adapter")
    parser.add_argument("--max-examples", type=int, default=500,
                        help="Maximum number of examples to evaluate (default: 500)")
    parser.add_argument("--cogbench-loader", action="store_true",
                        help="Use CogBench model loader instead of default")
    args = parser.parse_args()

    if args.model == "all":
        for model_key in STUDENT_MODELS:
            evaluate_bigbench_diplomacy(model_key, args.adapter, args.max_examples,
                                        args.cogbench_loader)
    else:
        evaluate_bigbench_diplomacy(args.model, args.adapter, args.max_examples,
                                    args.cogbench_loader)


if __name__ == "__main__":
    main()
