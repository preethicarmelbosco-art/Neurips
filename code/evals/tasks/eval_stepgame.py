"""Custom StepGame evaluator for spatial reasoning benchmark.

StepGame (Shi et al., 2022) tests spatial reasoning by requiring models to
track entity positions through a sequence of directional steps.

Each example has a story describing relative spatial movements and a question
asking about the final spatial relationship between two entities.

Dataset: bdu-t/stepgame on HuggingFace (multiple difficulty levels: k=1..10 steps)

This benchmark maps to the Spatial primitive in CogBench (Table 7, Table 12).

Usage:
    python -m evaluation.scripts.eval_stepgame --model llama-3-8b
    python -m evaluation.scripts.eval_stepgame --model llama-3-8b --adapter path/to/lora
    python -m evaluation.scripts.eval_stepgame --model llama-3-8b --max-steps 5
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
logger = logging.getLogger("eval_stepgame")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(LOG_DIR / "eval_stepgame.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(_fh)

try:
    from evaluation.scripts.cogbench_eval import COGBENCH_MODELS, load_model
except ImportError:
    COGBENCH_MODELS = None

RESULTS_DIR = Path(__file__).parent.parent / "results" / "stepgame"

# StepGame answer labels (spatial relations)
SPATIAL_LABELS = [
    "left", "right", "above", "below",
    "upper-left", "upper-right", "lower-left", "lower-right",
    "overlap",
]


def compute_log_prob(model, tokenizer, prompt: str, completion: str, device: str) -> float:
    """Compute length-normalized log probability of completion given prompt."""
    full_text = prompt + " " + completion
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

    prompt_len = prompt_ids.shape[1]
    completion_len = full_ids.shape[1] - prompt_len
    if completion_len <= 0:
        return float("-inf")

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits

    shift_logits = logits[:, prompt_len - 1:-1, :]
    shift_labels = full_ids[:, prompt_len:]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.sum().item() / completion_len


def run_heldout(model, tokenizer, model_key: str, adapter_path: str | None = None,
                max_steps: int = 5, max_examples: int = 500) -> dict:
    """Evaluate a live (model, tokenizer) on StepGame without load/unload."""
    print(f"\nLoading StepGame dataset...")
    try:
        dataset = load_dataset("bdu-t/stepgame", split="test",
                               trust_remote_code=True)
    except Exception:
        try:
            dataset = load_dataset("tasksource/stepgame", split="test",
                                   trust_remote_code=True)
        except Exception:
            print("  Could not load StepGame from HuggingFace.")
            print("  Attempting local data path...")
            local_path = Path(__file__).parent.parent.parent.parent / "ContrastiveData" / "data" / "stepgame.jsonl"
            if local_path.exists():
                from evaluation.scripts.cogbench_eval import load_jsonl
                dataset = load_jsonl(str(local_path))
            else:
                print(f"  Not found at {local_path}")
                return {"accuracy": 0.0, "correct": 0, "total": 0, "by_steps": {}}

    print(f"Loaded {len(dataset)} examples")

    # Filter by max_steps if the dataset has a step count field
    if hasattr(dataset, 'filter'):
        try:
            dataset = dataset.filter(lambda x: x.get("k", x.get("n_steps", 1)) <= max_steps)
            print(f"After filtering to k≤{max_steps}: {len(dataset)} examples")
        except Exception:
            pass

    if len(dataset) > max_examples:
        if hasattr(dataset, 'select'):
            dataset = dataset.select(range(max_examples))
        else:
            dataset = dataset[:max_examples]

    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total = 0
    by_steps = {}  # accuracy breakdown by step count

    for example in tqdm(dataset, desc="StepGame"):
        story = example.get("story", example.get("text", example.get("context", "")))
        question = example.get("question", "")
        label = example.get("label", example.get("answer", "")).strip().lower()
        n_steps = example.get("k", example.get("n_steps", "?"))

        if not story or not label:
            continue

        if question:
            prompt = f"Read the following story about spatial positions and answer the question.\n\nStory: {story}\n\nQuestion: {question}\n\nThe spatial relation is:"
        else:
            prompt = f"Read the following story about spatial positions. What is the final spatial relation between the queried entities?\n\nStory: {story}\n\nThe spatial relation is:"

        # Score each possible spatial label
        best_label = None
        best_score = float("-inf")
        for candidate in SPATIAL_LABELS:
            score = compute_log_prob(model, tokenizer, prompt, f" {candidate}", device)
            if score > best_score:
                best_score = score
                best_label = candidate

        is_correct = best_label == label
        if is_correct:
            correct += 1
        total += 1

        # Track by step count
        step_key = str(n_steps)
        if step_key not in by_steps:
            by_steps[step_key] = {"correct": 0, "total": 0}
        by_steps[step_key]["total"] += 1
        if is_correct:
            by_steps[step_key]["correct"] += 1

    accuracy = correct / total if total > 0 else 0

    logger.info("STEPGAME_RESULT model=%s accuracy=%.4f correct=%d total=%d adapter=%s",
                model_key, accuracy, correct, total, adapter_path or "none")
    for k in sorted(by_steps.keys(), key=lambda x: int(x) if x.isdigit() else 99):
        c = by_steps[k]
        acc = c["correct"] / c["total"] if c["total"] > 0 else 0
        logger.info("  STEPGAME_BYSTEP k=%s accuracy=%.4f n=%d", k, acc, c["total"])

    print(f"\nStepGame Results for {model_key}:")
    print(f"  Overall: {correct}/{total} = {accuracy:.4f}")
    for k in sorted(by_steps.keys(), key=lambda x: int(x) if x.isdigit() else 99):
        c = by_steps[k]
        acc = c["correct"] / c["total"] if c["total"] > 0 else 0
        print(f"  k={k}: {c['correct']}/{c['total']} = {acc:.4f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adapter_label = Path(adapter_path).name if adapter_path else "base"
    experiment = "lora" if adapter_path else "zero_shot"

    result = EvalResult(
        model=model_key,
        task="stepgame",
        accuracy=accuracy,
        n_samples=total,
        experiment=experiment,
        driver_path=adapter_path or "",
    )
    save_results([result], str(RESULTS_DIR / "stepgame_results.csv"))

    detail = {
        "model": model_key,
        "adapter": adapter_path,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "by_steps": {k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0}
                     for k, v in by_steps.items()},
    }
    detail_path = RESULTS_DIR / f"stepgame_{model_key}_{adapter_label}.json"
    with open(detail_path, "w") as f:
        json.dump(detail, f, indent=2)

    return detail


def evaluate_stepgame(model_key: str, adapter_path: str | None = None,
                      max_steps: int = 5, max_examples: int = 500,
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
                             adapter_path=adapter_path, max_steps=max_steps,
                             max_examples=max_examples)
    finally:
        del model
        torch.cuda.empty_cache()
    return detail["accuracy"]


def main():
    parser = argparse.ArgumentParser(description="StepGame evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=5,
                        help="Max step difficulty (1-10, default 5)")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--cogbench-loader", action="store_true")
    args = parser.parse_args()

    evaluate_stepgame(args.model, args.adapter, args.max_steps,
                      args.max_examples, args.cogbench_loader)


if __name__ == "__main__":
    main()
