"""Custom Balanced COPA (BCOPA-CE) evaluator for causal held-out benchmark.

Balanced COPA (Kavumba et al., 2019) reduces superficial cue exploitation
in the original COPA by mirroring premises to create balanced answer distributions.

Evaluation: log-likelihood scoring of two choices given a premise+question.
Dataset: pkavumba/balanced-copa on HuggingFace (500 test examples)

Usage:
    python -m evaluation.scripts.eval_balanced_copa --model llama-3-8b
    python -m evaluation.scripts.eval_balanced_copa --model llama-3-8b --adapter path/to/lora
    python -m evaluation.scripts.eval_balanced_copa --model all
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.seeds import STUDENT_MODELS
from utils.model_loader import load_student
from utils.metrics import EvalResult, save_results

RESULTS_DIR = Path(__file__).parent.parent / "results" / "balanced_copa"


def build_copa_prompt(premise: str, question: str, choice: str) -> str:
    """Build a COPA-style prompt for log-likelihood scoring."""
    if question == "cause":
        return f"{choice} so {premise.rstrip('.')}"
    else:  # effect
        return f"{premise.rstrip('.')} so {choice}"


def score_sequence(model, tokenizer, text: str, device: str) -> float:
    """Compute mean log probability of a sequence."""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model(ids)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = ids[:, 1:]
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mean log prob (length-normalized)
    return token_log_probs.mean().item()


def run_heldout(model, tokenizer, model_key: str, adapter_path: str | None = None) -> dict:
    """Evaluate a live (model, tokenizer) on Balanced COPA without loading/unloading."""
    print(f"\nLoading Balanced COPA dataset...")
    dataset = load_dataset("pkavumba/balanced-copa", split="test")
    print(f"Loaded {len(dataset)} test examples")

    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total = 0
    correct_mirrored = 0
    total_mirrored = 0
    correct_original = 0
    total_original = 0

    for ex in tqdm(dataset, desc="Balanced COPA"):
        premise = ex["premise"]
        question = ex["question"]
        choice1 = ex["choice1"]
        choice2 = ex["choice2"]
        label = ex["label"]
        mirrored = ex["mirrored"]

        text1 = build_copa_prompt(premise, question, choice1)
        text2 = build_copa_prompt(premise, question, choice2)

        score1 = score_sequence(model, tokenizer, text1, device)
        score2 = score_sequence(model, tokenizer, text2, device)

        pred = 0 if score1 > score2 else 1
        is_correct = pred == label

        if is_correct:
            correct += 1
        total += 1

        if mirrored:
            total_mirrored += 1
            if is_correct:
                correct_mirrored += 1
        else:
            total_original += 1
            if is_correct:
                correct_original += 1

    accuracy = correct / total if total > 0 else 0
    acc_mirror = correct_mirrored / total_mirrored if total_mirrored > 0 else 0
    acc_orig = correct_original / total_original if total_original > 0 else 0

    print(f"\nBalanced COPA Results for {model_key}:")
    print(f"  Overall:  {correct}/{total} = {accuracy:.4f}")
    print(f"  Original: {correct_original}/{total_original} = {acc_orig:.4f}")
    print(f"  Mirrored: {correct_mirrored}/{total_mirrored} = {acc_mirror:.4f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adapter_label = Path(adapter_path).name if adapter_path else "base"
    experiment = "lora" if adapter_path else "zero_shot"

    result = EvalResult(
        model=model_key,
        task="balanced_copa",
        accuracy=accuracy,
        n_samples=total,
        experiment=experiment,
        driver_path=adapter_path or "",
    )
    save_results([result], str(RESULTS_DIR / "balanced_copa_results.csv"))

    detail = {
        "model": model_key,
        "adapter": adapter_path,
        "overall_accuracy": accuracy,
        "total": total,
        "original_accuracy": acc_orig,
        "mirrored_accuracy": acc_mirror,
        "original_total": total_original,
        "mirrored_total": total_mirrored,
    }
    detail_path = RESULTS_DIR / f"bcopa_{model_key}_{adapter_label}.json"
    with open(detail_path, "w") as f:
        json.dump(detail, f, indent=2)

    return detail


def evaluate_balanced_copa(model_key: str, adapter_path: str | None = None):
    """Standalone entrypoint: load model, run heldout, unload."""
    print(f"Loading model: {model_key}...")
    model, tokenizer, cfg = load_student(model_key)

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"Loaded adapter: {adapter_path}")

    try:
        detail = run_heldout(model, tokenizer, model_key, adapter_path=adapter_path)
    finally:
        del model
        torch.cuda.empty_cache()
    return detail["overall_accuracy"]


def main():
    parser = argparse.ArgumentParser(description="Balanced COPA evaluation")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(STUDENT_MODELS.keys()) + ["all"])
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA/Q-DoRA adapter")
    args = parser.parse_args()

    if args.model == "all":
        for model_key in STUDENT_MODELS:
            evaluate_balanced_copa(model_key, args.adapter)
    else:
        evaluate_balanced_copa(args.model, args.adapter)


if __name__ == "__main__":
    main()
