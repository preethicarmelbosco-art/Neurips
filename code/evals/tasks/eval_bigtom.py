"""Custom BigToM evaluator for Theory of Mind held-out benchmark.

BigToM (Gandhi et al., 2024) tests forward and backward belief inference.
Each story has a true-belief (TB) and false-belief (FB) condition.
The model must correctly identify what a character believes given the story.

Evaluation approach: Multiple-choice log-likelihood scoring.
For each story, we construct a prompt asking about the character's belief
and score two completions: the true-belief answer and the false-belief answer.
Accuracy = fraction where the model assigns higher log-prob to the correct answer.

Dataset: ptsv/bigtom_train on HuggingFace (800 stories, 400 TB + 400 FB)

Usage:
    python -m evaluation.scripts.eval_bigtom --model llama-3-8b
    python -m evaluation.scripts.eval_bigtom --model llama-3-8b --adapter path/to/lora
    python -m evaluation.scripts.eval_bigtom --model all
"""

import argparse
import json
import re
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

RESULTS_DIR = Path(__file__).parent.parent / "results" / "bigtom"

# BigToM prompt template
# Stories end with a belief statement. We ask the model to judge if the belief
# matches the character's expected mental state given the story events.
PROMPT_TEMPLATE = """Read the following story and answer the question.

Story: {story}

Question: Based on the events in the story, is the character's belief at the end consistent with what they should believe given what they experienced?

Answer (Yes or No):"""


def extract_belief_statement(text: str) -> tuple[str, str]:
    """Split a BigToM story into narrative + final belief statement."""
    sentences = text.strip().split(". ")
    if len(sentences) >= 2:
        belief = sentences[-1]
        narrative = ". ".join(sentences[:-1]) + "."
        return narrative, belief
    return text, ""


def compute_log_prob(model, tokenizer, prompt: str, completion: str, device: str) -> float:
    """Mean per-token log P(completion | prompt).

    Thin wrapper around the shared :func:`choice_logprob` helper, which
    tokenises prompt and continuation separately (avoiding BPE-boundary
    misalignment), strips BOS from the continuation, and applies the
    tokenizer's chat template for instruct-tuned models.
    """
    return choice_logprob(model, tokenizer, prompt, completion, device, reduce="byte_mean")


def run_heldout(model, tokenizer, model_key: str, adapter_path: str | None = None) -> dict:
    """Evaluate a live (model, tokenizer) on BigToM without loading/unloading.

    Caller owns model lifecycle. Does NOT call del/empty_cache at end.
    Returns dict with accuracy, total, correct, by_type.
    """
    print(f"\nLoading BigToM dataset...")
    dataset = load_dataset("ptsv/bigtom_train", split="train")
    print(f"Loaded {len(dataset)} stories")

    model.eval()
    device = next(model.parameters()).device

    # Group stories by ID to pair TB/FB conditions
    stories_by_id = {}
    for ex in dataset:
        sid = ex["id"]
        if sid not in stories_by_id:
            stories_by_id[sid] = {}
        stories_by_id[sid][ex["condition"]] = ex["text"]
        stories_by_id[sid]["type"] = ex["type"]

    correct = 0
    total = 0
    results_by_type = {"forward": {"correct": 0, "total": 0},
                       "backward": {"correct": 0, "total": 0}}

    for sid, stories in tqdm(stories_by_id.items(), desc="BigToM eval"):
        if "text_tb" not in stories or "text_fb" not in stories:
            continue

        story_type = stories["type"]
        tb_text = stories["text_tb"]
        fb_text = stories["text_fb"]

        # For each condition, construct prompt and score Yes/No
        # TB condition: belief IS consistent -> correct answer is "Yes"
        tb_prompt = PROMPT_TEMPLATE.format(story=tb_text)
        tb_yes = compute_log_prob(model, tokenizer, tb_prompt, "Yes", device)
        tb_no = compute_log_prob(model, tokenizer, tb_prompt, "No", device)
        tb_correct = tb_yes > tb_no

        # FB condition: belief is NOT consistent -> correct answer is "No"
        fb_prompt = PROMPT_TEMPLATE.format(story=fb_text)
        fb_yes = compute_log_prob(model, tokenizer, fb_prompt, "Yes", device)
        fb_no = compute_log_prob(model, tokenizer, fb_prompt, "No", device)
        fb_correct = fb_no > fb_yes

        if tb_correct:
            correct += 1
            results_by_type[story_type]["correct"] += 1
        total += 1
        results_by_type[story_type]["total"] += 1

        if fb_correct:
            correct += 1
            results_by_type[story_type]["correct"] += 1
        total += 1
        results_by_type[story_type]["total"] += 1

    accuracy = correct / total if total > 0 else 0
    print(f"\nBigToM Results for {model_key}:")
    print(f"  Overall: {correct}/{total} = {accuracy:.4f}")
    for stype, counts in results_by_type.items():
        t = counts["total"]
        c = counts["correct"]
        print(f"  {stype}: {c}/{t} = {c / t:.4f}" if t > 0 else f"  {stype}: N/A")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adapter_label = Path(adapter_path).name if adapter_path else "base"
    experiment = "lora" if adapter_path else "zero_shot"

    result = EvalResult(
        model=model_key,
        task="bigtom",
        accuracy=accuracy,
        n_samples=total,
        experiment=experiment,
        driver_path=adapter_path or "",
    )
    save_results([result], str(RESULTS_DIR / "bigtom_results.csv"))

    detail = {
        "model": model_key,
        "adapter": adapter_path,
        "overall_accuracy": accuracy,
        "total": total,
        "correct": correct,
        "by_type": {k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0}
                    for k, v in results_by_type.items()},
    }
    detail_path = RESULTS_DIR / f"bigtom_{model_key}_{adapter_label}.json"
    with open(detail_path, "w") as f:
        json.dump(detail, f, indent=2)

    return detail


def evaluate_bigtom(model_key: str, adapter_path: str | None = None):
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
    parser = argparse.ArgumentParser(description="BigToM evaluation")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(STUDENT_MODELS.keys()) + ["all"])
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA/Q-DoRA adapter")
    args = parser.parse_args()

    if args.model == "all":
        for model_key in STUDENT_MODELS:
            evaluate_bigtom(model_key, args.adapter)
    else:
        evaluate_bigtom(args.model, args.adapter)


if __name__ == "__main__":
    main()
