"""Custom SpartQA evaluator for spatial reasoning benchmark (held-out).

SpartQA (Mirzaee et al., 2021) tests spatial reasoning with natural language
descriptions of spatial arrangements and questions about spatial relationships
between objects. More complex than StepGame, serving as the held-out benchmark
for the spatial domain (STP-CC corpus) in the CogBench evaluation pipeline.

Question types:
  FR (Find Relation) — "What is the relation between X and Y?"
  FB (Find Block)    — "What is to the left of X?"
  CO (Choose Object) — Multiple choice about spatial arrangement
  YN (Yes/No)        — "Is X to the left of Y?"

Dataset: tasksource/spartqa → allenai/spartqa → local JSONL fallback

This benchmark maps to the Spatial primitive (held-out) in CogBench (Table 7, Table 12).

Usage:
    python -m evaluation.scripts.eval_spartqa --model llama-3-8b
    python -m evaluation.scripts.eval_spartqa --model llama-3-8b --adapter path/to/lora
    python -m evaluation.scripts.eval_spartqa --model llama-3-8b --max-examples 200
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
logger = logging.getLogger("eval_spartqa")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(LOG_DIR / "eval_spartqa.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(_fh)

try:
    from evaluation.scripts.cogbench_eval import COGBENCH_MODELS, load_model
except ImportError:
    COGBENCH_MODELS = None

RESULTS_DIR = Path(__file__).parent.parent / "results" / "spartqa"

# SpartQA answer labels by question type
SPATIAL_LABELS = [
    "left", "right", "above", "below",
    "behind", "in front of", "near", "far",
    "inside", "outside", "touching", "overlap",
]
YN_LABELS = ["Yes", "No"]


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


def _get_question_type(example: dict) -> str:
    """Extract question type from a SpartQA example."""
    qtype = example.get("type", example.get("question_type", ""))
    if isinstance(qtype, str):
        qtype = qtype.strip().upper()
    # Normalize known types
    if qtype in ("FR", "FIND_RELATION", "FIND RELATION"):
        return "FR"
    if qtype in ("FB", "FIND_BLOCK", "FIND BLOCK"):
        return "FB"
    if qtype in ("CO", "CHOOSE_OBJECT", "CHOOSE OBJECT", "CHOOSE"):
        return "CO"
    if qtype in ("YN", "YES_NO", "YES NO", "YESNO"):
        return "YN"
    return qtype if qtype else "unknown"


def _get_candidates(example: dict, question_type: str) -> list[str]:
    """Determine the candidate answer set for an example."""
    # If explicit choices are provided, use them
    choices = example.get("choices", example.get("candidates", None))
    if choices and isinstance(choices, list) and len(choices) > 0:
        return [str(c).strip() for c in choices]

    # Otherwise fall back based on question type
    if question_type == "YN":
        return list(YN_LABELS)
    if question_type == "FR":
        return list(SPATIAL_LABELS)

    # For FB, CO, or unknown types without explicit choices, use spatial labels
    return list(SPATIAL_LABELS)


def evaluate_spartqa(model_key: str, adapter_path: str | None = None,
                     max_examples: int = 500,
                     use_cogbench_loader: bool = False):
    """Evaluate a model on SpartQA spatial reasoning.

    Args:
        model_key: Model identifier.
        adapter_path: Optional PEFT adapter path.
        max_examples: Max examples to evaluate (default 500).
        use_cogbench_loader: Use NF4 quantization for large models.
    """
    print(f"\nLoading SpartQA dataset...")
    dataset = None

    # Try HuggingFace sources in order
    for hf_name in ("tasksource/spartqa", "allenai/spartqa"):
        if dataset is not None:
            break
        for split in ("test", "validation"):
            try:
                dataset = load_dataset(hf_name, split=split,
                                       trust_remote_code=True)
                print(f"  Loaded {len(dataset)} examples from {hf_name} ({split})")
                break
            except Exception:
                continue

    # Fallback to local JSONL
    if dataset is None:
        local_path = (Path(__file__).parent.parent.parent.parent
                      / "ContrastiveData" / "data" / "spartqa.jsonl")
        if local_path.exists():
            print(f"  Loading from local path: {local_path}")
            try:
                from evaluation.scripts.cogbench_eval import load_jsonl
                dataset = load_jsonl(str(local_path))
            except ImportError:
                # Manual JSONL loading
                with open(local_path) as f:
                    dataset = [json.loads(line) for line in f if line.strip()]
            print(f"  Loaded {len(dataset)} examples from local JSONL")
        else:
            print(f"  Could not load SpartQA from HuggingFace or local path.")
            print(f"  Not found at {local_path}")
            return 0.0

    print(f"Loaded {len(dataset)} examples total")

    if len(dataset) > max_examples:
        if hasattr(dataset, 'select'):
            dataset = dataset.select(range(max_examples))
        else:
            dataset = dataset[:max_examples]

    # Load model
    if use_cogbench_loader and COGBENCH_MODELS and model_key in COGBENCH_MODELS:
        model, tokenizer = load_model(model_key)
    else:
        model, tokenizer, _ = load_student(model_key)
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total = 0
    by_question_type = {}  # accuracy breakdown by question type

    for example in tqdm(dataset, desc="SpartQA"):
        story = example.get("story", example.get("context",
                    example.get("passage", example.get("text", ""))))
        question = example.get("question", example.get("query", ""))
        label = example.get("answer", example.get("label",
                    example.get("targets", ""))).strip()
        qtype = _get_question_type(example)

        if not story or not label:
            continue

        # Build prompt
        prompt = (
            "Read the following description of spatial arrangements "
            "and answer the question.\n\n"
            f"Description: {story}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        # Determine candidates and score each
        candidates = _get_candidates(example, qtype)
        best_label = None
        best_score = float("-inf")
        for candidate in candidates:
            score = compute_log_prob(model, tokenizer, prompt,
                                     f" {candidate}", device)
            if score > best_score:
                best_score = score
                best_label = candidate

        is_correct = (best_label is not None
                      and best_label.strip().lower() == label.strip().lower())
        if is_correct:
            correct += 1
        total += 1

        # Track by question type
        if qtype not in by_question_type:
            by_question_type[qtype] = {"correct": 0, "total": 0}
        by_question_type[qtype]["total"] += 1
        if is_correct:
            by_question_type[qtype]["correct"] += 1

    accuracy = correct / total if total > 0 else 0

    logger.info("SPARTQA_RESULT model=%s accuracy=%.4f correct=%d total=%d adapter=%s",
                model_key, accuracy, correct, total, adapter_path or "none")
    for qt in sorted(by_question_type.keys()):
        c = by_question_type[qt]
        acc = c["correct"] / c["total"] if c["total"] > 0 else 0
        logger.info("  SPARTQA_BYTYPE type=%s accuracy=%.4f n=%d", qt, acc, c["total"])

    print(f"\nSpartQA Results for {model_key}:")
    print(f"  Overall: {correct}/{total} = {accuracy:.4f}")
    for qt in sorted(by_question_type.keys()):
        c = by_question_type[qt]
        acc = c["correct"] / c["total"] if c["total"] > 0 else 0
        print(f"  {qt}: {c['correct']}/{c['total']} = {acc:.4f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adapter_label = Path(adapter_path).name if adapter_path else "base"
    experiment = "lora" if adapter_path else "zero_shot"

    result = EvalResult(
        model=model_key,
        task="spartqa",
        accuracy=accuracy,
        n_samples=total,
        experiment=experiment,
        driver_path=adapter_path or "",
    )
    save_results([result], str(RESULTS_DIR / "spartqa_results.csv"))

    detail = {
        "model": model_key,
        "adapter": adapter_path,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "by_question_type": {
            qt: {**v, "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0}
            for qt, v in by_question_type.items()
        },
    }
    detail_path = RESULTS_DIR / f"spartqa_{model_key}_{adapter_label}.json"
    with open(detail_path, "w") as f:
        json.dump(detail, f, indent=2)

    del model
    torch.cuda.empty_cache()
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="SpartQA evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--cogbench-loader", action="store_true")
    args = parser.parse_args()

    evaluate_spartqa(args.model, args.adapter, args.max_examples,
                     args.cogbench_loader)


if __name__ == "__main__":
    main()
