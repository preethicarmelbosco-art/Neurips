"""Custom FinQA evaluator for financial numerical reasoning.

FinQA (Chen et al., 2021 EMNLP) tests numerical reasoning over financial
reports — questions require reading tables + text to compute answers.

Dataset: dreamerdeo/finqa on HuggingFace.
Format: question + table + context → numerical answer string.
Scoring: exact match on extracted numbers (with tolerance).

Maps to domain hypothesis: Math + Strategic + Causal reasoning.

Usage:
    python -m evaluation.scripts.eval_finqa --model llama-3-8b
    python -m evaluation.scripts.eval_finqa --model llama-3-8b --max-examples 200
"""

import argparse
import json
import logging
import re
import time
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.model_loader import load_student
from utils.metrics import EvalResult, save_results

LOG_DIR = Path(__file__).parent.parent / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("eval_finqa")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(LOG_DIR / "eval_finqa.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(_fh)

try:
    from evaluation.scripts.cogbench_eval import COGBENCH_MODELS, load_model
except ImportError:
    COGBENCH_MODELS = None

RESULTS_DIR = Path(__file__).parent.parent / "results" / "finqa"


def format_table(table_data):
    """Format a FinQA table (list of lists) into readable text."""
    if not table_data:
        return ""
    lines = []
    for row in table_data:
        lines.append(" | ".join(str(cell) for cell in row))
    return "\n".join(lines)


def extract_number(text: str) -> float | None:
    """Extract the primary numerical answer from model output."""
    # Remove % sign for comparison
    text = text.replace(",", "")
    # Find all numbers (including negative and decimal)
    numbers = re.findall(r'-?\d+\.?\d*%?', text)
    if not numbers:
        return None
    # Take the last number (usually the final answer)
    num_str = numbers[-1].rstrip("%")
    try:
        return float(num_str)
    except ValueError:
        return None


def numbers_match(pred: float, gold: float, tol: float = 0.01) -> bool:
    """Check if two numbers match within relative tolerance."""
    if gold == 0:
        return abs(pred) < tol
    return abs(pred - gold) / max(abs(gold), 1e-8) < tol


def run_heldout(model, tokenizer, model_key: str, adapter_path: str | None = None,
                max_examples: int = 200) -> dict:
    """Evaluate a live (model, tokenizer) on FinQA without load/unload."""
    print("\nLoading FinQA dataset...")
    try:
        dataset = load_dataset("dreamerdeo/finqa", split="test",
                               trust_remote_code=True)
    except Exception:
        dataset = load_dataset("ibm/finqa", split="test",
                               trust_remote_code=True)

    print(f"Loaded {len(dataset)} examples")

    if max_examples > 0 and len(dataset) > max_examples:
        dataset = dataset.select(range(max_examples))

    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total = 0

    for ex in tqdm(dataset, desc="FinQA"):
        # Build context from pre_text + table + post_text
        pre = " ".join(ex.get("pre_text", []))
        post = " ".join(ex.get("post_text", []))
        table = format_table(ex.get("table", []))
        question = ex["question"]
        gold_answer = str(ex["answer"]).strip()

        context = ""
        if pre:
            context += f"{pre}\n\n"
        if table:
            context += f"Table:\n{table}\n\n"
        if post:
            context += f"{post}\n\n"

        prompt = (
            f"Read the following financial report excerpt and answer the question "
            f"with a single number.\n\n{context}"
            f"Question: {question}\n\nAnswer:"
        )

        # Truncate prompt if too long
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=1024).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=32, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        answer_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Score
        pred_num = extract_number(answer_text)
        gold_num = extract_number(gold_answer)

        if pred_num is not None and gold_num is not None:
            if numbers_match(pred_num, gold_num):
                correct += 1
        elif answer_text.strip().lower() == gold_answer.strip().lower():
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    logger.info("FINQA_RESULT model=%s accuracy=%.4f correct=%d total=%d adapter=%s",
                model_key, accuracy, correct, total, adapter_path or "none")
    print(f"\nFinQA Results for {model_key}: {correct}/{total} = {accuracy:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adapter_label = Path(adapter_path).name if adapter_path else "base"

    result = EvalResult(
        model=model_key, task="finqa", accuracy=accuracy,
        n_samples=total, experiment="lora" if adapter_path else "zero_shot",
        driver_path=adapter_path or "",
    )
    save_results([result], str(RESULTS_DIR / "finqa_results.csv"))

    model_meta = {}
    if COGBENCH_MODELS and model_key in COGBENCH_MODELS:
        m = COGBENCH_MODELS[model_key]
        model_meta = {"size_b": m.get("size_b"), "family": m.get("family"),
                      "arch": m.get("arch"), "tier": m.get("tier")}

    detail = {"model": model_key, "adapter": adapter_path,
              "accuracy": accuracy, "correct": correct, "total": total,
              "model_metadata": model_meta,
              "hypothesis": ["Math", "Strategic", "Causal"],
              "paper_table": "Table 10 Row 5"}
    with open(RESULTS_DIR / f"finqa_{model_key}_{adapter_label}.json", "w") as f:
        json.dump(detail, f, indent=2)

    return detail


def evaluate_finqa(model_key: str, adapter_path: str | None = None,
                   max_examples: int = 200,
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
    parser = argparse.ArgumentParser(description="FinQA evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--cogbench-loader", action="store_true")
    args = parser.parse_args()
    evaluate_finqa(args.model, args.adapter, args.max_examples, args.cogbench_loader)


if __name__ == "__main__":
    main()
