"""Custom CyberMetric evaluator using WMDP-Cyber benchmark.

WMDP (Weapons of Mass Destruction Proxy) cybersecurity subset tests
technical security knowledge via 4-choice multiple-choice questions.

Dataset: cais/wmdp (config: wmdp-cyber), 1,987 test questions.
Scoring: log-likelihood MC accuracy.

Maps to domain hypothesis: Strategic + Causal + ToM reasoning.

Usage:
    python -m evaluation.scripts.eval_cybermetric --model llama-3-8b
    python -m evaluation.scripts.eval_cybermetric --model llama-3-8b --cogbench-loader
"""

import argparse
import json
import logging
import os
import sys
import time
# cais/wmdp has a loading script that internally calls load_dataset without
# forwarding trust_remote_code=True, so we must set the env var before the
# datasets package is imported.
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from _logprob_helpers import choice_logprob
from utils.model_loader import load_student

LOG_DIR = Path(__file__).parent.parent / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("eval_cybermetric")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(LOG_DIR / "eval_cybermetric.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(_fh)
from utils.metrics import EvalResult, save_results

try:
    from evaluation.scripts.cogbench_eval import COGBENCH_MODELS, load_model
except ImportError:
    COGBENCH_MODELS = None

RESULTS_DIR = Path(__file__).parent.parent / "results" / "cybermetric"


def compute_choice_log_prob(model, tokenizer, prompt, choice_text, device):
    """Mean per-token log P(completion | prompt).

    Thin wrapper around shared :func:`choice_logprob` — tokenises prompt and
    continuation separately, strips BOS, applies chat template for instruct
    models, and handles the space-prefix contract via the helper.
    """
    return choice_logprob(model, tokenizer, prompt, choice_text, device, reduce="byte_mean")


def run_heldout(model, tokenizer, model_key: str, adapter_path: str | None = None,
                max_examples: int = 0) -> dict:
    """Evaluate a live (model, tokenizer) on WMDP-Cyber without load/unload."""
    print("\nLoading WMDP-Cyber dataset...")
    dataset = load_dataset("cais/wmdp", "wmdp-cyber", split="test",
                           trust_remote_code=True)
    print(f"Loaded {len(dataset)} questions")

    if max_examples > 0 and len(dataset) > max_examples:
        dataset = dataset.select(range(max_examples))

    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total = 0

    for ex in tqdm(dataset, desc="WMDP-Cyber"):
        question = ex["question"]
        choices = ex["choices"]
        gold = ex["answer"]  # int 0-3

        prompt = f"Question: {question}\n\nChoices:\n"
        for i, c in enumerate(choices):
            prompt += f"  ({chr(65+i)}) {c}\n"
        prompt += "\nAnswer with a single letter:"

        # Score just the letter — standard MCQA practice (matches lm-eval).
        # Scoring ``f" ({letter}) {choice_text}"`` mixes answer selection with
        # the prior probability of the choice text, biasing toward shorter /
        # higher-prior answer strings independent of correctness.
        best_idx = 0
        best_score = float("-inf")
        for i in range(len(choices)):
            letter = chr(65 + i)
            score = compute_choice_log_prob(model, tokenizer, prompt, letter, device)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx == gold:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0

    logger.info("CYBERMETRIC_RESULT model=%s accuracy=%.4f correct=%d total=%d adapter=%s",
                model_key, accuracy, correct, total, adapter_path or "none")
    print(f"\nWMDP-Cyber Results for {model_key}: {correct}/{total} = {accuracy:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adapter_label = Path(adapter_path).name if adapter_path else "base"

    result = EvalResult(
        model=model_key, task="cybermetric", accuracy=accuracy,
        n_samples=total, experiment="lora" if adapter_path else "zero_shot",
        driver_path=adapter_path or "",
    )
    save_results([result], str(RESULTS_DIR / "cybermetric_results.csv"))

    # Model metadata for Tables 8-9 (scaling/architecture analysis)
    model_meta = {}
    if COGBENCH_MODELS and model_key in COGBENCH_MODELS:
        m = COGBENCH_MODELS[model_key]
        model_meta = {"size_b": m.get("size_b"), "family": m.get("family"),
                      "arch": m.get("arch"), "tier": m.get("tier")}

    detail = {"model": model_key, "adapter": adapter_path,
              "accuracy": accuracy, "correct": correct, "total": total,
              "model_metadata": model_meta,
              "hypothesis": ["Strategic", "Causal", "ToM"],
              "paper_table": "Table 10 Row 1"}
    with open(RESULTS_DIR / f"cybermetric_{model_key}_{adapter_label}.json", "w") as f:
        json.dump(detail, f, indent=2)

    return detail


def evaluate_cybermetric(model_key: str, adapter_path: str | None = None,
                         max_examples: int = 0,
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
    parser = argparse.ArgumentParser(description="CyberMetric (WMDP-Cyber) evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--cogbench-loader", action="store_true")
    args = parser.parse_args()
    evaluate_cybermetric(args.model, args.adapter, args.max_examples, args.cogbench_loader)


if __name__ == "__main__":
    main()
