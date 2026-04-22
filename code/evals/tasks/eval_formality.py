"""Custom formality classification evaluator for NULL-CC style benchmark.

Tests whether a model can distinguish formal academic prose from informal
conversational style. This is the validation benchmark for the NULL-CC
negative control corpus — verifying the adapter learned style even though
that style signal should NOT transfer to cognitive domain benchmarks
(β_null ≈ 0 in composition regression Table 11).

Evaluation approach: Log-likelihood binary classification. For each pair,
construct a prompt with the text and score completions "formal" vs "informal".
Accuracy = fraction where the model assigns higher log-prob to the correct label.

Additionally reports a "style gap" metric: mean log-prob difference between
correct and incorrect labels (larger gap = stronger style sensitivity).

Data source: NULL-CC holdout pairs from ContrastiveData/data/holdout/null_cc_holdout.jsonl
Each pair has target_formal (formal text) and retain_informal (informal text).

Usage:
    python -m evaluation.scripts.eval_formality --model llama-3-8b
    python -m evaluation.scripts.eval_formality --model llama-3-8b --adapter path/to/lora
    python -m evaluation.scripts.eval_formality --model all
"""

import argparse
import json
import logging
import torch
from pathlib import Path
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.seeds import STUDENT_MODELS, DOMAINS
from utils.model_loader import load_student
from utils.metrics import EvalResult, save_results

LOG_DIR = Path(__file__).parent.parent / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("eval_formality")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(LOG_DIR / "eval_formality.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(_fh)

try:
    from evaluation.scripts.cogbench_eval import COGBENCH_MODELS, load_model
except ImportError:
    COGBENCH_MODELS = None

RESULTS_DIR = Path(__file__).parent.parent / "results" / "formality"

PROMPT_TEMPLATE = """Read the following text and determine whether it is written in a formal academic style or an informal conversational style.

Text: {text}

This text is written in a style that is:"""


def compute_log_prob(model, tokenizer, prompt: str, completion: str, device: str) -> float:
    """Compute log probability of completion given prompt."""
    full_text = prompt + " " + completion
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits

    shift_logits = logits[:, prompt_len - 1:-1, :]
    shift_labels = full_ids[:, prompt_len:]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.mean().item()


def load_null_cc_holdout() -> list[dict]:
    """Load NULL-CC holdout pairs for post-injection formality evaluation."""
    null_cfg = DOMAINS.get("null")
    if null_cfg is None:
        raise ValueError("'null' domain not found in DOMAINS config")

    holdout_path = Path(null_cfg["holdout_data"])
    if not holdout_path.exists():
        # Fallback to standard location
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        holdout_path = repo_root / "ContrastiveData" / "data" / "holdout" / "null_cc_holdout.jsonl"

    if not holdout_path.exists():
        print(f"NULL-CC holdout not found at {holdout_path}")
        print("Generate it first: python -m ContrastiveData.src.null_cc_main")
        return []

    pairs = []
    with open(holdout_path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            formal = obj.get("target_formal", "").strip()
            informal = obj.get("retain_informal", "").strip()
            if formal and informal:
                pairs.append({
                    "line_num": line_num,
                    "formal": formal,
                    "informal": informal,
                    "category": obj.get("category", "unknown"),
                    "complexity": obj.get("complexity", "unknown"),
                })

    print(f"Loaded {len(pairs)} holdout pairs from {holdout_path}")
    return pairs


def evaluate_formality(model_key: str, adapter_path: str | None = None,
                       max_examples: int = 500,
                       use_cogbench_loader: bool = False):
    """Evaluate a model on NULL-CC formality classification.

    For each holdout pair, test both the formal and informal text.
    The model should label formal text as "formal" and informal as "informal".
    """
    pairs = load_null_cc_holdout()
    if not pairs:
        return None

    if len(pairs) > max_examples:
        pairs = pairs[:max_examples]

    # Load model
    print(f"Loading model: {model_key}...")
    if use_cogbench_loader and COGBENCH_MODELS and model_key in COGBENCH_MODELS:
        model, tokenizer = load_model(model_key)
    else:
        model, tokenizer, _ = load_student(model_key)
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            print(f"Loaded adapter: {adapter_path}")

    model.eval()
    device = next(model.parameters()).device

    correct_formal = 0
    correct_informal = 0
    total = 0
    style_gaps = []
    by_category = {}

    for pair in tqdm(pairs, desc="Formality eval"):
        category = pair["category"]
        if category not in by_category:
            by_category[category] = {"correct": 0, "total": 0}

        # Test formal text — should be classified as "formal"
        prompt_f = PROMPT_TEMPLATE.format(text=pair["formal"][:512])
        score_formal_f = compute_log_prob(model, tokenizer, prompt_f, " formal", device)
        score_informal_f = compute_log_prob(model, tokenizer, prompt_f, " informal", device)
        if score_formal_f > score_informal_f:
            correct_formal += 1
            by_category[category]["correct"] += 1
        style_gaps.append(score_formal_f - score_informal_f)

        # Test informal text — should be classified as "informal"
        prompt_i = PROMPT_TEMPLATE.format(text=pair["informal"][:512])
        score_formal_i = compute_log_prob(model, tokenizer, prompt_i, " formal", device)
        score_informal_i = compute_log_prob(model, tokenizer, prompt_i, " informal", device)
        if score_informal_i > score_formal_i:
            correct_informal += 1
            by_category[category]["correct"] += 1
        style_gaps.append(score_informal_i - score_formal_i)

        by_category[category]["total"] += 2
        total += 2

    correct = correct_formal + correct_informal
    accuracy = correct / total if total > 0 else 0
    acc_formal = correct_formal / len(pairs) if pairs else 0
    acc_informal = correct_informal / len(pairs) if pairs else 0
    mean_style_gap = sum(style_gaps) / len(style_gaps) if style_gaps else 0

    logger.info(
        "FORMALITY_RESULT model=%s accuracy=%.4f formal_acc=%.4f informal_acc=%.4f "
        "style_gap=%.4f adapter=%s",
        model_key, accuracy, acc_formal, acc_informal, mean_style_gap,
        adapter_path or "none"
    )

    print(f"\nFormality Classification Results for {model_key}:")
    print(f"  Overall:  {correct}/{total} = {accuracy:.4f}")
    print(f"  Formal:   {correct_formal}/{len(pairs)} = {acc_formal:.4f}")
    print(f"  Informal: {correct_informal}/{len(pairs)} = {acc_informal:.4f}")
    print(f"  Style gap (mean log-prob diff): {mean_style_gap:.4f}")
    for cat, counts in sorted(by_category.items()):
        c, t = counts["correct"], counts["total"]
        print(f"  {cat}: {c}/{t} = {c/t:.4f}" if t > 0 else f"  {cat}: N/A")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adapter_label = Path(adapter_path).name if adapter_path else "base"
    experiment = "lora" if adapter_path else "zero_shot"

    result = EvalResult(
        model=model_key,
        task="formality",
        accuracy=accuracy,
        n_samples=total,
        experiment=experiment,
        driver_path=adapter_path or "",
    )
    save_results([result], str(RESULTS_DIR / "formality_results.csv"))

    detail = {
        "model": model_key,
        "adapter": adapter_path,
        "overall_accuracy": accuracy,
        "formal_accuracy": acc_formal,
        "informal_accuracy": acc_informal,
        "mean_style_gap": mean_style_gap,
        "total": total,
        "correct": correct,
        "correct_formal": correct_formal,
        "correct_informal": correct_informal,
        "by_category": {k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0}
                        for k, v in by_category.items()},
    }
    detail_path = RESULTS_DIR / f"formality_{model_key}_{adapter_label}.json"
    with open(detail_path, "w") as f:
        json.dump(detail, f, indent=2)

    del model
    torch.cuda.empty_cache()
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Formality classification evaluation (NULL-CC)")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(STUDENT_MODELS.keys()) + ["all"])
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA/Q-DoRA adapter")
    parser.add_argument("--max-examples", type=int, default=500,
                        help="Max holdout pairs to evaluate (default 500)")
    parser.add_argument("--cogbench-loader", action="store_true",
                        help="Use CogBench model loader (NF4 for large models)")
    args = parser.parse_args()

    if args.model == "all":
        for model_key in STUDENT_MODELS:
            evaluate_formality(model_key, args.adapter, args.max_examples,
                               args.cogbench_loader)
    else:
        evaluate_formality(args.model, args.adapter, args.max_examples,
                           args.cogbench_loader)


if __name__ == "__main__":
    main()
