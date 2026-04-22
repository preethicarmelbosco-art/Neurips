"""Custom LegalBench evaluator for legal reasoning benchmark.

LegalBench (Guha et al., 2023) contains 162 legal reasoning tasks.
We evaluate on a curated subset of binary/multi-class tasks relevant to
Causal + Moral + ToM reasoning.

Dataset: nguha/legalbench on HuggingFace (CC-BY-4.0).
Format: text → answer (string label). Scored via exact match.

Selected task groups:
  - Contract NLI tasks (14 binary tasks) — causal/logical reasoning
  - SARA tasks — numerical/moral reasoning
  - Consumer contracts — moral/ToM reasoning
  - Supply chain / Unfair TOS — causal + moral

Usage:
    python -m evaluation.scripts.eval_legalbench --model llama-3-8b
    python -m evaluation.scripts.eval_legalbench --model llama-3-8b --max-tasks 20
"""

import argparse
import json
import logging
import time
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.model_loader import load_student

LOG_DIR = Path(__file__).parent.parent / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("eval_legalbench")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(LOG_DIR / "eval_legalbench.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(_fh)
from utils.metrics import EvalResult, save_results

try:
    from evaluation.scripts.cogbench_eval import COGBENCH_MODELS, load_model
except ImportError:
    COGBENCH_MODELS = None

RESULTS_DIR = Path(__file__).parent.parent / "results" / "legalbench"

# Curated subset of LegalBench tasks relevant to Causal + Moral + ToM
LEGALBENCH_TASKS = [
    # Contract NLI — binary Yes/No (causal/logical reasoning)
    "contract_nli_confidentiality_of_agreement",
    "contract_nli_explicit_identification",
    "contract_nli_inclusion_of_verbally_conveyed_information",
    "contract_nli_limited_use",
    "contract_nli_no_licensing",
    "contract_nli_notice_on_compelled_disclosure",
    "contract_nli_permissible_acquirement_of_similar_information",
    "contract_nli_permissible_copy",
    "contract_nli_permissible_development_of_similar_information",
    "contract_nli_permissible_post_agreement_possession",
    "contract_nli_return_of_confidential_information",
    "contract_nli_sharing_with_employees",
    "contract_nli_sharing_with_third-parties",
    "contract_nli_survival_of_obligations",
    # Unfair TOS — moral reasoning
    "unfair_tos",
    # Consumer contracts — causal + moral
    "consumer_contracts_qa",
    # Trademark — multi-class classification
    "abercrombie",
    # Causal reasoning in legal context
    "legal_reasoning_causality",
    # Hearsay — ToM (understanding speaker intent/knowledge)
    "hearsay",
    # Judicial ethics — moral reasoning
    "nys_judicial_ethics",
]


def _wrap_chat_if_available(tokenizer, user_prompt: str) -> tuple[str, bool]:
    """Apply chat template if the tokenizer exposes one (instruct models).

    Raw completion prompts to instruct-tuned models are out-of-distribution:
    log-probs over candidate answer tokens become unreliable. Chat-templating
    puts the model back into its trained regime. Returns (final_prompt, is_chat).
    """
    if getattr(tokenizer, "chat_template", None):
        try:
            wrapped = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return wrapped, True
        except Exception:
            pass
    return user_prompt, False


def compute_choice_log_prob(model, tokenizer, prompt, answer_text, device):
    """Mean per-token log P(answer_text | prompt).

    Tokenises prompt and continuation separately and concatenates IDs, which
    avoids the re-tokenisation-boundary bug where tokenise(prompt+X) ≠
    tokenise(prompt)+tokenise(X) on BPE/SentencePiece — the prior version read
    logits from misaligned positions. Caller owns the leading-space contract:
    pass ``" Yes"`` for raw-completion prompts and ``"Yes"`` for chat-templated
    prompts (where the template already closes the boundary).
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    cont_ids = tokenizer(answer_text, return_tensors="pt",
                         add_special_tokens=False).input_ids.to(device)
    # Some tokenisers still emit BOS with add_special_tokens=False; strip it.
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None and cont_ids.shape[1] > 0 and cont_ids[0, 0].item() == bos:
        cont_ids = cont_ids[:, 1:]

    c_len = cont_ids.shape[1]
    if c_len == 0:
        return float("-inf")
    p_len = prompt_ids.shape[1]
    full_ids = torch.cat([prompt_ids, cont_ids], dim=1)

    with torch.no_grad():
        logits = model(full_ids).logits

    shift_logits = logits[:, p_len - 1 : p_len - 1 + c_len, :]
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(2, cont_ids.unsqueeze(-1)).squeeze(-1)
    return token_lp.sum().item() / c_len


def run_heldout(model, tokenizer, model_key: str, adapter_path: str | None = None,
                max_tasks: int = 0, max_per_task: int = 100) -> dict:
    """Evaluate a live (model, tokenizer) on LegalBench without load/unload."""
    model.eval()
    device = next(model.parameters()).device

    tasks = LEGALBENCH_TASKS[:max_tasks] if max_tasks > 0 else LEGALBENCH_TASKS
    all_correct = 0
    all_total = 0
    per_task = {}

    for task_name in tasks:
        print(f"\n  Loading LegalBench/{task_name}...")
        try:
            ds = load_dataset("nguha/legalbench", task_name, split="test",
                              trust_remote_code=True)
        except Exception:
            try:
                ds = load_dataset("nguha/legalbench", task_name, split="train",
                                  trust_remote_code=True)
            except Exception as e:
                print(f"    SKIP: {e}")
                continue

        if len(ds) > max_per_task:
            ds = ds.select(range(max_per_task))

        # Collect unique answer labels
        answer_labels = list(set(str(ex["answer"]).strip() for ex in ds))
        if not answer_labels:
            continue

        correct = 0
        total = 0

        for ex in tqdm(ds, desc=f"    {task_name}", leave=False):
            text = ex.get("text", "")
            gold = str(ex["answer"]).strip()

            user_prompt = (
                f"Legal text: {text}\n\n"
                f"Based on the legal text above, what is the correct classification?\n\n"
                f"Answer:"
            )
            final_prompt, is_chat = _wrap_chat_if_available(tokenizer, user_prompt)
            # Raw completion: leading space for natural next-token alignment.
            # Chat-templated: bare label — the template's trailing tokens
            # (e.g. "<|assistant|>\n") already set up the boundary.
            label_fmt = (lambda lbl: lbl) if is_chat else (lambda lbl: f" {lbl}")

            best_label = None
            best_score = float("-inf")
            for label in answer_labels:
                score = compute_choice_log_prob(model, tokenizer, final_prompt,
                                                label_fmt(label), device)
                if score > best_score:
                    best_score = score
                    best_label = label

            if best_label == gold:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0
        per_task[task_name] = {"correct": correct, "total": total, "accuracy": round(acc, 4)}
        all_correct += correct
        all_total += total
        print(f"    {task_name}: {correct}/{total} = {acc:.4f}")

    accuracy = all_correct / all_total if all_total > 0 else 0
    logger.info("LEGALBENCH_RESULT model=%s accuracy=%.4f correct=%d total=%d n_tasks=%d adapter=%s",
                model_key, accuracy, all_correct, all_total, len(per_task), adapter_path or "none")
    for t, v in per_task.items():
        logger.info("  LEGALBENCH_TASK task=%s accuracy=%.4f n=%d", t, v["accuracy"], v["total"])
    print(f"\nLegalBench Overall for {model_key}: {all_correct}/{all_total} = {accuracy:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adapter_label = Path(adapter_path).name if adapter_path else "base"

    result = EvalResult(
        model=model_key, task="legalbench", accuracy=accuracy,
        n_samples=all_total, experiment="lora" if adapter_path else "zero_shot",
        driver_path=adapter_path or "",
    )
    save_results([result], str(RESULTS_DIR / "legalbench_results.csv"))

    model_meta = {}
    if COGBENCH_MODELS and model_key in COGBENCH_MODELS:
        m = COGBENCH_MODELS[model_key]
        model_meta = {"size_b": m.get("size_b"), "family": m.get("family"),
                      "arch": m.get("arch"), "tier": m.get("tier")}

    detail = {"model": model_key, "adapter": adapter_path,
              "accuracy": accuracy, "correct": all_correct, "total": all_total,
              "n_tasks_evaluated": len(per_task),
              "per_task": per_task,
              "model_metadata": model_meta,
              "hypothesis": ["Causal", "Moral", "ToM"],
              "paper_table": "Table 10 Row 3"}
    with open(RESULTS_DIR / f"legalbench_{model_key}_{adapter_label}.json", "w") as f:
        json.dump(detail, f, indent=2)

    return detail


def evaluate_legalbench(model_key: str, adapter_path: str | None = None,
                        max_tasks: int = 0, max_per_task: int = 100,
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
                             adapter_path=adapter_path, max_tasks=max_tasks,
                             max_per_task=max_per_task)
    finally:
        del model
        torch.cuda.empty_cache()
    return detail["accuracy"]


def main():
    parser = argparse.ArgumentParser(description="LegalBench evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--max-per-task", type=int, default=100)
    parser.add_argument("--cogbench-loader", action="store_true")
    args = parser.parse_args()
    evaluate_legalbench(args.model, args.adapter, args.max_tasks,
                        args.max_per_task, args.cogbench_loader)


if __name__ == "__main__":
    main()
