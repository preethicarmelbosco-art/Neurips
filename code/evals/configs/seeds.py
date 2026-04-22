"""Configuration for LoRA contrastive fine-tuning on CogBench.

Defines the student-model zoo, CogBench domain → split mapping, and the
held-out benchmark configuration used by `evaluate.py`.

Paths are resolved relative to the CogBench release root (the directory
containing `data/`). The default `_DATA_DIR` assumes the standard release
layout `<release_root>/data/{train,bench,holdout}/…`.
"""

from pathlib import Path

_LORA_DIR = Path(__file__).resolve().parent.parent
_RELEASE_ROOT = _LORA_DIR.parent.parent
_DATA_DIR = _RELEASE_ROOT / "data"

SEEDS = [42, 1337, 2024]
WANDB_PROJECT = "cogbench-lora"

STUDENT_MODELS = {
    "llama-3-8b": {
        "hf_id": "meta-llama/Meta-Llama-3-8B",
        "arch": "transformer",
        "n_layers": 32,
        "hidden_dim": 4096,
        "lora_targets": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "lm_eval_type": "hf",
    },
    "qwen-2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B",
        "arch": "transformer",
        "n_layers": 28,
        "hidden_dim": 3584,
        "lora_targets": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "lm_eval_type": "hf",
    },
    "falcon-mamba-7b": {
        "hf_id": "tiiuae/falcon-mamba-7b",
        "arch": "ssm",
        "n_layers": 64,
        "hidden_dim": 4096,
        "lora_targets": ["in_proj", "x_proj", "dt_proj"],
        "lm_eval_type": "hf",
        "lm_eval_args_extra": "trust_remote_code=True",
    },
    "mamba-2.8b": {
        "hf_id": "state-spaces/mamba-2.8b-hf",
        "arch": "ssm",
        "n_layers": 64,
        "hidden_dim": 2560,
        "lora_targets": ["in_proj", "x_proj", "dt_proj"],
        "lm_eval_type": "hf",
        "lm_eval_args_extra": "trust_remote_code=True",
    },
}


def _split(domain: str) -> dict:
    return {
        "train_data":   str(_DATA_DIR / "train"   / f"{domain}_train.jsonl"),
        "holdout_data": str(_DATA_DIR / "holdout" / f"{domain}_holdout.jsonl"),
        "bench_data":   str(_DATA_DIR / "bench"   / f"{domain}_bench.jsonl"),
    }


DOMAINS = {
    "math":      {"corpus": "SPL-CC",    **_split("spl_cc"),    "primary_bench": "hendrycks_math",     "heldout_bench": "gsm8k",         "target_field": "target",        "retain_field": "retain"},
    "tom":       {"corpus": "ToM-CC",    **_split("tom_cc"),    "primary_bench": "social_iqa",         "heldout_bench": "bigtom",        "target_field": "target",        "retain_field": "retain"},
    "causal":    {"corpus": "CTR-CC",    **_split("ctr_cc"),    "primary_bench": "copa",               "heldout_bench": "balanced_copa", "target_field": "target_text",   "retain_field": "retain_text"},
    "moral":     {"corpus": "MOR-CC",    **_split("mor_cc"),    "primary_bench": "ethics",             "heldout_bench": "truthfulqa",    "target_field": "target",        "retain_field": "retain"},
    "strategic": {"corpus": "STR-CC",    **_split("str_cc"),    "primary_bench": "bigbench_strategy",  "heldout_bench": "bigbench_diplomacy", "target_field": "target",   "retain_field": "retain"},
    "spatial":   {"corpus": "STP-CC",    **_split("stp_cc"),    "primary_bench": "stepgame",           "heldout_bench": "spartqa",       "target_field": "target",        "retain_field": "retain"},
    "core_math": {"corpus": "CORE-MATH", **_split("core_math"), "primary_bench": "hendrycks_math",     "heldout_bench": "gsm8k",         "target_field": "target_proof",  "retain_field": "retain_intuition"},
    "null":      {"corpus": "NULL-CC",   **_split("null_cc"),   "primary_bench": "formality",          "heldout_bench": None,            "target_field": "target_formal", "retain_field": "retain_informal"},
}


BENCHMARKS = {
    "hendrycks_math":     {"lm_eval_task": "minerva_math", "batch_size": 4, "num_fewshot": 4},
    "social_iqa":         {"lm_eval_task": "social_iqa", "batch_size": 8},
    "copa":               {"lm_eval_task": "copa", "batch_size": 8},
    "ethics":             {"lm_eval_task": "ethics_utilitarianism", "batch_size": 8},
    "bigbench_strategy":  {"lm_eval_task": None, "batch_size": 8, "custom": True},
    "stepgame":           {"lm_eval_task": None, "batch_size": 8, "custom": True},
    "gsm8k":              {"lm_eval_task": "gsm8k", "batch_size": 4},
    "bigtom":             {"lm_eval_task": None, "batch_size": 4, "custom": True},
    "balanced_copa":      {"lm_eval_task": None, "batch_size": 8, "custom": True},
    "truthfulqa":         {"lm_eval_task": "truthfulqa_mc2", "batch_size": 8},
    "bigbench_diplomacy": {"lm_eval_task": None, "batch_size": 8, "custom": True},
    "spartqa":            {"lm_eval_task": None, "batch_size": 8, "custom": True},
    "formality":          {"lm_eval_task": None, "batch_size": 8, "custom": True},
    "wikitext2":          {"lm_eval_task": "wikitext", "batch_size": 4},
}