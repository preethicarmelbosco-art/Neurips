"""LoRA contrastive fine-tuning on CogBench.

Trains a LoRA adapter on the `target` text of one CogBench domain.
See the paired `evaluate.py` for primary + held-out benchmark scoring.

Usage:
    python train.py --model llama-3-8b --domain math --seed 42
    python train.py --run-all
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.seeds import DOMAINS, SEEDS, STUDENT_MODELS, WANDB_PROJECT
from utils.model_loader import attach_lora, load_student

os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
LOG_DIR = RESULTS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("lora_train")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(LOG_DIR / "lora_train.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(_fh)
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)
    logger.addHandler(_ch)


def load_contrastive_data(domain_key: str, tokenizer, max_seq_len: int = 1024):
    """Tokenize the target column of a CogBench domain for causal-LM training."""
    domain = DOMAINS[domain_key]
    dataset = load_dataset("json", data_files=domain["train_data"], split="train")
    text_field = domain["target_field"]

    def tokenize(examples):
        toks = tokenizer(
            examples[text_field],
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
        )
        toks["labels"] = [
            [(t if m == 1 else -100) for t, m in zip(ids, mask)]
            for ids, mask in zip(toks["input_ids"], toks["attention_mask"])
        ]
        return toks

    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


def train_lora(model_key: str, domain_key: str, seed: int):
    """Train a single LoRA run and return its output directory."""
    run_name = f"lora_{model_key}_{domain_key}_seed{seed}"
    output_dir = RESULTS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if (output_dir / "adapter" / "adapter_config.json").exists():
        print(f"Skipping {run_name} — already completed")
        return output_dir

    print(f"\n{'=' * 60}\nTraining: {run_name}\n  Corpus: {DOMAINS[domain_key]['corpus']}\n{'=' * 60}\n")
    logger.info("LORA_START model=%s domain=%s seed=%d", model_key, domain_key, seed)
    start_time = time.time()

    model, tokenizer, cfg = load_student(model_key)
    model = attach_lora(model, cfg, rank=32, alpha=16, dropout=0.05)
    is_ssm = cfg.get("arch") == "ssm"
    if not is_ssm:
        model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    dataset = load_contrastive_data(domain_key, tokenizer)

    # SSMs need a smaller LR and no gradient checkpointing — the fused Mamba
    # kernels interact poorly with checkpointing and destabilise the loss.
    lr = 1e-5 if is_ssm else 2e-4

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
        seed=seed,
        report_to="wandb",
        run_name=run_name,
        gradient_checkpointing=not is_ssm,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    trainer.save_model(str(output_dir / "adapter"))

    final_loss = (
        train_result.training_loss
        if hasattr(train_result, "training_loss")
        else (train_result.metrics.get("train_loss") if getattr(train_result, "metrics", None) else None)
    )
    wall_time = time.time() - start_time

    meta = {
        "model": model_key,
        "domain": domain_key,
        "seed": seed,
        "method": "lora",
        "corpus": DOMAINS[domain_key]["corpus"],
        "adapter_path": str(output_dir / "adapter"),
        "final_loss": final_loss,
        "wall_time_s": wall_time,
        "train_metrics": train_result.metrics if getattr(train_result, "metrics", None) else {},
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("LORA_DONE model=%s domain=%s seed=%d final_loss=%s time=%.1fs",
                model_key, domain_key, seed, final_loss, wall_time)
    print(f"  Final loss: {final_loss}")

    # Close wandb so sequential runs get separate logs.
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass

    del model, trainer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return output_dir


def run_all():
    """Run the full M × D × S matrix (models × domains × seeds)."""
    for model_key in STUDENT_MODELS:
        for domain_key in DOMAINS:
            for seed in SEEDS:
                train_lora(model_key, domain_key, seed)


def main():
    parser = argparse.ArgumentParser(description="LoRA contrastive fine-tuning on CogBench")
    parser.add_argument("--model",  type=str, choices=list(STUDENT_MODELS.keys()))
    parser.add_argument("--domain", type=str, choices=list(DOMAINS.keys()))
    parser.add_argument("--seed",   type=int, choices=SEEDS)
    parser.add_argument("--run-all", action="store_true")
    args = parser.parse_args()

    if args.run_all:
        run_all()
    elif args.model and args.domain and args.seed is not None:
        train_lora(args.model, args.domain, args.seed)
    else:
        parser.error("Provide --model, --domain, --seed  OR  --run-all")


if __name__ == "__main__":
    main()