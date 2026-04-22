"""Unified model loading for all student architectures."""

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.seeds import STUDENT_MODELS

logger = logging.getLogger("model_loader")


def load_student(model_key: str, dtype=torch.bfloat16, device_map="auto"):
    """Load a student model by key (e.g. 'llama-3-8b')."""
    cfg = STUDENT_MODELS[model_key]
    logger.info("LOAD_MODEL key=%s hf_id=%s arch=%s dtype=%s",
                model_key, cfg["hf_id"], cfg.get("arch", "unknown"), dtype)
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"],
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("LOAD_MODEL_DONE key=%s total_params=%d (%.1fB)",
                model_key, n_params, n_params / 1e9)
    return model, tokenizer, cfg


def attach_lora(model, cfg, rank=32, alpha=16, dropout=0.05):
    """Attach LoRA adapters to a student model."""
    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=cfg["lora_targets"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("ATTACH_LORA rank=%d alpha=%d targets=%s trainable=%d (%.2f%%)",
                rank, alpha, cfg["lora_targets"], trainable, 100 * trainable / total)
    return model
