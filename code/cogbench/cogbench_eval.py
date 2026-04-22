"""CogBench Contrastive Accuracy Evaluator — Paper 2 (D&B) Tables 5, 6, 7, 8, 9.

Evaluates N models on 8 CogBench corpora using contrastive accuracy:
  CA(m, c) = acc_target(m, c) - acc_retain(m, c)

Each corpus uses a question template (Section 4.2.1 of the paper).
Auto-graded: SPL-CC (exact number), STP-CC (exact location).
Judge-graded: ToM-CC, CTR-CC, MOR-CC, STR-CC, CORE-MATH (3-judge majority vote, factual accuracy).
NULL-CC: style-only control — CA expected ≈ 0 (no cognitive operation differs).
COIN: cognitive bias / fallacy detection (judge-graded).

Fills:
  - Table 5:  Primitive Profiles (24 models × 6 primitives)
  - Table 6:  Inter-Primitive Correlation Matrix
  - Table 7:  Synthetic-to-Real Correlation (CogBench vs held-out benchmarks)
  - Table 8:  Primitive Scores by Model Scale
  - Table 9:  Primitive Profiles by Architecture Type
  - Figure 1: Radar Charts
  - Figure 2: Correlation Heatmap
  - Figure 3: Scaling Curves

Usage:
    python -m evaluation.scripts.cogbench_eval --model llama-3-8b --corpus spl_cc
    python -m evaluation.scripts.cogbench_eval --model llama-3-8b --all-corpora
    python -m evaluation.scripts.cogbench_eval --all-models --all-corpora
    python -m evaluation.scripts.cogbench_eval --analysis  # post-hoc Tables 6-9 from saved results
"""

import argparse
import asyncio
import csv
import gc
import json
import logging
import os
# Must be set before any `from datasets import ...` so the heldout-chain
# benchmarks (cais/wmdp, EleutherAI/hendrycks_ethics, etc.) can load their
# remote code without an interactive y/N prompt.
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

import re
import time
import torch
import numpy as np

# ── Compatibility shim for trust_remote_code models ───────────────────
# Several HF model repos (Phi-3, InternLM2, StableLM, Jamba) import
# is_flash_attn_greater_or_equal_2_10 which was removed in transformers ≥5.x.
# Alias it to the current API so trust_remote_code=True model loading works.
import transformers.utils
if not hasattr(transformers.utils, "is_flash_attn_greater_or_equal_2_10"):
    from functools import partial
    from transformers.utils import is_flash_attn_greater_or_equal
    transformers.utils.is_flash_attn_greater_or_equal_2_10 = partial(
        is_flash_attn_greater_or_equal, "2.10"
    )

# vLLM auto-detect: activated when SM>=8 and vllm importable (via PYTHONPATH=/path/to/vllm_site).
# Disable with COGBENCH_NO_VLLM=1. Judges always use HF regardless.
_VLLM_AVAILABLE = False
try:
    import vllm as _vllm_module  # noqa: F401
    _VLLM_AVAILABLE = True
except ImportError:
    pass


def _use_vllm() -> bool:
    if not _VLLM_AVAILABLE:
        return False
    if os.environ.get("COGBENCH_NO_VLLM") == "1":
        return False
    try:
        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability()
        if major < 8:
            return False
        # Verify installed torch has kernels for this SM (e.g. Blackwell sm_120
        # is not in torch 2.6's arch list even though major>=8).
        supported = torch.cuda.get_arch_list() or []
        cur = f"sm_{major}{minor}"
        if supported and cur not in supported:
            return False
        return True
    except Exception:
        return False

from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field, model_validator

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../ContrastiveData/src"))

from cogbench_model_json import safe_write_model_json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = PROJECT_ROOT.parent
DATA_DIR = REPO_ROOT / "ContrastiveData" / "data"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results" / "cogbench"
PENDING_DIR = RESULTS_DIR / "pending"
CLAIMS_DIR = RESULTS_DIR / "claims"
LOG_DIR = PROJECT_ROOT / "evaluation" / "results" / "logs"

# ── Logging setup ────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("cogbench_eval")
logger.setLevel(logging.DEBUG)

# Canonical log file — overridden by --log-file in main(), but this default
# is used if the module is imported directly or for pre-arg-parse messages.
_LOG_FILE = LOG_DIR / "cogbench_eval.log"


class _TeeStream:
    """Write to both the original stream (console) and a log file."""
    def __init__(self, original, log_file_path: Path):
        self._original = original
        self._log = open(log_file_path, "a")

    def write(self, text):
        self._original.write(text)
        self._log.write(text)
        self._log.flush()

    def flush(self):
        self._original.flush()
        self._log.flush()

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return False


_PER_MODEL_HANDLER: logging.FileHandler | None = None
_PER_MODEL_TEE_OUT: _TeeStream | None = None
_PER_MODEL_TEE_ERR: _TeeStream | None = None


def _setup_logging(log_file: Path | None = None):
    """(Re)configure logging + stdout/stderr tee to a single consolidated log file."""
    global _LOG_FILE
    if log_file is not None:
        _LOG_FILE = log_file

    # Remove any existing handlers so we can reconfigure cleanly
    logger.handlers.clear()

    # File handler — structured log lines
    fh = logging.FileHandler(_LOG_FILE)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(stream=sys.__stdout__)  # always use real stdout
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
    logger.addHandler(ch)

    # Tee stdout & stderr → same log file (captures tqdm, print, warnings)
    sys.stdout = _TeeStream(sys.__stdout__, _LOG_FILE)
    sys.stderr = _TeeStream(sys.__stderr__, _LOG_FILE)


def _attach_per_model_log(model_key: str, node_id: str) -> Path:
    """Add a second FileHandler + stdout/stderr tee for the active model so
    each (model, invocation) gets its own log file alongside the global one.
    Returns the per-model log path."""
    global _PER_MODEL_HANDLER, _PER_MODEL_TEE_OUT, _PER_MODEL_TEE_ERR
    # Detach any prior per-model handler first
    _detach_per_model_log()

    ts = time.strftime("%Y%m%d_%H%M%S")
    model_log = LOG_DIR / f"cogbench_{node_id}_{model_key}_{ts}.log"
    fh = logging.FileHandler(model_log)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)
    _PER_MODEL_HANDLER = fh

    # Extend tees so tqdm/print also land in the per-model log
    global_fh = _LOG_FILE
    _PER_MODEL_TEE_OUT = _TeeStream(sys.stdout, model_log)
    _PER_MODEL_TEE_ERR = _TeeStream(sys.stderr, model_log)
    sys.stdout = _PER_MODEL_TEE_OUT
    sys.stderr = _PER_MODEL_TEE_ERR
    logger.info("PER_MODEL_LOG attached: %s", model_log.name)
    return model_log


def _detach_per_model_log() -> None:
    """Detach the per-model handler/tees added by _attach_per_model_log()."""
    global _PER_MODEL_HANDLER, _PER_MODEL_TEE_OUT, _PER_MODEL_TEE_ERR
    if _PER_MODEL_HANDLER is not None:
        logger.removeHandler(_PER_MODEL_HANDLER)
        try:
            _PER_MODEL_HANDLER.close()
        except Exception:
            pass
        _PER_MODEL_HANDLER = None
    # Tees wrap the previous stream; restore to the global tee
    if _PER_MODEL_TEE_OUT is not None:
        sys.stdout = _TeeStream(sys.__stdout__, _LOG_FILE)
        _PER_MODEL_TEE_OUT = None
    if _PER_MODEL_TEE_ERR is not None:
        sys.stderr = _TeeStream(sys.__stderr__, _LOG_FILE)
        _PER_MODEL_TEE_ERR = None


# Default init so early import-time messages still get captured
_setup_logging()


# ── Judge Panel for CogBench grading ──────────────────────────────────────

class CogBenchVerdict(BaseModel):
    """Verdict model for CogBench factual accuracy grading."""
    correct: bool = Field(
        description="True if the answer is factually correct according to the source text."
    )
    reasoning: str = Field(
        default="",
        description="Brief reasoning for the verdict."
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_keys(cls, data):
        """Normalize LLM key variants to snake_case field names.

        Handles: 'Correct', 'is_correct', 'isCorrect', etc.
        """
        if not isinstance(data, dict) or "correct" in data:
            return data
        normalized = {}
        for key in list(data.keys()):
            norm = re.sub(r'([a-z])([A-Z])', r'\1_\2', key)
            norm = norm.strip().lower().replace(" ", "_").replace("-", "_")
            if norm == "correct" or norm == "is_correct":
                data["correct"] = data.pop(key)
                return data
            normalized[norm] = key
        # Fallback: any key containing "correct"
        for norm, orig in normalized.items():
            if "correct" in norm:
                data["correct"] = data.pop(orig)
                return data
        return data


JUDGE_SYSTEM_PROMPT = (
    "You are a factual accuracy judge. You will be given a source text, "
    "a question, and a model's answer. Judge whether the answer is "
    "factually correct according to the source text. "
    'Respond with JSON: {"correct": true/false, "reasoning": "..."}'
)

JUDGE_USER_TEMPLATE = (
    "Source text (excerpt):\n{source_text}\n\n"
    "Question: {question}\n\n"
    "Model's answer: {answer}\n\n"
    "{judge_instruction}\n\n"
    'Respond with JSON: {{"correct": true/false, "reasoning": "..."}}'
)

# Maximum characters of source text to include in judge context.
# Prevents judge prompt from exceeding context window.
_JUDGE_SOURCE_MAX_CHARS = 6000

# Lazy-initialized singleton
_judge_panel = None
_judge_mode = "ollama"   # "ollama" or "local" — set from CLI
_judge_device = "cuda:0"  # GPU device for local judges (single-GPU mode)
_judge_devices = None     # Multi-GPU: list of devices, one per judge (e.g. ["cuda:0","cuda:1","cuda:2"])

# HF IDs for judge trios (instruct variants for JSON output).
# "small" tier: 7-9B — used for corpus cleaning / revalidation pipeline.
# "large" tier: 9-14B — used for CogBench evaluation grading (default).
#   Larger judges grade more accurately, especially for outputs from 27-72B models.
#   Each fits on a single 32 GB GPU in fp16.
LOCAL_JUDGE_MODELS_SMALL = [
    {"label": "qwen2.5-7b", "hf_id": "Qwen/Qwen2.5-7B-Instruct"},
    {"label": "gemma-2-9b", "hf_id": "google/gemma-2-9b-it"},
    {"label": "llama-3.1-8b", "hf_id": "meta-llama/Llama-3.1-8B-Instruct"},
]
LOCAL_JUDGE_MODELS_LARGE = [
    # 2-judge unanimous panel (2026-04-14 onward). Gemma-2-9B removed:
    # (a) chat template rejects system role — required merged-prompt workaround,
    # (b) it was the slowest judge (~20 s/it vs 13/16 for the other two),
    # (c) halving judges per file freed GPUs to run 3 graders concurrently.
    # Early SPL-CC files (llama-3-8b, qwen-2.5-7b, olmo-2-13b) were graded with
    # the prior 3-judge panel; documented in the paper.
    {"label": "qwen2.5-14b", "hf_id": "Qwen/Qwen2.5-14B-Instruct"},
    {"label": "mistral-nemo-12b", "hf_id": "mistralai/Mistral-Nemo-Instruct-2407"},
]
LOCAL_JUDGE_MODELS = LOCAL_JUDGE_MODELS_LARGE  # default for evaluation


class LocalJudgePanel:
    """Judge panel that loads models directly on GPU — no Ollama needed.

    Two modes:
      1. Single-GPU (default): loads each judge sequentially on one device
         to conserve VRAM. Use via: --judge-mode local --judge-device cuda:0
      2. Multi-GPU (parallel): loads all 3 judges on separate GPUs simultaneously,
         grades in parallel using ThreadPoolExecutor. ~3× faster.
         Use via: --judge-mode local --judge-devices cuda:0,cuda:1,cuda:2
    """

    def __init__(self, judge_models=None, device="cuda:0", devices=None):
        self.judge_models = judge_models or LOCAL_JUDGE_MODELS
        self.device = device
        self.devices = devices  # list of devices for parallel mode, or None
        self.parallel = devices is not None and len(devices) >= len(self.judge_models)
        self.judge_count = len(self.judge_models)
        self.majority = self.judge_count // 2 + 1

        if self.parallel:
            logger.info("LocalJudgePanel PARALLEL: %d judges on %s: %s",
                        self.judge_count, devices,
                        [m["label"] for m in self.judge_models])
        else:
            logger.info("LocalJudgePanel SEQUENTIAL: %d judges on %s: %s",
                        self.judge_count, device,
                        [m["label"] for m in self.judge_models])

        # Pre-loaded models for parallel mode (populated by _load_all_judges)
        self._loaded_judges = None

    def _load_judge(self, hf_id, device=None):
        """Load a judge model. device can be a single str or a list of strs.

        Single device (e.g. "cuda:0"): loads entire model on that GPU.
        Multi-device (e.g. ["cuda:0","cuda:1"]): shards model across GPUs
        using device_map="balanced" with max_memory constraints.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import get_token
        token = get_token()
        target_device = device or self.device
        resolved = _resolve_hf_path(hf_id)

        # Pre-Ampere GPUs (compute capability < 8.0) don't efficiently support
        # bf16 — fall back to fp16 there, bf16 on Ampere+.
        judge_dtype = torch.float16
        try:
            dev_str = target_device[0] if isinstance(target_device, list) else target_device
            if "cuda" in str(dev_str):
                idx = int(str(dev_str).split(":")[-1]) if ":" in str(dev_str) else 0
                if torch.cuda.get_device_capability(idx)[0] >= 8:
                    judge_dtype = torch.bfloat16  # Ampere+ supports bf16 natively
        except Exception:
            pass  # fallback to fp16
        logger.info("  Loading %s with dtype=%s", hf_id, judge_dtype)

        tokenizer = AutoTokenizer.from_pretrained(resolved, token=token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if isinstance(target_device, list) and len(target_device) > 1:
            max_memory = {d: "28GiB" for d in target_device}
            logger.info("  Sharding %s across %s", hf_id, target_device)
            model = AutoModelForCausalLM.from_pretrained(
                resolved, dtype=judge_dtype,
                device_map="balanced", max_memory=max_memory,
                token=token,
            )
        else:
            if isinstance(target_device, list):
                target_device = target_device[0]
            model = AutoModelForCausalLM.from_pretrained(
                resolved, dtype=judge_dtype,
                device_map=target_device, token=token,
            )
        model.eval()
        return model, tokenizer

    @staticmethod
    def _probe_system_role(tokenizer, label):
        """Return True iff the tokenizer's chat template accepts a system turn.
        Some templates (notably Gemma-2) raise ValueError on system messages."""
        try:
            tokenizer.apply_chat_template(
                [{"role": "system", "content": "x"},
                 {"role": "user", "content": "y"}],
                tokenize=False, add_generation_prompt=True)
            return True
        except Exception as e:
            logger.info("Judge %s chat template rejects system role (%s); "
                        "will merge system prompt into user turn.", label, type(e).__name__)
            return False

    def _load_all_judges(self):
        """Load all judge models onto their assigned GPUs (parallel mode).

        self.devices is a list of device specs — each element is either a
        single device string ("cuda:0") or a list of devices for multi-GPU
        sharding (["cuda:0", "cuda:1"]).
        """
        if self._loaded_judges is not None:
            return self._loaded_judges
        self._loaded_judges = []
        for i, judge_cfg in enumerate(self.judge_models):
            dev = self.devices[i]
            label, hf_id = judge_cfg["label"], judge_cfg["hf_id"]
            logger.info("Loading judge %s → %s (%s)", label, dev, hf_id)
            model, tokenizer = self._load_judge(hf_id, device=dev)
            supports_system = self._probe_system_role(tokenizer, label)
            self._loaded_judges.append((label, model, tokenizer, dev, supports_system))
        logger.info("All %d judges loaded for parallel grading", len(self._loaded_judges))
        return self._loaded_judges

    def free_all_judges(self):
        """Free all pre-loaded judge models (parallel mode cleanup)."""
        if self._loaded_judges is None:
            return
        for label, model, tokenizer, dev, _supports_system in self._loaded_judges:
            logger.info("Freeing judge %s on %s", label, dev)
            _free_vram(model)
        self._loaded_judges = None

    def _format_judge_prompt(self, tokenizer, question, answer, source_text,
                             judge_prompt, supports_system=True):
        source_excerpt = source_text[-_JUDGE_SOURCE_MAX_CHARS:] if len(source_text) > _JUDGE_SOURCE_MAX_CHARS else source_text
        user_text = JUDGE_USER_TEMPLATE.format(
            source_text=source_excerpt, question=question,
            answer=answer, judge_instruction=judge_prompt)
        if supports_system:
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]
        else:
            # Gemma-style templates: prepend system prompt into the user turn.
            messages = [
                {"role": "user",
                 "content": f"{JUDGE_SYSTEM_PROMPT}\n\n{user_text}"},
            ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    @staticmethod
    def _parse_verdict(text):
        """Extract correct boolean from judge output."""
        match = re.search(
            r'"correct"\s*:\s*(true|false)', text, re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true"
        # Fallback: bare YES/NO response
        stripped = text.strip().upper()
        if stripped.startswith("YES"):
            return True
        if stripped.startswith("NO"):
            return False
        return None  # unparseable — skip this pair

    def _grade_batch(self, model, tokenizer, prompts, batch_size=8):
        """Run batched inference through one judge model."""
        results = []
        orig_pad_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        for i in tqdm(range(0, len(prompts), batch_size),
                      desc="    judge batch", leave=False):
            batch = prompts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                               max_length=2048, padding=True).to(model.device)  # 2048 chosen so bs=8 KV cache fits on a 32 GB GPU
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=32,  # 2026-04-14: cut from 128 — parser only needs "correct":t/f (~12 tokens); ~3x decode speedup, bit-exact verdicts
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            for output in outputs:
                text = tokenizer.decode(output[input_len:], skip_special_tokens=True)
                results.append(self._parse_verdict(text))

        tokenizer.padding_side = orig_pad_side
        return results

    @staticmethod
    def _judge_cache_path(cache_key, label):
        """Per-judge verdict cache path. None when cache_key is None (disabled)."""
        if not cache_key:
            return None
        cache_dir = PENDING_DIR / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{cache_key}__{label}.json"

    @staticmethod
    def _load_cached_verdicts(cache_path, expected_len):
        """Return list[bool|None] if cache is present and length-matches, else None."""
        if cache_path is None or not cache_path.exists():
            return None
        try:
            with open(cache_path) as f:
                verdicts = json.load(f)
            if len(verdicts) != expected_len:
                logger.warning("Judge cache %s length %d != expected %d; ignoring.",
                               cache_path.name, len(verdicts), expected_len)
                return None
            return verdicts
        except Exception as e:
            logger.warning("Failed to read judge cache %s: %s; ignoring.", cache_path, e)
            return None

    @staticmethod
    def _save_cached_verdicts(cache_path, verdicts):
        if cache_path is None:
            return
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(verdicts, f)
        tmp.rename(cache_path)

    def _grade_all_parallel(self, answers, source_texts, question, judge_prompt,
                            batch_size=8, cache_key=None):
        """Grade using all judges in parallel on separate GPUs (ThreadPoolExecutor).

        Each judge runs on its own CUDA device — no GIL contention because
        the heavy work is in torch CUDA kernels (releases GIL).

        If cache_key is set, per-judge verdicts are cached at
        PENDING_DIR/.cache/<cache_key>__<label>.json so a re-run after failure
        can skip judges that already finished.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        judges = self._load_all_judges()

        def _run_judge(label, model, tokenizer, dev, supports_system):
            cache_path = self._judge_cache_path(cache_key, label)
            cached = self._load_cached_verdicts(cache_path, len(answers))
            if cached is not None:
                logger.info("  [%s on %s] Using cached verdicts (%d answers)",
                            label, dev, len(cached))
                return cached
            prompts = [self._format_judge_prompt(tokenizer, question, ans, src,
                                                 judge_prompt, supports_system)
                       for ans, src in zip(answers, source_texts)]
            logger.info("  [%s on %s] Grading %d answers...", label, dev, len(prompts))
            verdicts = self._grade_batch(model, tokenizer, prompts, batch_size)
            valid = [v for v in verdicts if v is not None]
            true_count = sum(valid) if valid else 0
            logger.info("  [%s on %s] %d/%d true (%d unparseable)",
                        label, dev, true_count, len(verdicts), len(verdicts) - len(valid))
            self._save_cached_verdicts(cache_path, verdicts)
            return verdicts

        all_verdicts = [None] * len(judges)
        with ThreadPoolExecutor(max_workers=len(judges)) as pool:
            futures = {}
            for idx, (label, model, tokenizer, dev, supports_system) in enumerate(judges):
                fut = pool.submit(_run_judge, label, model, tokenizer, dev, supports_system)
                futures[fut] = idx
            for fut in as_completed(futures):
                idx = futures[fut]
                all_verdicts[idx] = fut.result()

        return self._majority_vote(all_verdicts, len(answers))

    def _grade_all_sequential(self, answers, source_texts, question, judge_prompt,
                              batch_size=8, cache_key=None):
        """Grade using one judge at a time on a single GPU (original behavior)."""
        all_verdicts = []
        for judge_cfg in self.judge_models:
            label, hf_id = judge_cfg["label"], judge_cfg["hf_id"]
            cache_path = self._judge_cache_path(cache_key, label)
            cached = self._load_cached_verdicts(cache_path, len(answers))
            if cached is not None:
                logger.info("Using cached verdicts for %s (%d answers)", label, len(cached))
                all_verdicts.append(cached)
                continue

            logger.info("Loading local judge: %s (%s)", label, hf_id)
            model, tokenizer = self._load_judge(hf_id)
            supports_system = self._probe_system_role(tokenizer, label)

            prompts = [self._format_judge_prompt(tokenizer, question, ans, src,
                                                 judge_prompt, supports_system)
                       for ans, src in zip(answers, source_texts)]
            logger.info("  Grading %d answers with %s...", len(prompts), label)
            verdicts = self._grade_batch(model, tokenizer, prompts, batch_size)
            self._save_cached_verdicts(cache_path, verdicts)
            all_verdicts.append(verdicts)

            valid_verdicts = [v for v in verdicts if v is not None]
            true_count = sum(valid_verdicts) if valid_verdicts else 0
            logger.info("  %s: %d/%d true (%d unparseable)", label, true_count,
                        len(verdicts), len(verdicts) - len(valid_verdicts))
            _free_vram(model)

        return self._majority_vote(all_verdicts, len(answers))

    @staticmethod
    def _majority_vote(all_verdicts, n_answers):
        """Compute majority vote across judge verdicts."""
        majority = len(all_verdicts) // 2 + 1
        results = []
        for i in range(n_answers):
            votes_for = [v[i] for v in all_verdicts]
            if any(v is None for v in votes_for):
                results.append(None)
            else:
                results.append(sum(votes_for) >= majority)

        valid_results = [r for r in results if r is not None]
        logger.info("LocalJudgePanel majority: %d/%d true (%d skipped)",
                    sum(valid_results) if valid_results else 0,
                    len(results), len(results) - len(valid_results))
        return results

    def grade_all(self, answers, source_texts, question, judge_prompt, batch_size=8,
                  cache_key=None):
        """Grade all answers through judge panel, then majority-vote.

        Dispatches to parallel (multi-GPU) or sequential (single-GPU) mode.
        Returns list of bool|None. None = unparseable verdict (skip pair).
        """
        if self.parallel:
            return self._grade_all_parallel(answers, source_texts, question, judge_prompt,
                                            batch_size, cache_key=cache_key)
        return self._grade_all_sequential(answers, source_texts, question, judge_prompt,
                                          batch_size, cache_key=cache_key)


def _get_judge_panel():
    """Lazy-init the JudgePanel from environment variables or local GPU."""
    global _judge_panel
    if _judge_panel is not None:
        return _judge_panel

    if _judge_mode == "local":
        _judge_panel = LocalJudgePanel(
            device=_judge_device,
            devices=_judge_devices,
        )
        return _judge_panel

    try:
        from judge_panel import JudgePanel, load_judge_configs_from_env
    except ImportError as e:
        logger.warning("Judge panel import failed (%s) — judge-graded corpora unavailable", e)
        return None

    configs = load_judge_configs_from_env(
        prefix="JUDGE",
        fallback_model=os.getenv("JUDGE_MODEL", "qwen2.5:14b"),
        fallback_base_url=os.getenv("JUDGE_BASE_URL", "http://localhost:11434/v1"),
    )

    if not configs:
        return None

    _judge_panel = JudgePanel(
        configs=configs,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt_template="{text}",  # We'll format the full text ourselves
        response_model=CogBenchVerdict,
        verdict_field="correct",
        evidence_field=None,
    )
    return _judge_panel

# ── Model Zoo (Paper 2 Table 4: 24 models) ──────────────────────────────

COGBENCH_MODELS = {
    # Dense Transformers
    "llama-3-8b":       {"hf_id": "meta-llama/Meta-Llama-3-8B", "size_b": 8, "family": "Meta", "arch": "Dense Transformer", "tier": "7-9B", "hardware": "5090"},
    "llama-3-70b":      {"hf_id": "meta-llama/Meta-Llama-3-70B", "size_b": 70, "family": "Meta", "arch": "Dense Transformer", "tier": "27-72B", "hardware": "A100"},
    "llama-3.2-3b":     {"hf_id": "meta-llama/Llama-3.2-3B", "size_b": 3, "family": "Meta", "arch": "Dense Transformer", "tier": "1.7-3.8B", "hardware": "5090"},
    "qwen-2.5-7b":      {"hf_id": "Qwen/Qwen2.5-7B", "size_b": 7, "family": "Alibaba", "arch": "Dense Transformer", "tier": "7-9B", "hardware": "5090"},
    "qwen-2.5-14b":     {"hf_id": "Qwen/Qwen2.5-14B", "size_b": 14, "family": "Alibaba", "arch": "Dense Transformer", "tier": "12-14B", "hardware": "5090"},
    "qwen-2.5-72b":     {"hf_id": "Qwen/Qwen2.5-72B", "size_b": 72, "family": "Alibaba", "arch": "Dense Transformer", "tier": "27-72B", "hardware": "A100"},
    "mistral-7b":       {"hf_id": "mistralai/Mistral-7B-v0.3", "size_b": 7, "family": "Mistral", "arch": "Dense Transformer", "tier": "7-9B", "hardware": "5090"},
    "gemma-2-9b":       {"hf_id": "google/gemma-2-9b", "size_b": 9, "family": "Google", "arch": "Dense Transformer", "tier": "7-9B", "hardware": "5090"},
    "gemma-2-27b":      {"hf_id": "google/gemma-2-27b", "size_b": 27, "family": "Google", "arch": "Dense Transformer", "tier": "27-72B", "hardware": "A100"},
    "phi-3-mini":       {"hf_id": "microsoft/Phi-3-mini-4k-instruct", "size_b": 3.8, "family": "Microsoft", "arch": "Dense Transformer", "tier": "1.7-3.8B", "hardware": "5090"},
    "phi-3-medium":     {"hf_id": "microsoft/Phi-3-medium-4k-instruct", "size_b": 14, "family": "Microsoft", "arch": "Dense Transformer", "tier": "12-14B", "hardware": "5090"},
    "yi-1.5-9b":        {"hf_id": "01-ai/Yi-1.5-9B", "size_b": 9, "family": "01.AI", "arch": "Dense Transformer", "tier": "7-9B", "hardware": "5090"},
    "yi-1.5-34b":       {"hf_id": "01-ai/Yi-1.5-34B", "size_b": 34, "family": "01.AI", "arch": "Dense Transformer", "tier": "27-72B", "hardware": "A100"},
    "qwen-2-7b":        {"hf_id": "Qwen/Qwen2-7B", "size_b": 7, "family": "Alibaba", "arch": "Dense Transformer", "tier": "7-9B", "hardware": "5090"},
    "olmo-7b":          {"hf_id": "allenai/OLMo-7B-0724-hf", "size_b": 7, "family": "AI2", "arch": "Dense Transformer", "tier": "7-9B", "hardware": "5090"},
    "pythia-6.9b":      {"hf_id": "EleutherAI/pythia-6.9b", "size_b": 6.9, "family": "EleutherAI", "arch": "Dense Transformer", "tier": "7-9B", "hardware": "5090"},
    "stablelm-2-12b":   {"hf_id": "stabilityai/stablelm-2-12b", "size_b": 12, "family": "Stability AI", "arch": "Dense Transformer", "tier": "12-14B", "hardware": "5090"},
    "granite-3.0-8b":   {"hf_id": "ibm-granite/granite-3.0-8b-base", "size_b": 8, "family": "IBM", "arch": "Dense Transformer", "tier": "7-9B", "hardware": "5090"},
    # Dense Transformer (Distilled)
    "deepseek-r1-distill-qwen-7b": {"hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "size_b": 7, "family": "DeepSeek", "arch": "Dense Transformer (Distilled)", "tier": "7-9B", "hardware": "5090"},
    # MoE
    # mixtral-8x7b, jamba-1.5-mini dropped: OOM on available hardware.
    # internlm-2.5-7b / internlm-2-7b dropped: trust_remote_code incompatible with transformers 5.x
    # Pure SSM
    "falcon-mamba-7b":  {"hf_id": "tiiuae/falcon-mamba-7b", "size_b": 7, "family": "TII", "arch": "Pure SSM", "tier": "7-9B", "hardware": "5090"},
    "mamba-2.8b":       {"hf_id": "state-spaces/mamba-2.8b-hf", "size_b": 2.8, "family": "EleutherAI", "arch": "Pure SSM", "tier": "1.7-3.8B", "hardware": "5090"},
    # Small-scale
    "smollm-1.7b":      {"hf_id": "HuggingFaceTB/SmolLM-1.7B", "size_b": 1.7, "family": "HuggingFace", "arch": "Dense Transformer", "tier": "1.7-3.8B", "hardware": "5090"},
    "llama-3.2-1b":     {"hf_id": "meta-llama/Llama-3.2-1B", "size_b": 1, "family": "Meta", "arch": "Dense Transformer", "tier": "1.7-3.8B", "hardware": "5090"},
    # ── Zoo expansion (v2): 8 models for 32-model regression (8 predictors, 4:1 ratio) ──
    "llama-3.1-8b-instruct": {"hf_id": "meta-llama/Llama-3.1-8B-Instruct", "size_b": 8, "family": "Meta", "arch": "Dense Transformer (Instruct)", "tier": "7-9B", "hardware": "5090"},
    "qwen-2.5-3b":      {"hf_id": "Qwen/Qwen2.5-3B", "size_b": 3, "family": "Alibaba", "arch": "Dense Transformer", "tier": "1.7-3.8B", "hardware": "5090"},
    "gemma-2-2b":        {"hf_id": "google/gemma-2-2b", "size_b": 2, "family": "Google", "arch": "Dense Transformer", "tier": "1.7-3.8B", "hardware": "5090"},
    "mistral-7b-instruct-v0.2": {"hf_id": "mistralai/Mistral-7B-Instruct-v0.2", "size_b": 7, "family": "Mistral", "arch": "Dense Transformer (Instruct)", "tier": "7-9B", "hardware": "5090"},
    "phi-3.5-mini":      {"hf_id": "microsoft/Phi-3.5-mini-instruct", "size_b": 3.8, "family": "Microsoft", "arch": "Dense Transformer (Instruct)", "tier": "1.7-3.8B", "hardware": "5090"},
    "olmo-2-7b":         {"hf_id": "allenai/OLMo-2-1124-7B", "size_b": 7, "family": "AI2", "arch": "Dense Transformer", "tier": "7-9B", "hardware": "5090"},
    "falcon-7b":         {"hf_id": "tiiuae/falcon-7b", "size_b": 7, "family": "TII", "arch": "Dense Transformer", "tier": "7-9B", "hardware": "5090"},
    "granite-3.1-8b":    {"hf_id": "ibm-granite/granite-3.1-8b-base", "size_b": 8, "family": "IBM", "arch": "Dense Transformer", "tier": "7-9B", "hardware": "5090"},
    # ── A100/CDC tier — 27-72B ──
    "falcon-40b":        {"hf_id": "tiiuae/falcon-40b", "size_b": 40, "family": "TII", "arch": "Dense Transformer", "tier": "27-72B", "hardware": "A100"},
    "qwen-2.5-72b-instruct": {"hf_id": "Qwen/Qwen2.5-72B-Instruct", "size_b": 72, "family": "Alibaba", "arch": "Dense Transformer (Instruct)", "tier": "27-72B", "hardware": "A100"},
    # Dropped from earlier zoo drafts: llama-2-70b, qwen-2-72b, llama-3-70b-instruct,
    # command-r-v01, dbrx-base (132B MoE OOM on 4×A100), mixtral-8x22b (141B MoE OOM on 4×A100).
    # Instruct variants at scale — base→instruct delta pairs
    "gemma-2-27b-it":    {"hf_id": "google/gemma-2-27b-it", "size_b": 27, "family": "Google", "arch": "Dense Transformer (Instruct)", "tier": "27-72B", "hardware": "A100"},
    "yi-1.5-34b-chat":   {"hf_id": "01-ai/Yi-1.5-34B-Chat", "size_b": 34, "family": "01.AI", "arch": "Dense Transformer (Instruct)", "tier": "27-72B", "hardware": "A100"},
    "deepseek-v2-lite":  {"hf_id": "deepseek-ai/DeepSeek-V2-Lite", "size_b": 16, "family": "DeepSeek", "arch": "Mixture of Experts", "tier": "12-14B", "hardware": "A100"},
    # Extend 5090 families upward
    "olmo-2-13b":        {"hf_id": "allenai/OLMo-2-1124-13B", "size_b": 13, "family": "AI2", "arch": "Dense Transformer", "tier": "12-14B", "hardware": "A100"},
    # Additional models in the 1.7–14 B range
    "qwen-2.5-1.5b":         {"hf_id": "Qwen/Qwen2.5-1.5B", "size_b": 1.5, "family": "Alibaba", "arch": "Dense Transformer", "tier": "1.7-3.8B", "hardware": "5090"},
    "falcon3-7b-base":       {"hf_id": "tiiuae/Falcon3-7B-Base", "size_b": 7, "family": "TII", "arch": "Dense Transformer", "tier": "7-9B", "hardware": "5090"},
    "gemma-2-9b-it":         {"hf_id": "google/gemma-2-9b-it", "size_b": 9, "family": "Google", "arch": "Dense Transformer (Instruct)", "tier": "7-9B", "hardware": "5090"},
    "mistral-nemo-12b-base": {"hf_id": "mistralai/Mistral-Nemo-Base-2407", "size_b": 12, "family": "Mistral", "arch": "Dense Transformer", "tier": "12-14B", "hardware": "5090"},
    "qwen-2.5-14b-instruct": {"hf_id": "Qwen/Qwen2.5-14B-Instruct", "size_b": 14, "family": "Alibaba", "arch": "Dense Transformer (Instruct)", "tier": "12-14B", "hardware": "5090"},
}

# ── Corpus Configs ────────────────────────────────────────────────────────

BENCH_DIR = DATA_DIR / "bench"  # combined train+holdout for full-corpus benchmarking

COGBENCH_CORPORA = {
    "spl_cc": {
        "bench": str(BENCH_DIR / "spl_cc_bench.jsonl"),
        "target_field": "target",
        "retain_field": "retain",
        "question_template": "What is the final numerical result of the calculation described in the text?",
        "grading": "auto_number",  # exact number extraction
        "primitive": "Math",
    },
    "tom_cc": {
        "bench": str(BENCH_DIR / "tom_cc_bench.jsonl"),
        "target_field": "target",
        "retain_field": "retain",
        "question_template": "What does {CHAR_A} withhold or leave unsaid during the events described?",
        "grading": "judge",
        "primitive": "ToM",
        "placeholders": lambda pair: {
            "CHAR_A": pair.get("character_a") or (
                pair.get("characters", ["someone"])[0]
                if isinstance(pair.get("characters"), list) else "the character"),
        },
        "judge_prompt": (
            "You are a factual accuracy judge. Read the source text, then "
            "evaluate the model's answer about what the character withholds "
            "or leaves unsaid. "
            "Grade YES if the answer accurately reflects what the text "
            "describes as withheld, concealed, or unstated — OR correctly "
            "states that nothing was withheld when the text contains no "
            "such evidence. "
            "Grade NO if the answer contradicts the text, fabricates "
            "concealed information not in the text, or misses key "
            "information the text explicitly describes as hidden."
        ),
    },
    "ctr_cc": {
        "bench": str(BENCH_DIR / "ctr_cc_bench.jsonl"),
        "target_field": "target_text",
        "retain_field": "retain_text",
        "question_template": "What was the determining factor in the outcome, and why was it decisive?",
        "grading": "judge",
        "primitive": "Causal",
        "judge_prompt": (
            "You are a factual accuracy judge. Read the source text, then "
            "evaluate the model's answer about the determining factor. "
            "Grade YES if the answer (1) correctly identifies the key "
            "causal factor described in the text AND (2) accurately "
            "explains why it was decisive according to the passage. "
            "If the text reaches no conclusion about a determining factor, "
            "accept answers that correctly state this. "
            "Grade NO if the answer misidentifies the factor, omits the "
            "explanation of why it was decisive, or contradicts the text."
        ),
    },
    "mor_cc": {
        "bench": str(BENCH_DIR / "mor_cc_bench.jsonl"),
        "target_field": "target",
        "retain_field": "retain",
        "question_template": "Did {CHAR_A}'s approach treat all affected parties equally?",
        "grading": "judge",
        "primitive": "Moral",
        "placeholders": lambda pair: {
            "CHAR_A": pair.get("character_a") or
                      pair.get("archetype_a") or "the decision-maker",
        },
        "judge_prompt": (
            "You are a factual accuracy judge. Read the source text, then "
            "evaluate the model's answer about whether the character's "
            "approach treated all parties equally. "
            "Grade YES if the answer correctly accounts for every group "
            "the text identifies as affected differently — OR correctly "
            "states that no differential treatment is described. "
            "Grade NO if the answer omits a group the text explicitly "
            "identifies as differently affected, fabricates differential "
            "treatment not in the text, or contradicts the passage."
        ),
    },
    "str_cc": {
        "bench": str(BENCH_DIR / "str_cc_bench.jsonl"),
        "target_field": "target",
        "retain_field": "retain",
        "question_template": "Why did {CHAR_A} take the actions described in the passage?",
        "grading": "judge",
        "primitive": "Strategic",
        "placeholders": lambda pair: {
            "CHAR_A": pair.get("character_a") or
                      pair.get("actor_a") or "the primary actor",
        },
        "judge_prompt": (
            "You are a factual accuracy judge. Read the source text, then "
            "evaluate the model's answer about why the character acted. "
            "Grade YES if the answer correctly identifies the motivation "
            "or purpose behind the character's actions as described or "
            "clearly implied in the text. "
            "Grade NO if the answer only restates what the character did "
            "without explaining why, contradicts the text, or attributes "
            "motivations not supported by the passage."
        ),
    },
    "stp_cc": {
        "bench": str(BENCH_DIR / "stp_cc_bench.jsonl"),
        "target_field": "target",
        "retain_field": "retain",
        "question_template": "Where is the key object located at the end of the described events?",
        "grading": "auto_location",  # exact location extraction
        "primitive": "Spatial",
    },
    "core_math": {
        "bench": str(BENCH_DIR / "core_math_bench.jsonl"),
        "target_field": "target_proof",
        "retain_field": "retain_intuition",
        "question_template": "How does the passage justify its main claim?",
        "grading": "judge",
        "primitive": "Proof",
        "judge_prompt": (
            "You are a factual accuracy judge. Read the source text, then "
            "evaluate the model's answer about how the passage justifies "
            "its main claim. "
            "Grade YES if the answer correctly describes the method of "
            "reasoning AND explains at least one specific step, mechanism, "
            "or assumption from the text (e.g., which base case is used, "
            "how an analogy maps to the concept, what logical dependency "
            "connects the steps). "
            "Grade NO if the answer only labels the method without "
            "explaining a specific step or mechanism (e.g., just saying "
            "'it uses induction' or 'it uses an analogy'), mischaracterizes "
            "the reasoning, or provides only vague statements."
        ),
    },
    "null_cc": {
        "bench": str(BENCH_DIR / "null_cc_bench.jsonl"),
        "target_field": "target_formal",
        "retain_field": "retain_informal",
        "question_template": "Describe the scenario in the style presented in the text.",
        "grading": "judge",  # style adherence check
        "primitive": "Null",
        "judge_prompt": (
            "You are a style-matching judge. Read the source text and the "
            "model's answer. Grade YES if the model's response adopts the "
            "same stylistic register (formal vs informal, technical vs "
            "conversational) as the source text. Grade NO if it uses a "
            "markedly different register."
        ),
    },
    "coin": {
        "bench": str(BENCH_DIR / "coin_bench.jsonl"),
        "target_field": "target",
        "retain_field": "retain",
        "question_template": "Is the reasoning in this text logically sound regarding {CATEGORY}?",
        "grading": "judge",
        "primitive": "COIN",
        "placeholders": lambda pair: {
            "CATEGORY": pair.get("category", "the reasoning pattern").replace("_", " "),
        },
        "judge_prompt": (
            "You are a logical reasoning judge. Read the source text and "
            "the model's answer. Grade YES if the model correctly identifies "
            "whether the reasoning in the source text is logically valid or "
            "contains a fallacy or bias. Grade NO if the model misjudges "
            "the reasoning quality (calls valid reasoning fallacious, or "
            "fails to detect a present fallacy)."
        ),
    },
}


# ── Grading Functions ─────────────────────────────────────────────────────

def extract_number(text: str) -> float | None:
    """Extract the last numerical value from model output."""
    numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None


def extract_location(text: str) -> str | None:
    """Extract a location mention from model output.

    Uses the LAST matching "is (now) in/at/located at X" to capture the
    final state in multi-step scenarios, and returns only the captured
    location (group(1)) — not the verb prefix — so that target-side
    ("is now in Room 415") and retain-side ("Room 415") extractions are
    comparable. Falls back to the last facility-keyword match, then to
    a 100-char head-of-text snippet.
    """
    patterns = [
        r'(?:is\s+(?:now\s+)?(?:in|at|located\s+at))\s+(.+?)(?:\.|$)',
        r'(?:Station|Room|Cabinet|Ward|Shelf|Rack|Bay|Bay|Dock|Depot|Zone|Area)\s+\w+',
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            last = matches[-1]
            loc = last.group(1) if last.groups() else last.group(0)
            return loc.strip().lower()
    return text.strip().lower()[:100]


def grade_auto_number(model_answer: str, reference_text: str) -> bool:
    """Grade by checking if model extracts the correct number from target text."""
    ref_num = extract_number(reference_text)
    ans_num = extract_number(model_answer)
    if ref_num is None or ans_num is None:
        return False
    return abs(ref_num - ans_num) < 0.01 * max(abs(ref_num), 1)


def grade_auto_location(model_answer: str, reference_text: str) -> bool:
    """Grade by checking if model identifies the correct final location."""
    ref_loc = extract_location(reference_text)
    ans_loc = extract_location(model_answer)
    if ref_loc is None or ans_loc is None:
        return False
    return ref_loc in ans_loc or ans_loc in ref_loc


_judge_available = None  # Cached judge availability check


def _check_judges_available() -> bool:
    """Check once whether judges are reachable. Cached for the session."""
    global _judge_available
    if _judge_available is not None:
        return _judge_available
    if _judge_mode == "local":
        _judge_available = True
        return True
    panel = _get_judge_panel()
    _judge_available = panel is not None
    if not _judge_available:
        logger.warning("Judge panel not available — judge-graded corpora will be SKIPPED")
    return _judge_available


def grade_judge(model_answer: str, source_text: str, question: str,
                judge_prompt: str) -> bool | None:
    """Grade via 3-judge majority vote using JudgePanel.

    Returns True/False for correct/incorrect, None if judge failed (skip pair).
    Requires judges to be configured. No heuristic fallback — heuristic
    makes both target and retain score ~95%, producing CA ≈ 0 (garbage data).
    Configure judges via env vars: JUDGE1_MODEL, JUDGE1_BASE_URL, etc.
    """
    panel = _get_judge_panel()

    if panel is None:
        raise RuntimeError(
            "Judge panel not available. Judge-graded corpora require Ollama judges. "
            "Set JUDGE_BASE_URL=http://localhost:11434/v1 or skip judge corpora."
        )

    source_excerpt = source_text[-_JUDGE_SOURCE_MAX_CHARS:] if len(source_text) > _JUDGE_SOURCE_MAX_CHARS else source_text
    prompt_text = JUDGE_USER_TEMPLATE.format(
        source_text=source_excerpt,
        question=question,
        answer=model_answer,
        judge_instruction=judge_prompt,
    )
    try:
        flagged, reason, _ = asyncio.get_event_loop().run_until_complete(
            panel.vote(prompt_text)
        )
        return flagged
    except RuntimeError:
        # No event loop — create one
        flagged, reason, _ = asyncio.run(panel.vote(prompt_text))
        return flagged
    except Exception as e:
        logger.warning("Judge vote failed: %s", e)
        return None


# ── Core Evaluation ───────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _is_instruct_model(model_key: str, tokenizer=None) -> bool:
    """Check if a model is instruction-tuned (needs chat template wrapping).

    Checks arch string, HF ID, AND tokenizer.chat_template. Some models
    (e.g. Qwen2.5-3B) have a chat template but no 'Instruct' in the name.
    """
    cfg = COGBENCH_MODELS.get(model_key, {})
    hf_id = cfg.get("hf_id", "")
    arch = cfg.get("arch", "")
    if "Instruct" in arch or "instruct" in hf_id.lower():
        return True
    if tokenizer is not None and hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        return True
    return False


def _format_prompt(text: str, question: str, tokenizer, model_key: str) -> str:
    """Format a CogBench prompt, using chat template for instruct models."""
    user_content = (
        f"Read the following text and answer the question.\n\n"
        f"Text: {text}\n\n"
        f"Question: {question}"
    )
    if _is_instruct_model(model_key, tokenizer):
        messages = [{"role": "user", "content": user_content}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"{user_content}\n\nAnswer:"


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate an answer from a model given a prompt (single-prompt fallback)."""
    if getattr(model, "_cogbench_vllm", False):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        outs = model.generate([prompt], sp, use_tqdm=False)
        return outs[0].outputs[0].text

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=4096).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the generated tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int = 256) -> list[str]:
    """Batched generation. vLLM path uses continuous batching; HF path left-pads."""
    if not prompts:
        return []

    if getattr(model, "_cogbench_vllm", False):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        outs = model.generate(prompts, sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    # Causal LMs require left-padding for correct batched generation.
    # Save and restore original padding_side to avoid side effects.
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    inputs = tokenizer(prompts, return_tensors="pt", truncation=True,
                       max_length=4096, padding=True).to(model.device)

    tokenizer.padding_side = orig_padding_side

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Slice off the padded input (same length for all items in batch)
    input_len = inputs["input_ids"].shape[1]
    results = []
    for output in outputs:
        generated = output[input_len:]
        results.append(tokenizer.decode(generated, skip_special_tokens=True))
    return results


def _get_batch_size(model_size_b: float, arch: str = "") -> int:
    """Dynamic batch size based on model size and GPU count.

    SSM models (Mamba) MUST use batch_size=1: they have no attention mask
    mechanism, so left-padding corrupts the hidden state for shorter sequences.
    """
    if "SSM" in arch or "Mamba" in arch:
        return 1

    n_gpus = max(1, torch.cuda.device_count())
    # Scale factor: 4×A100 → 4x, single 5090 → 1x
    gpu_mult = min(n_gpus, 4)  # cap at 4x

    # Base batch sizes (tuned for single 32GB GPU)
    if "Distilled" in arch:
        base = 2
    elif "Hybrid" in arch:
        base = 2
    elif model_size_b <= 3:
        base = 32
    elif model_size_b <= 4:
        base = 16
    elif model_size_b <= 9:
        base = 8
    elif model_size_b <= 14:
        base = 4
    elif model_size_b <= 35:
        base = 2
    else:
        base = 1

    bs = min(base * gpu_mult, 64)
    logger.info("batch_size auto: %.0fB model, %d GPUs, base=%d → bs=%d",
                model_size_b, n_gpus, base, bs)
    return bs


def generate_corpus_answers(model, tokenizer, corpus_key: str, max_pairs: int = 0,
                            batch_size: int = 8,
                            model_key: str = "") -> tuple[list[dict], list[str], list[str], dict]:
    """Generate all model answers for a corpus (GPU-only, no judge calls).

    Returns (valid_pairs, target_answers, retain_answers, cfg) for deferred grading.
    """
    cfg = COGBENCH_CORPORA[corpus_key]
    bench_path = cfg["bench"]

    if not Path(bench_path).exists():
        logger.warning("SKIP %s: bench file not found at %s", corpus_key, bench_path)
        return [], [], [], cfg

    pairs = load_jsonl(bench_path)
    if max_pairs > 0:
        # Deterministic shuffle so subsets are representative, not just first-N
        from random import Random
        Random(42).shuffle(pairs)
        pairs = pairs[:max_pairs]

    question_template = cfg["question_template"]
    placeholder_fn = cfg.get("placeholders")  # optional per-pair placeholder extractor

    valid_pairs = []
    for pair in pairs:
        target_text = pair.get(cfg["target_field"], "")
        retain_text = pair.get(cfg["retain_field"], "")
        if target_text and retain_text:
            valid_pairs.append(pair)

    logger.info("  %s: %d valid pairs, batch_size=%d", corpus_key, len(valid_pairs), batch_size)

    all_target_answers = []
    all_retain_answers = []

    # Process half_bs pairs per iteration: combine target+retain into one
    # generate_batch call of batch_size prompts → halves iteration count.
    half_bs = max(1, batch_size // 2)
    for batch_start in tqdm(range(0, len(valid_pairs), half_bs),
                            desc=f"  {corpus_key} gen (bs={batch_size})", leave=False):
        batch = valid_pairs[batch_start:batch_start + half_bs]

        combined_prompts = []
        for pair in batch:
            target_text = pair[cfg["target_field"]]
            retain_text = pair[cfg["retain_field"]]
            # Resolve per-pair placeholders (e.g. {CHAR_A} in tom_cc)
            if placeholder_fn:
                question = question_template.format(**placeholder_fn(pair))
            else:
                question = question_template
            combined_prompts.append(
                _format_prompt(target_text, question, tokenizer, model_key)
            )
            combined_prompts.append(
                _format_prompt(retain_text, question, tokenizer, model_key)
            )

        combined_answers = generate_batch(model, tokenizer, combined_prompts)

        # De-interleave: even indices = target, odd indices = retain
        for i in range(0, len(combined_answers), 2):
            all_target_answers.append(combined_answers[i])
            all_retain_answers.append(combined_answers[i + 1])

    return valid_pairs, all_target_answers, all_retain_answers, cfg


def _grade_judge_batch_async(answers: list[str], source_texts: list[str],
                             question: str, judge_prompt: str) -> list[bool | None]:
    """Grade answers via Ollama judges — one answer at a time, fresh panel per call.

    Under nohup, instructor+AsyncOpenAI hang after the first batch when reusing
    async clients. Fix: create panel inside asyncio.run() and process sequentially.
    Ollama serves one model at a time anyway, so parallelism doesn't help.

    Returns list of bool|None. None = judge failure (skip pair, don't count as NO).
    """
    from judge_panel import JudgePanel, load_judge_configs_from_env

    total = len(answers)
    results: list[bool | None] = []

    for i, (answer, source_text) in enumerate(zip(answers, source_texts)):
        source_excerpt = source_text[-_JUDGE_SOURCE_MAX_CHARS:] if len(source_text) > _JUDGE_SOURCE_MAX_CHARS else source_text
        prompt_text = JUDGE_USER_TEMPLATE.format(
            source_text=source_excerpt,
            question=question,
            answer=answer,
            judge_instruction=judge_prompt,
        )

        async def _vote_one(text):
            configs = load_judge_configs_from_env(
                prefix="JUDGE",
                fallback_model=os.getenv("JUDGE_MODEL", "qwen2.5:14b"),
                fallback_base_url=os.getenv("JUDGE_BASE_URL", "http://localhost:11434/v1"),
            )
            panel = JudgePanel(
                configs=configs,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_prompt_template="{text}",
                response_model=CogBenchVerdict,
                verdict_field="correct",
                evidence_field=None,
            )
            flagged, _, _ = await panel.vote(text)
            return flagged

        try:
            flagged = asyncio.run(_vote_one(prompt_text))
        except Exception as e:
            logger.warning("Judge vote failed (len=%d): %s", len(answer), e)
            flagged = None  # skip pair, don't count as NO

        results.append(flagged)
        done = i + 1
        if done % 200 == 0 or done == total:
            valid = [r for r in results if r is not None]
            true_count = sum(valid) if valid else 0
            skip_count = len(results) - len(valid)
            logger.info("    JUDGE_PROGRESS %d/%d (%.0f%%) — %d true, %d skipped",
                        done, total, 100 * done / total, true_count, skip_count)

    return results


def grade_corpus_answers(valid_pairs: list[dict], target_answers: list[str],
                         retain_answers: list[str], cfg: dict,
                         corpus_key: str, cache_key: str | None = None) -> dict:
    """Grade pre-generated answers. No model needed — safe to call after freeing GPU."""
    from collections import defaultdict

    question_template = cfg["question_template"]
    placeholder_fn = cfg.get("placeholders")
    # For judge grading, resolve placeholders using first pair as representative
    # (the judge uses question as context, not for per-pair matching)
    if placeholder_fn and valid_pairs:
        question = question_template.format(**placeholder_fn(valid_pairs[0]))
    else:
        question = question_template
    grading = cfg["grading"]

    target_correct = 0
    retain_correct = 0
    total = 0
    empty_skipped = 0
    per_cell = defaultdict(lambda: {"target_ok": 0, "retain_ok": 0, "total": 0})

    judge_prompt = cfg.get("judge_prompt", "Grade YES if the answer is factually correct according to the source text. Grade NO otherwise.")

    if grading == "judge":
        all_answers = list(target_answers) + list(retain_answers)
        all_source_texts = [pair[cfg["target_field"]] for pair in valid_pairs] + \
                           [pair[cfg["retain_field"]] for pair in valid_pairs]
        if _judge_mode == "local":
            panel = _get_judge_panel()
            logger.info("  %s: grading %d pairs via local GPU judges...",
                        corpus_key, len(valid_pairs))
            # Chunk large payloads to stay within 32 GB judge-GPU memory.
            max_chunk = int(os.getenv("COGBENCH_GRADER_CHUNK", "5000"))
            # Per-judge batch size. core_math prompts fill the 2048 max_length,
            # so bs=8 KV cache OOMs on a 32 GiB GPU — default bs=2 there, bs=8 elsewhere.
            default_judge_bs = 2 if corpus_key == "core_math" else 8
            judge_bs = int(os.getenv("COGBENCH_JUDGE_BATCH", str(default_judge_bs)))
            if len(all_answers) > max_chunk:
                n_chunks = (len(all_answers) + max_chunk - 1) // max_chunk
                logger.info("  %s: chunking %d answers into %d chunks of up to %d (judge_bs=%d)",
                            corpus_key, len(all_answers), n_chunks, max_chunk, judge_bs)
                all_results = []
                for ci in range(n_chunks):
                    s = ci * max_chunk
                    e = min(s + max_chunk, len(all_answers))
                    ck = f"{cache_key}::chunk{ci}of{n_chunks}" if cache_key else None
                    logger.info("    chunk %d/%d: answers[%d:%d]", ci + 1, n_chunks, s, e)
                    chunk_results = panel.grade_all(
                        all_answers[s:e], all_source_texts[s:e],
                        question, judge_prompt, batch_size=judge_bs, cache_key=ck,
                    )
                    all_results.extend(chunk_results)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            else:
                all_results = panel.grade_all(all_answers, all_source_texts, question,
                                              judge_prompt, batch_size=judge_bs,
                                              cache_key=cache_key)
        else:
            logger.info("  %s: grading %d pairs via Ollama judges...",
                        corpus_key, len(valid_pairs))
            all_results = _grade_judge_batch_async(all_answers, all_source_texts, question, judge_prompt)
        target_results = all_results[:len(target_answers)]
        retain_results = all_results[len(target_answers):]

        for i, pair in enumerate(valid_pairs):
            # Empty generation → exclude from CA (else retain_acc is artifactually
            # depressed when generation yielded "" and judge grades empty as NO).
            if not (target_answers[i] or "").strip() or not (retain_answers[i] or "").strip():
                empty_skipped += 1
                continue
            t_ok = target_results[i]
            r_ok = retain_results[i]

            # Skip pairs where judge failed (None) — don't count as NO
            if t_ok is None or r_ok is None:
                logger.debug("  Skipping pair %d: judge returned None (t=%s, r=%s)", i, t_ok, r_ok)
                continue

            if t_ok:
                target_correct += 1
            if r_ok:
                retain_correct += 1
            total += 1

            category = pair.get("category", pair.get("seed_topic", "unknown"))
            difficulty = pair.get("difficulty", "unknown")
            cell_key = f"{category}|{difficulty}"
            per_cell[cell_key]["total"] += 1
            if t_ok:
                per_cell[cell_key]["target_ok"] += 1
            if r_ok:
                per_cell[cell_key]["retain_ok"] += 1
    else:
        for i, pair in enumerate(tqdm(valid_pairs,
                                      desc=f"  {corpus_key} grade", leave=False)):
            target_answer = target_answers[i]
            retain_answer = retain_answers[i]

            # Empty generation → exclude from CA (see batched-judge path above).
            if not (target_answer or "").strip() or not (retain_answer or "").strip():
                empty_skipped += 1
                continue

            if grading == "auto_number":
                t_ok = grade_auto_number(target_answer, pair[cfg["target_field"]])
                r_ok = grade_auto_number(retain_answer, pair[cfg["retain_field"]])
            elif grading == "auto_location":
                t_ok = grade_auto_location(target_answer, pair[cfg["target_field"]])
                r_ok = grade_auto_location(retain_answer, pair[cfg["retain_field"]])
            else:
                target_text = pair[cfg["target_field"]]
                retain_text = pair[cfg["retain_field"]]
                t_ok = grade_judge(target_answer, target_text, question, judge_prompt)
                r_ok = grade_judge(retain_answer, retain_text, question, judge_prompt)

            # Skip pairs where judge failed (None)
            if t_ok is None or r_ok is None:
                continue

            if t_ok:
                target_correct += 1
            if r_ok:
                retain_correct += 1
            total += 1

            category = pair.get("category", pair.get("seed_topic", "unknown"))
            difficulty = pair.get("difficulty", "unknown")
            cell_key = f"{category}|{difficulty}"
            per_cell[cell_key]["total"] += 1
            if t_ok:
                per_cell[cell_key]["target_ok"] += 1
            if r_ok:
                per_cell[cell_key]["retain_ok"] += 1

    acc_target = target_correct / total if total > 0 else 0
    acc_retain = retain_correct / total if total > 0 else 0
    contrastive_accuracy = acc_target - acc_retain

    # Build per-category breakdown
    per_category = {}
    for cell_key, counts in per_cell.items():
        cat, diff = cell_key.split("|", 1)
        n = counts["total"]
        if n == 0:
            continue
        t_acc = counts["target_ok"] / n
        r_acc = counts["retain_ok"] / n
        per_category[cell_key] = {
            "category": cat,
            "difficulty": diff,
            "n": n,
            "acc_target": round(t_acc, 4),
            "acc_retain": round(r_acc, 4),
            "contrastive_accuracy": round(t_acc - r_acc, 4),
        }

    grading = cfg["grading"]
    result = {
        "corpus": corpus_key,
        "primitive": cfg["primitive"],
        "status": "ok",
        "total_pairs": total,
        "empty_skipped": empty_skipped,
        "acc_target": round(acc_target, 4),
        "acc_retain": round(acc_retain, 4),
        "contrastive_accuracy": round(contrastive_accuracy, 4),
        "per_category": per_category,
        "grading_method": grading,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    logger.info(
        "CORPUS_RESULT corpus=%s primitive=%s CA=%.4f acc_target=%.4f "
        "acc_retain=%.4f n=%d empty_skipped=%d grading=%s categories=%d",
        corpus_key, cfg["primitive"], contrastive_accuracy,
        acc_target, acc_retain, total, empty_skipped, grading, len(per_category),
    )
    for cell_key, cell_data in per_category.items():
        logger.debug(
            "  CATEGORY %s/%s: CA=%.4f target=%.4f retain=%.4f n=%d",
            cell_data["category"], cell_data["difficulty"],
            cell_data["contrastive_accuracy"],
            cell_data["acc_target"], cell_data["acc_retain"], cell_data["n"],
        )

    return result


# ── Pending answer pipeline (generate-only / grade-pending split) ────────

def save_pending(model_key: str, corpus_key: str, valid_pairs: list[dict],
                 target_answers: list[str], retain_answers: list[str], cfg: dict):
    """Save generated answers to pending/ for deferred judge grading."""
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    path = PENDING_DIR / f"{model_key}_{corpus_key}.json"
    # Strip non-serializable keys (e.g. lambda placeholders) before saving
    cfg_serializable = {k: v for k, v in cfg.items() if callable(v) is False}
    data = {
        "model_key": model_key,
        "corpus_key": corpus_key,
        "cfg": cfg_serializable,
        "valid_pairs": valid_pairs,
        "target_answers": target_answers,
        "retain_answers": retain_answers,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(path, "w") as f:
        json.dump(data, f)
    logger.info("PENDING saved: %s (%d pairs)", path.name, len(valid_pairs))


def grade_pending_file(path: Path) -> tuple[str, str, dict] | None:
    """Grade one pending answer file. Returns (model_key, corpus_key, result) or None."""
    with open(path) as f:
        data = json.load(f)
    model_key = data["model_key"]
    corpus_key = data["corpus_key"]
    cfg = data["cfg"]
    # Restore non-serializable keys from COGBENCH_CORPORA (e.g. placeholders lambda)
    if corpus_key in COGBENCH_CORPORA:
        for k, v in COGBENCH_CORPORA[corpus_key].items():
            if k not in cfg:
                cfg[k] = v
    valid_pairs = data["valid_pairs"]
    target_answers = data["target_answers"]
    retain_answers = data["retain_answers"]

    logger.info("GRADING pending: %s on %s (%d pairs)", model_key, corpus_key, len(valid_pairs))
    cache_key = f"{model_key}_{corpus_key}"
    result = grade_corpus_answers(valid_pairs, target_answers, retain_answers, cfg,
                                  corpus_key, cache_key=cache_key)

    # Merge into model's result JSON (chokepoint: safe_write_model_json
    # preserves all existing corpora, atomic write).
    model_json = RESULTS_DIR / f"{model_key}.json"
    safe_write_model_json(
        model_json, model_key,
        {corpus_key: result},
        model_meta=COGBENCH_MODELS.get(model_key),
    )

    # Clear per-judge verdict cache now that the file is fully graded.
    cache_dir = PENDING_DIR / ".cache"
    if cache_dir.exists():
        for cf in cache_dir.glob(f"{cache_key}__*.json"):
            try:
                cf.unlink()
            except OSError:
                pass

    # Move to done
    done_dir = PENDING_DIR / "done"
    done_dir.mkdir(exist_ok=True)
    path.rename(done_dir / path.name)
    logger.info("GRADED %s/%s → CA=%.4f (moved to done/)", model_key, corpus_key,
                result.get("contrastive_accuracy", 0))
    return model_key, corpus_key, result


def run_grade_pending_loop(poll_interval: int = 30, max_idle: int = 600,
                           node_id: str = "grader"):
    """Poll pending/ for answer files and grade them.

    Uses atomic file claims so multiple grader processes can run in parallel
    without double-grading the same file. Exits after max_idle seconds idle.
    """
    idle_time = 0
    graded = 0
    logger.info("Judge grader started (poll=%ds, max_idle=%ds, node=%s)",
                poll_interval, max_idle, node_id)

    while idle_time < max_idle:
        PENDING_DIR.mkdir(parents=True, exist_ok=True)
        pending_files = sorted(PENDING_DIR.glob("*.json"))
        if not pending_files:
            time.sleep(poll_interval)
            idle_time += poll_interval
            continue

        idle_time = 0  # reset on work found
        claimed_any = False
        for pf in pending_files:
            # Atomic claim — skip if another grader process already has it
            if not _try_claim_pending(pf, node_id):
                continue
            claimed_any = True
            try:
                result = grade_pending_file(pf)
                if result:
                    graded += 1
            except FileNotFoundError:
                # Race: another grader finished this file (moved to done/) between
                # our glob snapshot and our claim. Not an error — just skip.
                logger.info("SKIP %s: already moved by another grader", pf.name)
            except Exception as e:
                logger.error("GRADE_FAILED %s: %s", pf.name, e)
                # Move to failed/ to avoid retry loop (guard against file already gone)
                if pf.exists():
                    failed_dir = PENDING_DIR / "failed"
                    failed_dir.mkdir(exist_ok=True)
                    pf.rename(failed_dir / pf.name)
            finally:
                _release_pending_claim(pf)

        if not claimed_any:
            # All pending files were claimed by other graders — wait
            time.sleep(poll_interval)
            idle_time += poll_interval

    # Clean up parallel judge models if loaded
    panel = _judge_panel
    if panel is not None and hasattr(panel, 'free_all_judges'):
        panel.free_all_judges()
    logger.info("Judge grader exiting: %d files graded, idle timeout reached", graded)


def evaluate_corpus(model, tokenizer, corpus_key: str, max_pairs: int = 0,
                    batch_size: int = 8, model_key: str = "") -> dict:
    """Legacy wrapper: generate + grade in one call. Used for auto-graded corpora."""
    valid_pairs, target_answers, retain_answers, cfg = generate_corpus_answers(
        model, tokenizer, corpus_key, max_pairs, batch_size, model_key=model_key)
    if not valid_pairs:
        return {"corpus": corpus_key, "status": "skipped"}
    save_pending(model_key, corpus_key, valid_pairs, target_answers, retain_answers, cfg)
    return grade_corpus_answers(valid_pairs, target_answers, retain_answers, cfg, corpus_key)


def _free_vram(model=None):
    """Aggressively release GPU memory between model evaluations.

    accelerate's device_map hooks keep hidden references to tensors.
    We must: delete the model → break cyclic refs with gc.collect() →
    then release the PyTorch CUDA cache.
    """
    if model is not None:
        try:
            if getattr(model, "_cogbench_vllm", False):
                # vLLM: shut down worker processes so next model can load clean
                try:
                    from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
                    destroy_model_parallel()
                    destroy_distributed_environment()
                except Exception as ve:
                    logger.warning("_free_vram: vllm teardown failed (%s: %s)", type(ve).__name__, ve)
            elif hasattr(model, "_hf_hook"):
                from accelerate.hooks import remove_hook_from_module
                remove_hook_from_module(model, recurse=True)
        except Exception as e:
            logger.warning("_free_vram: remove_hook failed (%s: %s)", type(e).__name__, e)
        del model
    try:
        gc.collect()
    except Exception as e:
        logger.warning("_free_vram: gc.collect failed (%s: %s)", type(e).__name__, e)
    # RTX 5090 can hit cudaErrorLaunchTimeout during empty_cache on fragmented VRAM.
    # A single failure here must not kill a multi-model sweep; the next model's
    # load will force a fresh allocator state.
    try:
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning("_free_vram: empty_cache failed (%s: %s) — continuing", type(e).__name__, e)
    try:
        gc.collect()
    except Exception as e:
        logger.warning("_free_vram: final gc failed (%s: %s)", type(e).__name__, e)



# ── Multi-node claim system ──────────────────────────────────────────────
# Prevents race conditions when multiple nodes run --all-models in parallel.
# Each node atomically claims a model before loading it. Claims include a
# heartbeat timestamp that is updated during evaluation so stale claims
# (from crashed nodes) can be reclaimed.

def _try_claim(model_key: str, node_id: str, timeout_minutes: int = 120) -> bool:
    """Atomically claim a model for this node. Returns True if claimed successfully.

    Uses O_CREAT|O_EXCL for atomic file creation — if two nodes race, exactly
    one wins and the other gets FileExistsError. Stale claims (heartbeat older
    than timeout_minutes) are reclaimed.
    """
    CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
    claim_path = CLAIMS_DIR / f"{model_key}.claim"

    # Check for stale claims
    if claim_path.exists():
        try:
            with open(claim_path) as f:
                claim = json.load(f)
            from datetime import datetime
            heartbeat = datetime.fromisoformat(claim.get("heartbeat", claim["started"]))
            age_minutes = (datetime.now() - heartbeat).total_seconds() / 60
            if age_minutes < timeout_minutes:
                logger.info("SKIP %s — claimed by node '%s' (%d min ago)",
                            model_key, claim.get("node", "?"), int(age_minutes))
                return False
            else:
                logger.warning("RECLAIM %s — stale claim by '%s' (heartbeat %d min ago, timeout %d)",
                               model_key, claim.get("node", "?"), int(age_minutes), timeout_minutes)
                claim_path.unlink()
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning("RECLAIM %s — corrupt claim file, removing", model_key)
            claim_path.unlink(missing_ok=True)

    # Atomic create — exactly one node wins the race
    try:
        from datetime import datetime
        fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        claim_data = json.dumps({
            "node": node_id,
            "pid": os.getpid(),
            "started": datetime.now().isoformat(),
            "heartbeat": datetime.now().isoformat(),
            "model": model_key,
        })
        os.write(fd, claim_data.encode())
        os.close(fd)
        logger.info("CLAIMED %s for node '%s' (pid %d)", model_key, node_id, os.getpid())
        return True
    except FileExistsError:
        logger.info("SKIP %s — claimed by another node between check and create", model_key)
        return False


def _update_heartbeat(model_key: str, node_id: str):
    """Update heartbeat timestamp on an active claim. Call periodically during long evaluations."""
    claim_path = CLAIMS_DIR / f"{model_key}.claim"
    if not claim_path.exists():
        return
    try:
        from datetime import datetime
        with open(claim_path) as f:
            claim = json.load(f)
        # Only update our own claims
        if claim.get("node") == node_id and claim.get("pid") == os.getpid():
            claim["heartbeat"] = datetime.now().isoformat()
            with open(claim_path, "w") as f:
                json.dump(claim, f)
    except (json.JSONDecodeError, KeyError):
        pass


def _release_claim(model_key: str):
    """Release claim after evaluation completes (results file is the permanent record)."""
    claim_path = CLAIMS_DIR / f"{model_key}.claim"
    claim_path.unlink(missing_ok=True)


def _start_heartbeat_thread(model_key: str, node_id: str, interval_minutes: int = 30):
    """Spawn a daemon thread that refreshes the claim heartbeat periodically.

    Without this, big models (falcon-40b, gemma-2-27b) exceed the 120-min
    stale-claim threshold during generation — a second node then reclaims
    the model while the original is still running. Returns (thread, stop_event)
    so the caller can stop it cleanly via _stop_heartbeat_thread().
    """
    import threading
    stop_event = threading.Event()

    def _loop():
        while not stop_event.wait(interval_minutes * 60):
            try:
                _update_heartbeat(model_key, node_id)
            except Exception as e:
                logger.warning("heartbeat thread: update failed (%s: %s)",
                               type(e).__name__, e)

    t = threading.Thread(target=_loop, name=f"heartbeat-{model_key}", daemon=True)
    t.start()
    logger.info("HEARTBEAT_THREAD started model=%s interval=%dmin", model_key, interval_minutes)
    return t, stop_event


def _stop_heartbeat_thread(thread, stop_event):
    """Signal the heartbeat thread to stop and wait briefly for it to exit."""
    if stop_event is None:
        return
    stop_event.set()
    if thread is not None:
        thread.join(timeout=5)


# ── Pending file claim system ───────────────────────────────────────────
# Prevents multiple --grade-pending processes from grading the same file.
# Same O_CREAT|O_EXCL pattern as the model claim system above.

PENDING_CLAIMS_DIR = RESULTS_DIR / "pending" / "claims"


def _try_claim_pending(pending_path: Path, node_id: str) -> bool:
    """Atomically claim a pending file for grading. Returns True if claimed."""
    PENDING_CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
    claim_path = PENDING_CLAIMS_DIR / f"{pending_path.stem}.grading"
    try:
        from datetime import datetime
        fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        claim_data = json.dumps({
            "node": node_id,
            "pid": os.getpid(),
            "started": datetime.now().isoformat(),
            "file": pending_path.name,
        })
        os.write(fd, claim_data.encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _release_pending_claim(pending_path: Path):
    """Release claim on a pending file after grading completes or fails."""
    claim_path = PENDING_CLAIMS_DIR / f"{pending_path.stem}.grading"
    claim_path.unlink(missing_ok=True)


def _hf_token():
    """Return the HF token if available (for gated model access)."""
    from huggingface_hub import get_token
    return get_token()


def _resolve_hf_path(hf_id: str) -> str:
    """Resolve a HuggingFace model ID to a local snapshot path if available.

    Some HF cache + library version combos fail to resolve the cache index.
    This falls back to the snapshot directory directly when the normal
    from_pretrained path fails in offline environments.
    """
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cache_dir = os.path.join(hf_home, "hub")
    cache_name = f"models--{hf_id.replace('/', '--')}"
    model_cache = os.path.join(cache_dir, cache_name)
    snapshots_dir = os.path.join(model_cache, "snapshots")

    if not os.path.isdir(snapshots_dir):
        return hf_id  # not cached, let from_pretrained handle it

    # Return the first (usually only) snapshot revision
    revisions = [d for d in os.listdir(snapshots_dir)
                 if os.path.isdir(os.path.join(snapshots_dir, d))]
    if revisions:
        snap_path = os.path.join(snapshots_dir, revisions[0])
        # Verify it has config.json (not a partial download)
        if os.path.exists(os.path.join(snap_path, "config.json")):
            logger.info("Resolved %s → local snapshot: %s", hf_id, snap_path)
            return snap_path

    return hf_id


def _needs_remote_code(hf_id: str) -> bool:
    """Check if a model requires trust_remote_code (no native transformers support)."""
    from transformers import AutoConfig
    resolved = _resolve_hf_path(hf_id)
    try:
        AutoConfig.from_pretrained(resolved, trust_remote_code=False, token=_hf_token())
        return False
    except Exception:
        return True


def load_model(model_key: str):
    """Load a CogBench model. If vLLM available on SM>=8, use vllm.LLM (tensor-parallel + continuous batching); else HF AutoModelForCausalLM."""
    from transformers import AutoTokenizer

    cfg = COGBENCH_MODELS[model_key]
    hf_id = cfg["hf_id"]
    resolved_path = _resolve_hf_path(hf_id)
    remote_code = _needs_remote_code(hf_id)

    token = _hf_token()
    tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=remote_code, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    arch = cfg.get("arch", "")
    if _use_vllm() and ("SSM" not in arch) and ("Mamba" not in arch):
        # MLA attention (DeepSeek-V2 family) crashes on vLLM's V1 engine with
        # AttributeError: 'MLACommonMetadataBuilder' object has no attribute
        # 'page_size'. Force the V0 engine for these models.
        if hf_id.lower().startswith("deepseek-ai/deepseek-v2"):
            os.environ["VLLM_USE_V1"] = "0"
            logger.info("VLLM_USE_V1=0 forced for %s (MLA V1 engine bug)", model_key)
        from vllm import LLM
        n_gpus = max(1, torch.cuda.device_count())
        # Pre-load VRAM reclamation — previous HF/vLLM teardowns can leave fragmented
        # memory that prevents vLLM from meeting its gpu_memory_utilization request.
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        # Pick gpu_memory_utilization based on free VRAM: aggressive for roomy GPUs,
        # conservative when previous runs left residue.
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_frac = free_bytes / total_bytes
        except Exception:
            free_frac = 1.0
        # If >80% free → ask for 0.85 (fast path). Else back off so we fit.
        if free_frac >= 0.80:
            util = 0.85
        elif free_frac >= 0.65:
            util = 0.70
        else:
            util = 0.55
        logger.info("vLLM LOAD model=%s tp=%d free_vram=%.0f%% util=%.2f",
                    model_key, n_gpus, free_frac * 100, util)
        print(f"Loading {model_key} ({hf_id}) via vLLM (tp={n_gpus}, util={util})...")
        def _looks_like_oom(exc: Exception) -> bool:
            s = str(exc).lower()
            return (
                "memory" in s
                or "oom" in s
                or "engine core" in s
                or "no available memory" in s
                or "kv cache" in s
            )

        # Per-model max_model_len cap — some small/legacy models have
        # max_position_embeddings<4096 and vLLM validation rejects our default.
        # Cap to the config-derived value so loading doesn't fail.
        model_max_model_len = 4096
        try:
            from transformers import AutoConfig
            _cfg = AutoConfig.from_pretrained(resolved_path,
                                              trust_remote_code=remote_code)
            _derived = getattr(_cfg, "max_position_embeddings", None)
            if isinstance(_derived, (int, float)) and _derived > 0:
                model_max_model_len = min(4096, int(_derived))
        except Exception as _cfg_exc:
            logger.warning("config probe for max_position_embeddings failed: %s",
                           _cfg_exc)

        llm = None
        last_exc: Exception | None = None
        for retry_util in (util, 0.50, 0.35):
            try:
                llm = LLM(
                    model=resolved_path,
                    tensor_parallel_size=n_gpus,
                    dtype="bfloat16",
                    gpu_memory_utilization=retry_util,
                    trust_remote_code=remote_code,
                    max_model_len=model_max_model_len,
                    enforce_eager=False,
                )
                if retry_util != util:
                    logger.info("vLLM LOAD ok at util=%.2f (after retry)", retry_util)
                break
            except Exception as e:
                last_exc = e
                if not _looks_like_oom(e):
                    raise
                logger.warning("vLLM load OOM at util=%.2f — backing off", retry_util)
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        if llm is None:
            raise last_exc  # type: ignore[misc]
        llm._cogbench_vllm = True
        return llm, tokenizer

    from transformers import AutoModelForCausalLM
    print(f"Loading {model_key} ({hf_id}, remote_code={remote_code})...")

    # MoE / hybrid models (e.g. Jamba) may need disk offloading when they
    # exceed GPU VRAM.  Provide an offload_folder so accelerate can spill
    # weights that don't fit.
    offload_dir = Path("/tmp/cogbench_offload") / model_key
    offload_dir.mkdir(parents=True, exist_ok=True)

    # Stale remote_code models (e.g. internlm2-7b) expect rope_scaling["type"]
    # and rope_scaling["factor"], but transformers 5.x restructured rope_scaling
    # to use "rope_type" at config top level instead. When rope_type is "default"
    # the old code just needs rope_scaling=None to take the standard RoPE path.
    # Only apply to remote_code models — native models (gemma-2, etc.) handle
    # their own rope_scaling correctly and nullifying it causes crashes.
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(resolved_path, trust_remote_code=remote_code, token=token)
    if remote_code and hasattr(model_config, "rope_scaling") and isinstance(model_config.rope_scaling, dict):
        rs = model_config.rope_scaling
        if "type" not in rs:
            rope_type = rs.get("rope_type", "default")
            if rope_type == "default":
                model_config.rope_scaling = None
            else:
                rs["type"] = rope_type
                rs.setdefault("factor", 1.0)

    model = AutoModelForCausalLM.from_pretrained(
        resolved_path, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=remote_code,
        offload_folder=str(offload_dir),
        config=model_config, token=token,
    )

    # Patch stale trust_remote_code models whose prepare_inputs_for_generation
    # breaks on transformers 5.x (cache API changes: get_max_cache_shape
    # returns tuple not int, cache_position removed, etc.).
    if remote_code and hasattr(model, "prepare_inputs_for_generation"):
        from transformers import GenerationMixin
        model.prepare_inputs_for_generation = (
            GenerationMixin.prepare_inputs_for_generation.__get__(model)
        )

    model.eval()
    return model, tokenizer


def evaluate_model(model_key: str, corpora: list[str], max_pairs: int = 0,
                   batch_size_override: int = 0,
                   generate_only: bool = False,
                   post_cogbench_heldout: bool = False,
                   heldout_only: bool = False) -> dict:
    """Evaluate one model across specified corpora.

    If generate_only=True, auto-graded corpora are graded inline but judge-graded
    corpora are saved to pending/ for the background judge worker. GPU is freed
    ASAP so the next model can load immediately.

    post_cogbench_heldout: if True, chain Set A (6 held-out correlates) +
    Set B (7 domain benchmarks) on the live model+tokenizer before freeing VRAM.
    Saves combined JSON to results/heldout_chain/heldout_chain_{model_key}.json.

    heldout_only: if True, skip all corpus generation and run only the Set A/B
    heldout chain on the loaded model. Forces post_cogbench_heldout=True.
    """
    if heldout_only:
        post_cogbench_heldout = True

    model, tokenizer = load_model(model_key)

    size_b = COGBENCH_MODELS[model_key]["size_b"]
    arch = COGBENCH_MODELS[model_key].get("arch", "")
    batch_size = batch_size_override if batch_size_override > 0 else _get_batch_size(size_b, arch)
    logger.info("Model %s (%.1fB, %s) → batch_size=%d", model_key, size_b, arch, batch_size)

    auto_corpora = []
    judge_corpora = []
    for c in corpora:
        if COGBENCH_CORPORA[c]["grading"] in ("auto_number", "auto_location"):
            auto_corpora.append(c)
        else:
            judge_corpora.append(c)

    results = {}

    if _is_instruct_model(model_key, tokenizer):
        logger.info("Instruct model detected — using chat template for prompts")

    # Phase A: auto-graded corpora — generate and grade inline (no judge needed)
    for corpus_key in auto_corpora:
        print(f"\n  Evaluating {model_key} on {corpus_key}...")
        result = evaluate_corpus(model, tokenizer, corpus_key, max_pairs,
                                 batch_size=batch_size, model_key=model_key)
        results[corpus_key] = result
        save_partial_results(model_key, results)

    # Phase B: judge-graded corpora — generate answers on GPU
    for corpus_key in judge_corpora:
        print(f"\n  Generating answers for {model_key} on {corpus_key}...")
        valid_pairs, t_ans, r_ans, cfg = generate_corpus_answers(
            model, tokenizer, corpus_key, max_pairs, batch_size=batch_size,
            model_key=model_key)
        if not valid_pairs:
            continue

        if generate_only:
            # Save to pending/ — background judge worker will grade
            save_pending(model_key, corpus_key, valid_pairs, t_ans, r_ans, cfg)
        else:
            # Legacy mode: accumulate for inline grading after model freed
            if "_deferred" not in results:
                results["_deferred"] = {}
            results["_deferred"][corpus_key] = (valid_pairs, t_ans, r_ans, cfg)

    # Phase B.5: chained held-out + domain benchmarks on live model (optional)
    if post_cogbench_heldout:
        try:
            from evaluation.scripts.cogbench_chain import run_all_heldout
            # Heldout benchmarks (eval_bigtom.py, eval_balanced_copa.py, ...) call
            # HF model.generate() directly and require an HF AutoModel. If we loaded
            # via vLLM for corpus gen, tear it down and reload as HF for heldout.
            if getattr(model, "_cogbench_vllm", False):
                logger.info("HELDOUT_CHAIN: tearing down vLLM and reloading as HF for benchmark compatibility")
                _free_vram(model)
                model = None
                os.environ["COGBENCH_NO_VLLM"] = "1"
                try:
                    model, tokenizer = load_model(model_key)
                finally:
                    os.environ.pop("COGBENCH_NO_VLLM", None)
            logger.info("HELDOUT_CHAIN_START model=%s", model_key)
            results["_heldout_chain"] = run_all_heldout(model, tokenizer, model_key)
            logger.info("HELDOUT_CHAIN_DONE model=%s", model_key)
        except Exception as e:
            logger.error("HELDOUT_CHAIN_FAIL model=%s err=%s", model_key, e)
            import traceback
            logger.error("%s", traceback.format_exc())

    # Phase C: free GPU immediately
    _free_vram(model)
    model = None

    # Phase D: inline judge grading (only in legacy non-pipeline mode)
    deferred = results.pop("_deferred", {})
    if deferred:
        logger.info("Inline judge grading for %d corpora...", len(deferred))
        for corpus_key, (valid_pairs, t_ans, r_ans, cfg) in deferred.items():
            print(f"\n  Judge-grading {model_key} on {corpus_key}...")
            result = grade_corpus_answers(valid_pairs, t_ans, r_ans, cfg, corpus_key)
            results[corpus_key] = result

    return results


def save_partial_results(model_key: str, results: dict):
    """Idempotent per-model JSON + per-category CSV write (no master-CSV append).

    Called after each auto-graded corpus in evaluate_model so RAM-only results
    survive a mid-run crash. Routes through safe_write_model_json so a
    partial-corpus run can never wipe previously-graded corpora on disk.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{model_key}.json"
    safe_write_model_json(path, model_key, results, model_meta=COGBENCH_MODELS[model_key])
    for corpus_key, r in results.items():
        if not isinstance(r, dict) or r.get("status") != "ok" or not r.get("per_category"):
            continue
        cat_csv = RESULTS_DIR / f"percategory_{model_key}_{corpus_key}.csv"
        cat_fields = ["model", "corpus", "primitive", "category", "difficulty",
                      "n", "acc_target", "acc_retain", "contrastive_accuracy"]
        with open(cat_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cat_fields)
            writer.writeheader()
            for cell_data in r["per_category"].values():
                writer.writerow({
                    "model": model_key, "corpus": corpus_key,
                    "primitive": r["primitive"], **cell_data,
                })
    logger.info("Partial save: %s (%d corpora persisted)", path.name, len(results))


def save_model_results(model_key: str, results: dict):
    """Save per-model results to JSON + append to master CSV (Table 5).

    Uses atomic write: tmp file + fsync + rename. Also merges with any existing
    file on disk to guard against sync races that might swap in a stale copy
    between our read and write (e.g. external rsync daemon).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Per-model JSON — chokepoint safe_write_model_json unions with disk,
    # atomic tmp+fsync+rename, preserves all prior corpora.
    path = RESULTS_DIR / f"{model_key}.json"
    payload = safe_write_model_json(path, model_key, results,
                                    model_meta=COGBENCH_MODELS[model_key])
    merged_results = payload["results"]
    logger.info("Saved per-model JSON: %s (corpora=%d, heldout=%s)",
                path, len([k for k in merged_results if not k.startswith("_")]),
                "present" if "_heldout_chain" in merged_results else "absent")

    # Post-save verification — catch concurrent overwrite within the save window.
    # If _heldout_chain is present in-memory but missing on disk, re-save and alert.
    if "_heldout_chain" in merged_results:
        for attempt in range(3):
            try:
                with open(path) as rf:
                    on_disk_now = json.load(rf).get("results", {})
            except (json.JSONDecodeError, OSError):
                on_disk_now = {}
            if "_heldout_chain" in on_disk_now:
                break
            logger.warning(
                "POST_SAVE_VERIFY model=%s — _heldout_chain disappeared from disk "
                "(attempt %d/3); re-saving. External writer racing with us?",
                model_key, attempt + 1,
            )
            safe_write_model_json(path, model_key, merged_results,
                                  model_meta=COGBENCH_MODELS[model_key])
        else:
            logger.error(
                "POST_SAVE_VERIFY_FAILED model=%s — could not retain "
                "_heldout_chain on disk after 3 save attempts; "
                "results in memory, check tripwire log for culprit.",
                model_key,
            )

    # 2. Append to master CSV for Table 5 (primitive profiles)
    master_csv = RESULTS_DIR / "table5_master.csv"
    primitives = ["Math", "ToM", "Causal", "Moral", "Strategic", "Spatial", "Proof", "Null", "COIN"]
    row = {"model": model_key, "size_b": COGBENCH_MODELS[model_key]["size_b"],
           "arch": COGBENCH_MODELS[model_key]["arch"],
           "tier": COGBENCH_MODELS[model_key]["tier"],
           "family": COGBENCH_MODELS[model_key]["family"]}
    ca_vals = []
    for corpus_key, r in results.items():
        if r.get("status") == "ok":
            row[r["primitive"]] = r["contrastive_accuracy"]
            row[f"{r['primitive']}_target"] = r["acc_target"]
            row[f"{r['primitive']}_retain"] = r["acc_retain"]
            row[f"{r['primitive']}_n"] = r["total_pairs"]
            ca_vals.append(r["contrastive_accuracy"])
    row["Avg"] = round(np.mean(ca_vals), 4) if ca_vals else 0

    fieldnames = (["model", "size_b", "arch", "tier", "family"] + primitives +
                  [f"{p}_target" for p in primitives] +
                  [f"{p}_retain" for p in primitives] +
                  [f"{p}_n" for p in primitives] + ["Avg"])
    write_header = not master_csv.exists()
    with open(master_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.info("Appended to master CSV: %s", master_csv)

    # 2b. Update central manifest (best-effort — don't fail eval if manifest broken)
    try:
        import subprocess as _sp
        _manifest_script = Path(__file__).resolve().parent.parent.parent / "results" / "update_manifest.py"
        if _manifest_script.exists():
            _sp.run([sys.executable, str(_manifest_script), "--update"],
                    capture_output=True, timeout=30)
            logger.info("Central manifest updated")
    except Exception as _e:
        logger.debug("Manifest update skipped: %s", _e)

    # 3. Save per-category CSV for Tables 16/17 (category × difficulty)
    for corpus_key, r in results.items():
        if r.get("status") != "ok" or not r.get("per_category"):
            continue
        cat_csv = RESULTS_DIR / f"percategory_{model_key}_{corpus_key}.csv"
        cat_fields = ["model", "corpus", "primitive", "category", "difficulty",
                      "n", "acc_target", "acc_retain", "contrastive_accuracy"]
        with open(cat_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cat_fields)
            writer.writeheader()
            for cell_data in r["per_category"].values():
                writer.writerow({
                    "model": model_key, "corpus": corpus_key,
                    "primitive": r["primitive"], **cell_data,
                })
        logger.debug("Saved per-category CSV: %s", cat_csv)


# ── Post-hoc Analysis (Tables 6, 8, 9) ──────────────────────────────────

def run_analysis():
    """Generate Tables 6, 8, 9 from saved per-model result JSONs."""
    import csv

    # Load all results
    all_results = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        model = data["model"]
        all_results[model] = data

    if len(all_results) < 3:
        print(f"Only {len(all_results)} model results found. Need more for analysis.")
        return

    primitives = ["Math", "ToM", "Causal", "Moral", "Strategic", "Spatial", "Proof", "Null", "COIN"]
    corpus_to_primitive = {k: v["primitive"] for k, v in COGBENCH_CORPORA.items()}

    # Build Table 5 matrix
    table5 = []
    for model_key, data in all_results.items():
        row = {"model": model_key}
        for corpus_key, prim in corpus_to_primitive.items():
            r = data.get("results", {}).get(corpus_key, {})
            row[prim] = r.get("contrastive_accuracy", None)
        row["Avg"] = np.mean([v for v in row.values() if isinstance(v, (int, float)) and v is not None])
        table5.append(row)

    # Write Table 5 CSV
    table5_path = RESULTS_DIR / "table5_primitive_profiles.csv"
    with open(table5_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model"] + primitives + ["Avg"])
        writer.writeheader()
        for row in table5:
            writer.writerow(row)
    print(f"\n[Table 5] Primitive Profiles → {table5_path}")

    # Table 6: Inter-Primitive Correlation Matrix
    scores = {p: [] for p in primitives}
    for row in table5:
        for p in primitives:
            scores[p].append(row.get(p, 0) or 0)

    print(f"\n[Table 6] Inter-Primitive Correlations (n={len(table5)}):")
    print(f"{'':>12}", end="")
    for p in primitives:
        print(f"{p:>10}", end="")
    print()
    for p1 in primitives:
        print(f"{p1:>12}", end="")
        for p2 in primitives:
            r = np.corrcoef(scores[p1], scores[p2])[0, 1]
            print(f"{r:>10.3f}", end="")
        print()

    corr_path = RESULTS_DIR / "table6_correlations.json"
    corr_matrix = {}
    for p1 in primitives:
        corr_matrix[p1] = {}
        for p2 in primitives:
            corr_matrix[p1][p2] = round(float(np.corrcoef(scores[p1], scores[p2])[0, 1]), 4)
    with open(corr_path, "w") as f:
        json.dump(corr_matrix, f, indent=2)

    # Table 8: By Scale Tier
    print(f"\n[Table 8] Primitive Scores by Scale:")
    tiers = ["1.7-3.8B", "7-9B", "12-14B", "27-72B"]
    table8_rows = []
    for tier in tiers:
        tier_models = [m for m, d in all_results.items()
                       if d.get("tier") == tier]
        if not tier_models:
            continue
        avgs = {}
        for p in primitives:
            vals = [all_results[m].get("results", {}).get(
                next(k for k, v in corpus_to_primitive.items() if v == p), {}
            ).get("contrastive_accuracy", 0) or 0 for m in tier_models]
            avgs[p] = np.mean(vals)
        print(f"  {tier} (n={len(tier_models)}): " +
              " | ".join(f"{p}={avgs[p]:.3f}" for p in primitives))
        table8_rows.append({"tier": tier, "n": len(tier_models),
                            **{p: round(avgs[p], 4) for p in primitives}})

    # Save Table 8 CSV + JSON
    table8_csv = RESULTS_DIR / "table8_scaling.csv"
    with open(table8_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tier", "n"] + primitives)
        writer.writeheader()
        for row in table8_rows:
            writer.writerow(row)
    table8_json = RESULTS_DIR / "table8_scaling.json"
    with open(table8_json, "w") as f:
        json.dump(table8_rows, f, indent=2)
    logger.info("Saved Table 8: %s + %s", table8_csv, table8_json)

    # Table 9: By Architecture Type
    print(f"\n[Table 9] Primitive Scores by Architecture:")
    arch_types = set(d.get("arch", "") for d in all_results.values())
    table9_rows = []
    for arch in sorted(arch_types):
        arch_models = [m for m, d in all_results.items() if d.get("arch") == arch]
        if not arch_models:
            continue
        avgs = {}
        for p in primitives:
            vals = [all_results[m].get("results", {}).get(
                next(k for k, v in corpus_to_primitive.items() if v == p), {}
            ).get("contrastive_accuracy", 0) or 0 for m in arch_models]
            avgs[p] = np.mean(vals)
        print(f"  {arch} (n={len(arch_models)}): " +
              " | ".join(f"{p}={avgs[p]:.3f}" for p in primitives))
        table9_rows.append({"architecture": arch, "n": len(arch_models),
                            **{p: round(avgs[p], 4) for p in primitives}})

    # Save Table 9 CSV + JSON
    table9_csv = RESULTS_DIR / "table9_architecture.csv"
    with open(table9_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["architecture", "n"] + primitives)
        writer.writeheader()
        for row in table9_rows:
            writer.writerow(row)
    table9_json = RESULTS_DIR / "table9_architecture.json"
    with open(table9_json, "w") as f:
        json.dump(table9_rows, f, indent=2)
    logger.info("Saved Table 9: %s + %s", table9_csv, table9_json)

    logger.info("Analysis complete. Results in %s/", RESULTS_DIR)


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CogBench Contrastive Accuracy Evaluator")
    parser.add_argument("--model", type=str, choices=list(COGBENCH_MODELS.keys()) + ["all"],
                        help="Model to evaluate")
    parser.add_argument("--corpus", type=str, choices=list(COGBENCH_CORPORA.keys()),
                        help="Single corpus to evaluate")
    parser.add_argument("--corpora", type=str,
                        help="Comma-separated list of corpora to evaluate (e.g. spl_cc,tom_cc,ctr_cc)")
    parser.add_argument("--all-corpora", action="store_true",
                        help="Evaluate on all available corpora")
    parser.add_argument("--all-models", action="store_true",
                        help="Evaluate all 24 models (sequential)")
    parser.add_argument("--hardware", type=str, choices=["5090", "A100"],
                        help="Only run models for this hardware tier")
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="Limit pairs per corpus (0=all, useful for testing)")
    parser.add_argument("--max-params", type=float, default=0,
                        help="Skip models larger than N billion params (0=no limit)")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Override batch size (0=auto based on model size)")
    parser.add_argument("--analysis", action="store_true",
                        help="Run post-hoc analysis (Tables 6-9) from saved results")
    parser.add_argument("--generate-only", action="store_true",
                        help="GPU generation only — judge corpora saved to pending/ for background grader")
    parser.add_argument("--post-cogbench-heldout", action="store_true",
                        help="Chain Set A (6 held-out correlates) + Set B (7 domain benchmarks) "
                             "on the live model before freeing VRAM. Saves combined JSON to "
                             "results/heldout_chain/heldout_chain_{model_key}.json.")
    parser.add_argument("--heldout-only", action="store_true",
                        help="Skip corpus generation entirely; load model and run only the "
                             "Set A/B heldout chain. Useful for models whose corpora were "
                             "already generated (e.g. prior wall-clock-killed runs). "
                             "Implies --post-cogbench-heldout.")
    parser.add_argument("--grade-pending", action="store_true",
                        help="Run background judge grader: poll pending/ and grade via Ollama")
    parser.add_argument("--grade-poll-interval", type=int, default=30,
                        help="Seconds between pending/ polls (default: 30)")
    parser.add_argument("--grade-max-idle", type=int, default=1800,
                        help="Exit grader after N seconds with no pending files (default: 1800)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path to consolidated log file (default: auto-timestamped in logs/)")
    parser.add_argument("--judge-mode", type=str, choices=["ollama", "local"],
                        default="ollama",
                        help="Judge backend: 'ollama' (HTTP API) or 'local' (load on GPU directly)")
    parser.add_argument("--judge-device", type=str, default="cuda:0",
                        help="GPU device for local judges in single-GPU mode (default: cuda:0)")
    parser.add_argument("--judge-devices", type=str, default=None,
                        help="Comma-separated GPU devices for parallel multi-GPU judging "
                             "(e.g. cuda:0,cuda:1,cuda:2). One device per judge model. "
                             "Overrides --judge-device. All 3 judges load simultaneously.")
    parser.add_argument("--judge-tier", type=str, choices=["small", "large"],
                        default="large",
                        help="Judge model tier: 'large' (14B+9B+12B, default for eval) "
                             "or 'small' (7-9B, for revalidation/cleaning)")
    # Multi-node coordination
    parser.add_argument("--node", type=str, default=None,
                        help="Node identifier for multi-node claim system (default: hostname)")
    parser.add_argument("--claim-timeout", type=int, default=120,
                        help="Minutes before a stale claim can be reclaimed (default: 120)")
    parser.add_argument("--no-claim", action="store_true",
                        help="Disable claim system (single-node mode, no coordination)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated list of model keys for hard partitioning "
                             "(e.g. llama-3-8b,qwen-2.5-7b,mistral-7b). "
                             "Overrides --all-models and --hardware.")
    args = parser.parse_args()

    # Apply judge mode globals (must happen before any judge availability checks)
    global _judge_mode, _judge_device, _judge_devices, LOCAL_JUDGE_MODELS
    _judge_mode = args.judge_mode
    if args.judge_tier == "small":
        LOCAL_JUDGE_MODELS = LOCAL_JUDGE_MODELS_SMALL
        logger.info("Judge tier: SMALL (7-9B, revalidation/cleaning)")
    else:
        LOCAL_JUDGE_MODELS = LOCAL_JUDGE_MODELS_LARGE
        logger.info("Judge tier: LARGE (9-14B, evaluation grading)")
    if args.judge_devices:
        _judge_devices = [d.strip() for d in args.judge_devices.split(",")]
        logger.info("Multi-GPU judging: %d devices → %s", len(_judge_devices), _judge_devices)
    _judge_device = args.judge_device

    # Resolve node identifier for claims
    import socket
    node_id = args.node or socket.gethostname()

    # Set up consolidated logging — one file gets logger output + tqdm + print
    if args.log_file:
        log_path = Path(args.log_file)
    else:
        from datetime import datetime
        suffix = "grader" if args.grade_pending else "gen" if args.generate_only else "eval"
        log_path = LOG_DIR / f"cogbench_{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _setup_logging(log_path)
    logger.info("Consolidated log: %s", log_path)

    if args.analysis:
        run_analysis()
        return

    # ── Background judge grader mode ──
    if args.grade_pending:
        run_grade_pending_loop(
            poll_interval=args.grade_poll_interval,
            max_idle=args.grade_max_idle,
            node_id=f"{node_id}-grader-{os.getpid()}",
        )
        return

    # Determine corpora to evaluate
    if args.all_corpora:
        corpora = list(COGBENCH_CORPORA.keys())
    elif args.corpora:
        corpora = [c.strip() for c in args.corpora.split(",")]
        invalid = [c for c in corpora if c not in COGBENCH_CORPORA]
        if invalid:
            parser.error(f"Unknown corpora: {invalid}. Valid: {list(COGBENCH_CORPORA.keys())}")
    elif args.corpus:
        corpora = [args.corpus]
    else:
        parser.error("Specify --corpus, --corpora, --all-corpora, --analysis, or --grade-pending")

    # Filter to available corpora (bench file exists + enough pairs)
    MIN_BENCH_PAIRS = 400  # skip corpora with too few pairs
    available = []
    for c in corpora:
        bench = Path(COGBENCH_CORPORA[c]["bench"])
        if not bench.exists():
            continue
        n_lines = sum(1 for _ in open(bench))
        if n_lines < MIN_BENCH_PAIRS:
            logger.warning("SKIP %s — only %d bench pairs (min %d)", c, n_lines, MIN_BENCH_PAIRS)
            continue
        available.append(c)
    skipped = [c for c in corpora if c not in available]
    if skipped:
        print(f"Skipping corpora: {skipped}")

    # In generate-only mode, keep judge corpora even if judges aren't local —
    # the background grader on another machine will handle them.
    if not args.generate_only:
        judges_ok = _check_judges_available()
        if not judges_ok:
            judge_corpora = [c for c in available if COGBENCH_CORPORA[c]["grading"] == "judge"]
            if judge_corpora:
                logger.warning("Skipping judge-graded corpora (no judges): %s", judge_corpora)
                available = [c for c in available if COGBENCH_CORPORA[c]["grading"] != "judge"]

    if not available:
        print("No corpora available. Check bench files and judge availability.")
        return

    # Determine models to evaluate
    if args.models:
        # Hard partitioning: explicit model list for this node
        models = [m.strip() for m in args.models.split(",")]
        invalid = [m for m in models if m not in COGBENCH_MODELS]
        if invalid:
            parser.error(f"Unknown models: {invalid}. Valid: {list(COGBENCH_MODELS.keys())}")
        logger.info("Hard partition: %d models assigned to this node", len(models))
    elif args.all_models:
        models = list(COGBENCH_MODELS.keys())
        if args.hardware:
            models = [m for m in models if COGBENCH_MODELS[m]["hardware"] == args.hardware]
    elif args.model:
        models = [args.model] if args.model != "all" else list(COGBENCH_MODELS.keys())
    else:
        parser.error("Specify --model, --models, or --all-models")

    # Filter by max params (skip models too large for fp16 on available VRAM)
    if args.max_params > 0:
        skipped_models = [m for m in models if COGBENCH_MODELS[m]["size_b"] > args.max_params]
        models = [m for m in models if COGBENCH_MODELS[m]["size_b"] <= args.max_params]
        if skipped_models:
            logger.info("Skipping %d models above %.0fB params: %s",
                        len(skipped_models), args.max_params, skipped_models)

    mode = "generate-only" if args.generate_only else "full"
    logger.info("CogBench %s: %d models × %d corpora", mode, len(models), len(available))
    logger.info("Models: %s", models)
    logger.info("Corpora: %s", available)

    # ── wandb init ──
    import wandb
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "cogbench-eval"),
        group="eval",
        name=f"cogbench_{mode}_{len(models)}m_{len(available)}c",
        config={
            "mode": mode,
            "models": models,
            "corpora": available,
            "max_params": args.max_params,
            "hardware": args.hardware,
            "batch_size_override": args.batch_size,
            "post_cogbench_heldout": args.post_cogbench_heldout,
        },
        tags=["cogbench", "paper2"],
    )

    use_claims = not args.no_claim
    if use_claims:
        logger.info("Claim system ENABLED: node='%s', timeout=%d min", node_id, args.claim_timeout)
    else:
        logger.info("Claim system DISABLED (--no-claim)")

    eval_start = time.time()
    failed_models: list[str] = []
    for model_key in models:
        # ── Multi-node claim: atomically reserve this model before any work ──
        if use_claims:
            if not _try_claim(model_key, node_id, timeout_minutes=args.claim_timeout):
                continue  # Another node owns this model

        # Check which corpora still need evaluation (auto-graded only in generate-only mode;
        # judge corpora are checked via pending/ existence instead)
        existing = RESULTS_DIR / f"{model_key}.json"
        prior_results = {}
        if existing.exists():
            with open(existing) as f:
                prior = json.load(f)
            prior_results = prior.get("results", {})

        # Also check pending/ for in-flight judge corpora
        pending_corpora = set()
        if PENDING_DIR.exists():
            for pf in PENDING_DIR.glob(f"{model_key}_*.json"):
                c = pf.stem.replace(f"{model_key}_", "")
                pending_corpora.add(c)

        prior_corpora = set(prior_results.keys()) | pending_corpora
        remaining = [c for c in available if c not in prior_corpora]
        if args.heldout_only:
            # Heldout-only mode: force-run heldout. If all corpora present, skip
            # gen entirely; otherwise generate the missing ones first, then heldout.
            if prior_results.get("_heldout_chain"):
                logger.info("SKIP %s — _heldout_chain already present", model_key)
                if use_claims:
                    _release_claim(model_key)
                continue
            if not remaining:
                logger.info("HELDOUT_ONLY %s — all %d corpora present, running Set A/B only",
                            model_key, len(available))
            else:
                logger.info("HELDOUT_ONLY %s — %d corpora missing %s, gen first then Set A/B",
                            model_key, len(remaining), remaining)
        elif not remaining:
            logger.info("SKIP %s — results/pending already exist for %s", model_key, available)
            if use_claims:
                _release_claim(model_key)
            continue
        elif prior_corpora & set(available):
            logger.info("RESUME %s — have %s, need %s", model_key,
                        list(prior_corpora & set(available)), remaining)

        model_start = time.time()
        # Per-model log file (in addition to the global --log-file) so each
        # model's output is independently greppable.
        _attach_per_model_log(model_key, node_id)
        logger.info("=" * 60)
        logger.info("EVAL_START model=%s corpora=%s generate_only=%s node=%s",
                    model_key, remaining, args.generate_only, node_id)

        # Start heartbeat thread so long generations don't exceed stale-claim threshold
        hb_thread, hb_stop = (None, None)
        if use_claims:
            hb_thread, hb_stop = _start_heartbeat_thread(model_key, node_id, interval_minutes=30)

        try:
            results = evaluate_model(model_key, remaining, max_pairs=args.max_pairs,
                                    batch_size_override=args.batch_size,
                                    generate_only=args.generate_only,
                                    post_cogbench_heldout=args.post_cogbench_heldout,
                                    heldout_only=args.heldout_only)
            # Final heartbeat refresh before grading phase (belt-and-suspenders)
            if use_claims:
                _update_heartbeat(model_key, node_id)

            # Merge auto-graded results with prior (judge results come via grader)
            merged = {**prior_results, **results}
            if results:  # only save if we got auto-graded results
                save_model_results(model_key, merged)

            model_elapsed = time.time() - model_start
            log_row = {"model": model_key, "wall_time_s": round(model_elapsed, 1)}
            for corpus_key, r in results.items():
                if r.get("status") == "ok":
                    prim = r["primitive"]
                    log_row[f"CA_{prim}"] = r["contrastive_accuracy"]
                    log_row[f"acc_target_{prim}"] = r["acc_target"]
                    log_row[f"acc_retain_{prim}"] = r["acc_retain"]
                    logger.info(
                        "  RESULT model=%s corpus=%s primitive=%s CA=%.4f "
                        "target=%.4f retain=%.4f n=%d",
                        model_key, corpus_key, r["primitive"],
                        r["contrastive_accuracy"],
                        r["acc_target"], r["acc_retain"], r["total_pairs"],
                    )
            wandb.log(log_row)
            logger.info("EVAL_DONE model=%s wall_time=%.1fs", model_key, model_elapsed)
        except Exception as e:
            import traceback
            logger.error("EVAL_FAILED model=%s error=%s", model_key, e)
            logger.error("Traceback:\n%s", traceback.format_exc())
            _free_vram()  # no model ref available — gc.collect will catch leaked tensors
            failed_models.append(model_key)
            continue
        finally:
            _stop_heartbeat_thread(hb_thread, hb_stop)
            if use_claims:
                _release_claim(model_key)
            _detach_per_model_log()

    total_elapsed = time.time() - eval_start
    logger.info("All evaluations complete in %.1fs", total_elapsed)
    if args.generate_only:
        n_pending = len(list(PENDING_DIR.glob("*.json"))) if PENDING_DIR.exists() else 0
        logger.info("Pending judge files: %d (background grader will process them)", n_pending)
    else:
        logger.info("Run --analysis to generate Tables 6-9.")
    wandb.log({"total_elapsed_s": round(total_elapsed, 1)})
    wandb.finish()
    if failed_models:
        logger.error("EXIT_FAILED n_failed=%d models=%s", len(failed_models), failed_models)
        sys.exit(1)


if __name__ == "__main__":
    main()
