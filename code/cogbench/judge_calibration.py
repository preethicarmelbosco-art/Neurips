"""Judge Calibration Audit — Asymmetric Bias Measurement.

Measures whether the 2-judge unanimous panel (Qwen-2.5-14B-Instruct +
Mistral-Nemo-Instruct-2407) grades cognitively-rich target answers more
leniently than bare retain answers. Two-phase workflow:

  Phase 1 (--generate-gold): use a local open-weights LLM to generate
    pair-specific gold answers for the target side. Retain-side golds are
    trivial templates ("nothing is withheld", etc.). Saved under
    results/judge_calibration/gold_answers/.

  Phase 2 (--calibrate): submit gold answers through the judge panel.
    Compute δ = accept_target - accept_retain per judge and panel unanimity.

If |δ| < 5 pp for the panel, asymmetric bias is bounded. If δ > 5 pp,
we report CA_corrected = CA_raw - δ.

All generation and judging is done with local open-weights models; no
commercial APIs are required.

Usage:
    # Phase 1: Generate gold answers with a local HF model
    python judge_calibration.py --generate-gold --all-corpora

    # Phase 1 dry-run: show prompts without running the model
    python judge_calibration.py --generate-gold --all-corpora --dry-run

    # Phase 2a: Run calibration via local HF judges (matches cogbench_eval.py panel)
    python judge_calibration.py --calibrate --all-corpora \\
        --judge-mode local --judge-devices cuda:2,cuda:3

    # Phase 2b: Alternative — Ollama judges (needs qwen2.5:14b-instruct + mistral-nemo)
    python judge_calibration.py --calibrate --all-corpora --judge-mode ollama

    # Both phases in one shot
    python judge_calibration.py --generate-gold --calibrate --all-corpora \\
        --judge-mode local --judge-devices cuda:2,cuda:3

Requires:
  Phase 1: transformers + torch, fp16 weights for a small instruct LLM
  Phase 2 (local): transformers + torch, fp16 weights for the two judges
  Phase 2 (ollama): qwen2.5:14b-instruct + mistral-nemo-instruct-2407 pulled
"""

import argparse
import csv
import json
import logging
import os
import random
import time
from pathlib import Path

import wandb

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "judge_calibration"
GOLD_DIR = RESULTS_DIR / "gold_answers"

# Judge-graded corpora to calibrate (excludes auto-graded: spl_cc, stp_cc)
JUDGE_GRADED_CORPORA = ["tom_cc", "ctr_cc", "mor_cc", "str_cc", "core_math", "null_cc"]

N_SAMPLE = 50  # pairs per corpus

# Bench data lives at <release_root>/data/bench/
BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "bench"


# ═══════════════════════════════════════════════════════════════════════════
# Corpus Configuration
# ═══════════════════════════════════════════════════════════════════════════

CORPUS_CFG = {
    "tom_cc": {
        "bench": str(BENCH_DIR / "tom_cc_bench.jsonl"),
        "target_field": "target", "retain_field": "retain",
        "question": "What does {character_a} withhold or leave unsaid during the events described?",
        "char_field": "character_a", "char_default": "the character",
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
        "retain_gold_template": (
            "Nothing is withheld or left unsaid. The text describes all "
            "actions and events openly, with no indication that {char} "
            "conceals any information."
        ),
        "instruction": (
            "Read this text carefully. The question is: 'What does {char} "
            "withhold or leave unsaid during the events described?'\n\n"
            "Write a SHORT (2-3 sentence) factually correct answer that "
            "identifies exactly what is concealed, hidden, or left unsaid "
            "by {char} according to this text. Be specific — cite the "
            "actual information that is withheld. Do NOT add interpretation "
            "beyond what the text states."
        ),
    },
    "ctr_cc": {
        "bench": str(BENCH_DIR / "ctr_cc_bench.jsonl"),
        "target_field": "target_text", "retain_field": "retain_text",
        "question": "What was the determining factor in the outcome, and why was it decisive?",
        "char_field": None, "char_default": None,
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
        "retain_gold_template": (
            "The text does not identify a single determining factor. It "
            "describes events and outcomes as a sequence of occurrences "
            "without establishing a clear causal link between any specific "
            "condition and the result."
        ),
        "instruction": (
            "Read this text carefully. The question is: 'What was the "
            "determining factor in the outcome, and why was it decisive?'\n\n"
            "Write a SHORT (2-3 sentence) factually correct answer that "
            "identifies the key causal factor and explains why it was "
            "decisive, based solely on what the text states. Be specific."
        ),
    },
    "mor_cc": {
        "bench": str(BENCH_DIR / "mor_cc_bench.jsonl"),
        "target_field": "target", "retain_field": "retain",
        "question": "Did {character_a}'s approach treat all affected parties equally?",
        "char_field": "character_a", "char_default": "the decision-maker",
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
        "retain_gold_template": (
            "The text does not describe differential treatment of affected "
            "parties. Based on the passage, {char}'s approach does not "
            "show evidence of treating groups unequally."
        ),
        "instruction": (
            "Read this text carefully. The question is: 'Did {char}'s "
            "approach treat all affected parties equally?'\n\n"
            "Write a SHORT (2-3 sentence) factually correct answer. "
            "Identify which groups were affected differently and how, "
            "based solely on what the text states."
        ),
    },
    "str_cc": {
        "bench": str(BENCH_DIR / "str_cc_bench.jsonl"),
        "target_field": "target", "retain_field": "retain",
        "question": "Why did {character_a} take the actions described in the passage?",
        "char_field": "character_a", "char_default": "the primary actor",
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
        "retain_gold_template": (
            "The text describes what {char} did but does not explain a "
            "strategic motivation or purpose behind the actions. The events "
            "are presented as a straightforward factual account."
        ),
        "instruction": (
            "Read this text carefully. The question is: 'Why did {char} "
            "take the actions described in the passage?'\n\n"
            "Write a SHORT (2-3 sentence) factually correct answer that "
            "identifies the strategic motivation or purpose behind the "
            "character's actions, based solely on what the text states."
        ),
    },
    "core_math": {
        "bench": str(BENCH_DIR / "core_math_bench.jsonl"),
        "target_field": "target_proof", "retain_field": "retain_intuition",
        "question": "How does the passage justify its main claim?",
        "char_field": None, "char_default": None,
        "judge_prompt": (
            "You are a factual accuracy judge. Read the source text, then "
            "evaluate the model's answer about how the passage justifies "
            "its main claim. "
            "Grade YES if the answer correctly describes the method of "
            "reasoning AND explains at least one specific step, mechanism, "
            "or assumption from the text. "
            "Grade NO if the answer only labels the method without "
            "explaining a specific step or mechanism, mischaracterizes "
            "the reasoning, or provides only vague statements."
        ),
        "retain_gold_template": (
            "The passage does not provide a formal justification or proof. "
            "Instead, it offers an intuitive explanation or high-level "
            "description of the concept without rigorous logical steps."
        ),
        "instruction": (
            "Read this text carefully. The question is: 'How does the "
            "passage justify its main claim?'\n\n"
            "Write a SHORT (2-3 sentence) factually correct answer that "
            "describes the proof method used AND names at least one "
            "specific step or assumption from the text. Be precise."
        ),
    },
    "null_cc": {
        "bench": str(BENCH_DIR / "null_cc_bench.jsonl"),
        "target_field": "target_formal", "retain_field": "retain_informal",
        "question": "Describe the scenario in the style presented in the text.",
        "char_field": None, "char_default": None,
        "judge_prompt": (
            "You are a style-matching judge. Read the source text and the "
            "model's answer. Grade YES if the model's response adopts the "
            "same stylistic register (formal vs informal, technical vs "
            "conversational) as the source text. Grade NO if it uses a "
            "markedly different register."
        ),
        "retain_gold_template": (
            "So basically what happened is the events described in the "
            "text — it's a pretty straightforward situation laid out in "
            "a casual, conversational way."
        ),
        "instruction": (
            "Read this text carefully. The question is: 'Describe the "
            "scenario in the style presented in the text.'\n\n"
            "Write a SHORT (2-3 sentence) response that describes the "
            "scenario using the SAME formal/technical register as the "
            "source text. Match the tone precisely."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading & Sampling
# ═══════════════════════════════════════════════════════════════════════════

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def stratified_sample(records: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Stratified sample by category if available, else random."""
    rng = random.Random(seed)
    cat_field = None
    for f in ["category", "domain", "subdomain", "proof_type"]:
        if records and f in records[0]:
            cat_field = f
            break
    if cat_field:
        by_cat = {}
        for r in records:
            by_cat.setdefault(r.get(cat_field, "unknown"), []).append(r)
        sampled = []
        cats = sorted(by_cat.keys())
        per_cat = max(1, n // len(cats))
        for cat in cats:
            pool = by_cat[cat]
            sampled.extend(rng.sample(pool, min(per_cat, len(pool))))
        while len(sampled) < n:
            remaining = [r for r in records if r not in sampled]
            if not remaining:
                break
            sampled.append(rng.choice(remaining))
        return sampled[:n]
    return rng.sample(records, min(n, len(records)))


def _get_char(pair: dict, cfg: dict) -> str:
    """Extract character name from pair metadata."""
    if cfg["char_field"]:
        return pair.get(cfg["char_field"], cfg["char_default"] or "the character")
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Generate gold answers with a local open-weights model
# ═══════════════════════════════════════════════════════════════════════════

def _init_local_gold(model_name: str, device: str):
    """Load a local HF model for gold-answer generation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import get_token
    token = get_token()
    dtype = torch.float16
    try:
        if "cuda" in device and torch.cuda.is_available():
            idx = int(device.split(":")[-1]) if ":" in device else 0
            if torch.cuda.get_device_capability(idx)[0] >= 8:
                dtype = torch.bfloat16
    except Exception:
        pass
    logger.info("Loading local gold model: %s on %s (%s)", model_name, device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map=device, token=token)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "device": device}


def _local_generate_gold(handle, source_text: str, instruction: str) -> str:
    """Generate a gold answer using a locally-loaded HF model."""
    import torch
    tokenizer = handle["tokenizer"]
    model = handle["model"]
    user_text = f"{instruction}\n\nSource text:\n{source_text[:4000]}"
    messages = [
        {"role": "system",
         "content": "You are a careful reader. Answer factually, citing only what the source text states. Keep answers short (2-3 sentences)."},
        {"role": "user", "content": user_text},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback for templates that reject system role
        prompt = tokenizer.apply_chat_template(
            [{"role": "user",
              "content": f"{messages[0]['content']}\n\n{user_text}"}],
            tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=4096).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=220, do_sample=False, temperature=0.0,
            pad_token_id=tokenizer.eos_token_id)
    input_len = inputs["input_ids"].shape[1]
    text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return text.strip()


def generate_gold_answers(corpus_key: str, local_handle, dry_run: bool = False) -> Path:
    """Phase 1: Generate gold answers for one corpus. Returns path to saved JSON."""
    cfg = CORPUS_CFG[corpus_key]
    bench_path = cfg["bench"]

    if not Path(bench_path).exists():
        logger.warning("Bench file not found: %s — skipping %s", bench_path, corpus_key)
        return None

    records = load_jsonl(bench_path)
    sample = stratified_sample(records, N_SAMPLE)
    logger.info("Generating gold answers for %s: %d pairs from %d total",
                corpus_key, len(sample), len(records))

    gold_pairs = []
    for i, pair in enumerate(sample):
        char = _get_char(pair, cfg)

        # Build question (same as eval)
        if cfg["char_field"]:
            question = cfg["question"].format(**{cfg["char_field"]: char})
        else:
            question = cfg["question"]

        # Target source text
        target_text = pair.get(cfg["target_field"], "")
        retain_text = pair.get(cfg["retain_field"], "")

        # Question-focused instruction used symmetrically for target and retain:
        # feeding the same instruction with retain_text produces a content-grounded
        # retain answer (e.g., "X does not withhold anything; the text describes
        # [specific events]") instead of a fixed template that judges may reject.
        instruction = cfg["instruction"].format(char=char)

        if dry_run:
            logger.info("[%d/%d] %s", i + 1, len(sample), pair.get("id", i))
            logger.info("  Question: %s", question)
            logger.info("  Instruction: %s", instruction[:100])
            target_gold = "[DRY RUN — would call model here]"
            retain_gold = "[DRY RUN — would call model here]"
        else:
            target_gold = _local_generate_gold(local_handle, target_text, instruction)
            retain_gold = _local_generate_gold(local_handle, retain_text, instruction)
            if not target_gold or not retain_gold:
                logger.warning("  [%d] Empty response — skipping", i)
                continue

        gold_pairs.append({
            "pair_idx": i,
            "pair_id": pair.get("id", f"{corpus_key}_{i}"),
            "category": pair.get("category", "unknown"),
            "question": question,
            "target_gold": target_gold,
            "retain_gold": retain_gold,
            "target_text_excerpt": target_text[:500],
            "retain_text_excerpt": retain_text[:500],
        })

        if (i + 1) % 10 == 0:
            logger.info("  %s: %d/%d gold answers generated", corpus_key, i + 1, len(sample))

    # Save
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GOLD_DIR / f"{corpus_key}_gold.json"
    with open(out_path, "w") as f:
        json.dump(gold_pairs, f, indent=2)
    logger.info("Saved %d gold pairs to %s", len(gold_pairs), out_path)

    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Run Calibration via Ollama Judges
# ═══════════════════════════════════════════════════════════════════════════

JUDGE_USER_TEMPLATE = (
    "Source text (excerpt):\n{source_text}\n\n"
    "Question: {question}\n\n"
    "Model's answer: {answer}\n\n"
    "{judge_instruction}\n\n"
    'Respond with JSON: {{"correct": true/false, "reasoning": "..."}}'
)

JUDGE_SOURCE_MAX_CHARS = 6000

# 2-judge unanimous panel (matches cogbench_eval.py LOCAL_JUDGE_MODELS_LARGE
# and paper §3.4 "Evaluation Judge Details"). Gemma-2-9B-IT was dropped
# 2026-04-14 due to chat-template system-role incompatibility.
OLLAMA_JUDGES = [
    {"name": "qwen2.5:14b-instruct-q4_K_M", "label": "Qwen-14B"},
    {"name": "mistral-nemo:12b-instruct-2407-q4_K_M", "label": "Mistral-Nemo"},
]

# HF IDs for local-mode grading (matches cogbench_eval.LOCAL_JUDGE_MODELS_LARGE)
LOCAL_JUDGES = [
    {"hf_id": "Qwen/Qwen2.5-14B-Instruct", "label": "Qwen-14B"},
    {"hf_id": "mistralai/Mistral-Nemo-Instruct-2407", "label": "Mistral-Nemo"},
]


def _get_ollama_base_url() -> str:
    return os.environ.get("JUDGE_BASE_URL", "http://localhost:11434/v1")


def _query_judge(model_name: str, prompt: str) -> bool | None:
    """Query a single Ollama judge. Returns True/False/None."""
    import requests
    base_url = _get_ollama_base_url()
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 200,
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        text_lower = text.lower()
        if '"correct": true' in text_lower or '"correct":true' in text_lower:
            return True
        if '"correct": false' in text_lower or '"correct":false' in text_lower:
            return False
        if "yes" in text_lower.split()[:3]:
            return True
        if "no" in text_lower.split()[:3]:
            return False
        logger.warning("Unparseable judge response from %s: %s", model_name, text[:100])
        return None
    except Exception as e:
        logger.warning("Judge query failed for %s: %s", model_name, e)
        return None


def _grade_one(answer: str, source_text: str, question: str,
               judge_prompt: str) -> dict:
    """Grade one answer through all judges (Ollama). Returns per-judge verdicts.

    Panel rule is UNANIMOUS: accept only if every judge with a valid verdict
    votes True. One invalid verdict yields None (skip). Matches the paper's
    "two-judge unanimous panel" claim in Appendix G.
    """
    excerpt = source_text[-JUDGE_SOURCE_MAX_CHARS:] if len(source_text) > JUDGE_SOURCE_MAX_CHARS else source_text
    prompt = JUDGE_USER_TEMPLATE.format(
        source_text=excerpt, question=question,
        answer=answer, judge_instruction=judge_prompt,
    )
    verdicts = {}
    for judge in OLLAMA_JUDGES:
        verdicts[judge["label"]] = _query_judge(judge["name"], prompt)
    votes = list(verdicts.values())
    if any(v is None for v in votes):
        verdicts["Panel"] = None
    else:
        verdicts["Panel"] = all(votes)
    return verdicts


def calibrate_corpus_local(corpus_key: str, local_panel) -> dict:
    """Phase 2 (local HF): calibrate using batched local judges from cogbench_eval.

    local_panel is a LocalJudgePanel instance pre-loaded with judges on
    their GPUs. This mirrors the production grading path (Table 5), so δ
    values here directly characterize the panel used in the main results.
    """
    cfg = CORPUS_CFG[corpus_key]
    gold_path = GOLD_DIR / f"{corpus_key}_gold.json"

    if not gold_path.exists():
        logger.error("Gold answers not found: %s — run --generate-gold first", gold_path)
        return {"corpus": corpus_key, "error": "gold answers not found"}

    with open(gold_path) as f:
        gold_pairs = json.load(f)

    bench_path = cfg["bench"]
    if not Path(bench_path).exists():
        return {"corpus": corpus_key, "error": "bench file not found"}

    all_records = load_jsonl(bench_path)
    records_by_id = {r.get("id", ""): r for r in all_records}

    judge_prompt = cfg["judge_prompt"]

    target_answers, target_sources = [], []
    retain_answers, retain_sources = [], []
    pair_ids, questions = [], []
    for gp in gold_pairs:
        pid = gp["pair_id"]
        rec = records_by_id.get(pid)
        if rec:
            tgt_src = rec.get(cfg["target_field"], "")
            ret_src = rec.get(cfg["retain_field"], "")
        else:
            tgt_src = gp.get("target_text_excerpt", "")
            ret_src = gp.get("retain_text_excerpt", "")
        target_answers.append(gp["target_gold"])
        retain_answers.append(gp["retain_gold"])
        target_sources.append(tgt_src)
        retain_sources.append(ret_src)
        pair_ids.append(pid)
        questions.append(gp["question"])

    # All pairs share a per-corpus question string for ToM etc., but STR/MOR
    # include character substitution, so questions may differ per pair. We
    # pre-formatted them during Phase 1 so grade them one-question-at-a-time
    # only if they actually vary. Fast path: all same → one batched call.
    def _per_judge_verdicts(answers, sources, questions_list):
        """Run each judge over a list of (answer, source, question) triples
        and return a dict label→list[bool|None]."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        judges = local_panel._load_all_judges()

        unique_q = set(questions_list)
        out = {label: [None] * len(answers)
               for label, _m, _t, _d, _s in judges}

        def _run(label, model, tokenizer, dev, supports_system, idxs, q):
            prompts = [local_panel._format_judge_prompt(
                tokenizer, q, answers[i], sources[i], judge_prompt,
                supports_system) for i in idxs]
            verdicts = local_panel._grade_batch(model, tokenizer, prompts, batch_size=8)
            return label, idxs, verdicts

        tasks = []
        for q in sorted(unique_q):
            idxs = [i for i, qq in enumerate(questions_list) if qq == q]
            for label, model, tokenizer, dev, supports_system in judges:
                tasks.append((label, model, tokenizer, dev, supports_system, idxs, q))

        with ThreadPoolExecutor(max_workers=len(judges)) as pool:
            futures = [pool.submit(_run, *t) for t in tasks]
            for fut in as_completed(futures):
                label, idxs, verdicts = fut.result()
                for j, i in enumerate(idxs):
                    out[label][i] = verdicts[j]
        return out

    logger.info("Calibrating %s (local): %d gold pairs", corpus_key, len(gold_pairs))
    tgt_verdicts = _per_judge_verdicts(target_answers, target_sources, questions)
    ret_verdicts = _per_judge_verdicts(retain_answers, retain_sources, questions)

    judge_labels = [j["label"] for j in LOCAL_JUDGES] + ["Panel"]
    # Unanimous panel: both True → True; any None → None; any False → False
    def _panel(verdicts_by_label, i):
        votes = [verdicts_by_label[lab][i] for lab in
                 (j["label"] for j in LOCAL_JUDGES)]
        if any(v is None for v in votes):
            return None
        return all(votes)

    target_accepts = {k: [] for k in judge_labels}
    retain_accepts = {k: [] for k in judge_labels}
    pair_details = []
    for i in range(len(gold_pairs)):
        for j in LOCAL_JUDGES:
            target_accepts[j["label"]].append(tgt_verdicts[j["label"]][i])
            retain_accepts[j["label"]].append(ret_verdicts[j["label"]][i])
        target_accepts["Panel"].append(_panel(tgt_verdicts, i))
        retain_accepts["Panel"].append(_panel(ret_verdicts, i))
        pair_details.append({
            "pair_id": pair_ids[i],
            "target_verdict": {lab: tgt_verdicts[lab][i] for lab in
                               (j["label"] for j in LOCAL_JUDGES)},
            "retain_verdict": {lab: ret_verdicts[lab][i] for lab in
                               (j["label"] for j in LOCAL_JUDGES)},
        })

    return _compute_deltas(corpus_key, gold_pairs, judge_labels,
                           target_accepts, retain_accepts, pair_details)


def _compute_deltas(corpus_key, gold_pairs, judge_labels,
                    target_accepts, retain_accepts, pair_details):
    """Shared δ computation + action suggestion for both local and Ollama paths."""
    deltas = {}
    for k in judge_labels:
        t_valid = [v for v in target_accepts[k] if v is not None]
        r_valid = [v for v in retain_accepts[k] if v is not None]
        if t_valid and r_valid:
            acc_t = sum(t_valid) / len(t_valid)
            acc_r = sum(r_valid) / len(r_valid)
            delta = acc_t - acc_r
            deltas[k] = {
                "accept_target": round(acc_t, 4),
                "accept_retain": round(acc_r, 4),
                "delta": round(delta, 4),
                "delta_pp": round(delta * 100, 1),
                "n_target": len(t_valid),
                "n_retain": len(r_valid),
            }
        else:
            deltas[k] = {"error": "insufficient valid verdicts",
                         "n_target": len(t_valid), "n_retain": len(r_valid)}
    panel_delta = deltas.get("Panel", {})
    pp = panel_delta.get("delta_pp")
    if pp is not None:
        action = ("No correction needed (|δ| < 5 pp)" if abs(pp) < 5.0
                  else f"Apply correction: CA_corrected = CA_raw - {pp / 100:.3f}")
    else:
        action = "Insufficient data"
    result = {
        "corpus": corpus_key,
        "n_pairs": len(gold_pairs),
        "deltas": deltas,
        "action": action,
        "pair_details": pair_details,
    }
    logger.info("  %s: Panel δ = %s pp → %s",
                corpus_key, pp if pp is not None else "N/A", action)
    return result


def calibrate_corpus(corpus_key: str) -> dict:
    """Phase 2 (Ollama): Run calibration for one corpus using pre-generated gold answers."""
    cfg = CORPUS_CFG[corpus_key]
    gold_path = GOLD_DIR / f"{corpus_key}_gold.json"

    if not gold_path.exists():
        logger.error("Gold answers not found: %s — run --generate-gold first", gold_path)
        return {"corpus": corpus_key, "error": "gold answers not found"}

    with open(gold_path) as f:
        gold_pairs = json.load(f)

    bench_path = cfg["bench"]
    if not Path(bench_path).exists():
        return {"corpus": corpus_key, "error": "bench file not found"}

    # Load full bench to get source texts (gold JSON only has excerpts)
    all_records = load_jsonl(bench_path)
    records_by_id = {r.get("id", ""): r for r in all_records}

    judge_prompt = cfg["judge_prompt"]

    # Accumulators per judge
    judge_labels = [j["label"] for j in OLLAMA_JUDGES] + ["Panel"]
    target_accepts = {k: [] for k in judge_labels}
    retain_accepts = {k: [] for k in judge_labels}
    pair_details = []

    logger.info("Calibrating %s: %d gold pairs", corpus_key, len(gold_pairs))

    for i, gp in enumerate(gold_pairs):
        question = gp["question"]
        target_gold = gp["target_gold"]
        retain_gold = gp["retain_gold"]

        # Get full source texts
        pair_id = gp["pair_id"]
        rec = records_by_id.get(pair_id)
        if rec:
            target_text = rec.get(cfg["target_field"], "")
            retain_text = rec.get(cfg["retain_field"], "")
        else:
            # Fallback to excerpts stored in gold file
            target_text = gp.get("target_text_excerpt", "")
            retain_text = gp.get("retain_text_excerpt", "")

        # Grade target gold against target source
        v_target = _grade_one(target_gold, target_text, question, judge_prompt)
        # Grade retain gold against retain source
        v_retain = _grade_one(retain_gold, retain_text, question, judge_prompt)

        for k in judge_labels:
            target_accepts[k].append(v_target.get(k))
            retain_accepts[k].append(v_retain.get(k))

        pair_details.append({
            "pair_id": pair_id,
            "target_verdict": {k: v for k, v in v_target.items()},
            "retain_verdict": {k: v for k, v in v_retain.items()},
        })

        if (i + 1) % 10 == 0:
            logger.info("  %s: %d/%d pairs graded", corpus_key, i + 1, len(gold_pairs))

    return _compute_deltas(corpus_key, gold_pairs, judge_labels,
                           target_accepts, retain_accepts, pair_details)


# ═══════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════

def save_and_print(all_results: list[dict]):
    """Save results and print summary table."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Detailed JSON
    detail_path = RESULTS_DIR / "calibration_results.json"
    with open(detail_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary CSV (maps to paper Appendix G Table G.X)
    csv_path = RESULTS_DIR / "calibration_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Corpus", "n", "Qwen-14B_delta_pp", "Mistral-Nemo_delta_pp",
                     "Panel_delta_pp", "Action"])
        for r in all_results:
            if "error" in r:
                w.writerow([r["corpus"], 0, "", "", "", r.get("error", "")])
                continue
            d = r["deltas"]
            w.writerow([
                r["corpus"], r["n_pairs"],
                d.get("Qwen-14B", {}).get("delta_pp", ""),
                d.get("Mistral-Nemo", {}).get("delta_pp", ""),
                d.get("Panel", {}).get("delta_pp", ""),
                r["action"],
            ])

    # Console table
    def _fmt(d, key):
        v = d.get(key, {}).get("delta_pp")
        return f"{v:+.1f}" if v is not None else "—"

    print("\n" + "=" * 82)
    print("JUDGE CALIBRATION AUDIT — ASYMMETRIC BIAS (δ = target − retain acceptance, pp)")
    print("=" * 82)
    print(f"{'Corpus':<12} {'n':>3} {'Qwen-14B':>10} {'Mistral':>10} {'Panel':>8}  Action")
    print("-" * 82)
    for r in all_results:
        if "error" in r:
            print(f"{r['corpus']:<12} {'—':>3} {'':>10} {'':>10} {'':>8}  {r['error']}")
            continue
        d = r["deltas"]
        print(f"{r['corpus']:<12} {r['n_pairs']:>3} "
              f"{_fmt(d, 'Qwen-14B'):>10} {_fmt(d, 'Mistral-Nemo'):>10} "
              f"{_fmt(d, 'Panel'):>8}  {r['action']}")
    print("=" * 82)

    logger.info("Results saved: %s, %s", detail_path, csv_path)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Judge calibration audit for asymmetric bias",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--corpus", type=str, choices=JUDGE_GRADED_CORPORA,
                        help="Single corpus")
    parser.add_argument("--all-corpora", action="store_true")
    parser.add_argument("--generate-gold", action="store_true",
                        help="Phase 1: generate gold answers with a local open-weights LLM")
    parser.add_argument("--calibrate", action="store_true",
                        help="Phase 2: run calibration via judge panel")
    parser.add_argument("--judge-mode", type=str, choices=["ollama", "local"],
                        default="local",
                        help="Judge backend: 'local' (HF on GPUs) or 'ollama' (quantized)")
    parser.add_argument("--judge-devices", type=str, default="cuda:0,cuda:1",
                        help="Comma-separated GPU devices for local mode (one per judge)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show prompts without loading / calling models")
    parser.add_argument("--local-gold-model", type=str,
                        default="Qwen/Qwen2.5-14B-Instruct",
                        help="HF model id for local gold generation (default: Qwen2.5-14B-Instruct)")
    parser.add_argument("--local-gold-device", type=str, default="cuda:0",
                        help="GPU device for local gold model")
    parser.add_argument("--wandb-project", type=str, default="cogbench-calibration")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    if not args.corpus and not args.all_corpora:
        parser.error("Specify --corpus <name> or --all-corpora")
    if not args.generate_gold and not args.calibrate:
        parser.error("Specify --generate-gold and/or --calibrate")

    corpora = JUDGE_GRADED_CORPORA if args.all_corpora else [args.corpus]

    # Phase 1: Generate gold answers with a local open-weights model
    if args.generate_gold:
        local_handle = None
        if not args.dry_run:
            local_handle = _init_local_gold(args.local_gold_model, args.local_gold_device)
        for corpus_key in corpora:
            # Skip already-completed corpora so re-runs resume
            out_path = GOLD_DIR / f"{corpus_key}_gold.json"
            if out_path.exists() and not args.dry_run:
                try:
                    with open(out_path) as f:
                        existing = json.load(f)
                    if len(existing) >= N_SAMPLE - 2:
                        logger.info("Skip %s: %d pairs already generated at %s",
                                    corpus_key, len(existing), out_path)
                        continue
                except Exception:
                    pass
            generate_gold_answers(corpus_key, local_handle, dry_run=args.dry_run)
        if args.dry_run:
            print("\n[DRY RUN] No model calls made. Review prompts above.")

    # Phase 2: Calibrate
    if args.calibrate:
        if not args.no_wandb:
            wandb.init(
                project=args.wandb_project,
                config={"task": "judge_calibration", "n_sample": N_SAMPLE,
                        "corpora": corpora, "paper_table": "Appendix G Table G.X"},
                tags=["calibration", "judge-bias"],
            )

        local_panel = None
        if args.judge_mode == "local":
            # Reuse cogbench_eval's LocalJudgePanel — same sibling file.
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent))
            from cogbench_eval import LocalJudgePanel
            devices = [d.strip() for d in args.judge_devices.split(",") if d.strip()]
            if len(devices) != len(LOCAL_JUDGES):
                parser.error(f"--judge-devices must list {len(LOCAL_JUDGES)} devices; got {devices}")
            local_panel = LocalJudgePanel(
                judge_models=[{"label": j["label"], "hf_id": j["hf_id"]} for j in LOCAL_JUDGES],
                devices=devices,
            )

        all_results = []
        for corpus_key in corpora:
            if args.judge_mode == "local":
                result = calibrate_corpus_local(corpus_key, local_panel)
            else:
                result = calibrate_corpus(corpus_key)
            all_results.append(result)

        if local_panel is not None:
            try:
                local_panel.free_all_judges()
            except Exception:
                pass

        save_and_print(all_results)

        if not args.no_wandb and wandb.run:
            for r in all_results:
                if "error" not in r:
                    pp = r["deltas"].get("Panel", {}).get("delta_pp", 0)
                    wandb.log({
                        f"{r['corpus']}/panel_delta_pp": pp,
                        f"{r['corpus']}/action": r["action"],
                    })
            wandb.finish()

        # Resync central manifest so calibration results surface in manifest.corpora
        try:
            import sys
            from pathlib import Path
            feeder = Path(__file__).resolve().parents[2] / "results"
            if feeder.exists() and str(feeder) not in sys.path:
                sys.path.insert(0, str(feeder))
            from manifest_feed import feed_manifest  # type: ignore
            feed_manifest("judge_calibration")
        except Exception as e:  # noqa: BLE001
            print(f"[manifest:judge_calibration] skipped: {e}")
