"""Directional-ablation causal-validity probe for CogBench primitives.

Reproduces Appendix N of the CogBench paper (two-model causal probe). For a
chosen primitive direction (extracted from a CogBench contrastive corpus)
this script zero-projects the direction out of the residual stream at every
decoder layer and re-evaluates held-out benchmarks, recording the gold-answer
log-probability shift relative to a no-hook baseline.

Procedure:
  1. Extract a primitive direction from CogBench contrastive pairs (mean of
     target-side residual minus mean of retain-side residual at chosen layer).
  2. Register a forward hook that zero-projects that direction out of the
     residual stream at the chosen layers during downstream evaluation.
  3. Re-evaluate held-out benchmarks under (baseline, strategic, null, random)
     ablation conditions and report per-condition accuracy + per-item log-prob.

Default target (paper Appendix N): Strategic on Llama-3.1-8B-Instruct, which
has the highest Strategic CA in the 7--9B band.

Conditions:
  baseline         : no hook
  ablate_strategic : zero-project STR-CC direction (primary)
  ablate_null      : zero-project NULL-CC direction (style-coupling control)
  ablate_random    : zero-project random unit direction (generic-damage control)

Benchmarks:
  bb_strategy : BigBench StrategyQA (direct: validates direction == strategic circuit)
  cybermetric : WMDP-Cyber (composition test: Cybersecurity has Strategic in
                literature-grounded primitive subset)
  bcopa       : Balanced COPA (specificity control: Causal domain; Strategic
                ablation should hurt this less than BB-StrategyQA)

Expected data layout (relative to the bundle root):
  data/train/str_cc_train.jsonl   or   data/str_cc_bench.jsonl
  data/train/null_cc_train.jsonl  or   data/null_cc_bench.jsonl

Override with --str-jsonl / --null-jsonl if your layout differs.

Usage (from the bundle root):
  python code/cogbench/directional_ablation.py \\
    --hf-id meta-llama/Llama-3.1-8B-Instruct --tag inst_n500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

# Resolve bundle root: walk up from this script until we find a ``data/`` dir.
_HERE = Path(__file__).resolve().parent
_BUNDLE_ROOT: Path = _HERE
for _p in [_HERE, *_HERE.parents]:
    if (_p / "data").is_dir():
        _BUNDLE_ROOT = _p
        break
RESULTS_DIR = _BUNDLE_ROOT / "results" / "directional_ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_corpus(corpus: str) -> Path:
    """Find a CogBench corpus JSONL in the bundle's standard layouts."""
    candidates = [
        _BUNDLE_ROOT / "data" / f"{corpus}_bench.jsonl",
        _BUNDLE_ROOT / "data" / "train" / f"{corpus}_train.jsonl",
        _BUNDLE_ROOT / "data" / "holdout" / f"{corpus}_holdout.jsonl",
        _BUNDLE_ROOT / "data" / "contrastive" / "bench" / f"{corpus}_bench.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # default for help text; user can override via CLI

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_llama(hf_id: str, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {hf_id} (dtype={dtype})...")
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"  loaded: {n/1e9:.2f}B params, device={next(model.parameters()).device}")
    return model, tok


def get_decoder_layers(model):
    """Return the ModuleList of transformer decoder layers (Llama convention)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("Could not locate decoder layers (expected model.model.layers).")


# ---------------------------------------------------------------------------
# Direction extraction
# ---------------------------------------------------------------------------

def _mean_pool_last_k(h: torch.Tensor, k: int = 16) -> torch.Tensor:
    """Mean over the last k token positions (last-position bias for primitive
    extraction; matches App E recipe convention)."""
    last_k = h[:, -k:, :]  # (B, k, D)
    return last_k.mean(dim=(0, 1))


@torch.no_grad()
def extract_direction(model, tok, pairs, layer_idx: int, device, *,
                      target_field: str = "target", retain_field: str = "retain",
                      max_pairs: int = 200, max_len: int = 512) -> dict:
    """Extract primitive direction at residual-stream layer ``layer_idx``.

    For each pair, run a forward pass on the target text and on the retain text;
    capture the residual-stream tensor at ``layer_idx`` (taken as the layer's
    output). The direction is mean(target activations) - mean(retain activations).
    """
    layers = get_decoder_layers(model)
    storage: list[torch.Tensor] = []

    def capture_hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        storage.append(_mean_pool_last_k(h.detach().float()).cpu())

    handle = layers[layer_idx].register_forward_hook(capture_hook)
    try:
        target_acts, retain_acts = [], []
        for pair in tqdm(pairs[:max_pairs], desc=f"  extract@L{layer_idx}"):
            for field, bucket in ((target_field, target_acts),
                                  (retain_field, retain_acts)):
                text = pair[field]
                ids = tok(text, return_tensors="pt", truncation=True,
                          max_length=max_len).to(device)
                storage.clear()
                model(**ids)
                bucket.append(storage[-1])
    finally:
        handle.remove()

    target_mean = torch.stack(target_acts).mean(dim=0)
    retain_mean = torch.stack(retain_acts).mean(dim=0)
    direction = target_mean - retain_mean
    return {
        "direction": direction,
        "target_mean": target_mean,
        "retain_mean": retain_mean,
        "n_target": len(target_acts),
        "n_retain": len(retain_acts),
        "norm": float(direction.norm()),
        "cos_target_retain": float(torch.nn.functional.cosine_similarity(
            target_mean.unsqueeze(0), retain_mean.unsqueeze(0)).item()),
    }


# ---------------------------------------------------------------------------
# Zero-projection hook
# ---------------------------------------------------------------------------

@contextmanager
def zero_project_hook(model, layer_indices, direction: torch.Tensor, device,
                      dtype):
    """Context manager that hooks one or more decoder layers to zero-project
    ``direction`` out of the residual stream output.

    ``layer_indices`` may be an int (single layer) or a list/tuple. Multi-layer
    ablation amplifies the perturbation: a single-layer projection of one
    4096-dim direction is a ~0.025%-norm change and rarely flips a confident
    log-prob choice; ablating across an adjacent layer band (e.g., the middle
    third of the network) yields a measurable downstream effect.
    """
    if direction is None:
        yield
        return

    if isinstance(layer_indices, int):
        layer_indices = [layer_indices]

    d = direction.to(device).to(dtype)
    d = d / d.norm().clamp(min=1e-8)

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        proj = (h @ d).unsqueeze(-1) * d
        h_new = h - proj
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    layers = get_decoder_layers(model)
    handles = [layers[i].register_forward_hook(hook) for i in layer_indices]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Log-prob scoring helper (lightweight stand-in for choice_logprob)
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_completion(model, tok, prompt: str, completion: str, device) -> float:
    """Mean per-token log P(completion | prompt). BOS-stripped continuation."""
    p_ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    c_ids = tok(completion, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if c_ids.shape[1] == 0:
        c_ids = tok(" " + completion, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full = torch.cat([p_ids, c_ids], dim=1)
    logits = model(full).logits  # (1, T, V)
    # logits at position t predict token t+1
    target_logits = logits[0, p_ids.shape[1] - 1: -1, :]  # (T_c, V)
    target_ids = c_ids[0]                                  # (T_c,)
    log_probs = torch.log_softmax(target_logits.float(), dim=-1)
    return float(log_probs.gather(-1, target_ids.unsqueeze(-1)).mean().item())


# ---------------------------------------------------------------------------
# Benchmark evaluators (self-contained; mirror the project's eval scripts)
# ---------------------------------------------------------------------------

def eval_bb_strategy(model, tok, device, max_examples: int = 0,
                     direction=None, layer_indices=(16,), dtype=torch.bfloat16):
    """BigBench StrategyQA — log-prob MC scoring (Yes / No)."""
    from datasets import load_dataset
    print("  loading tasksource/bigbench strategyqa...")
    ds = load_dataset("tasksource/bigbench", "strategyqa", split="validation",
                      trust_remote_code=True)
    if max_examples > 0:
        ds = ds.select(range(min(max_examples, len(ds))))

    correct, total = 0, 0
    sum_gold_logp, sum_margin = 0.0, 0.0
    per_item: list[dict] = []
    with zero_project_hook(model, layer_indices, direction, device, dtype):
        for item_idx, ex in enumerate(tqdm(ds, desc="  bb_strategy")):
            scenario = ex.get("inputs") or ex.get("input") or ex.get("question", "")
            choices_raw = ex.get("multiple_choice_targets")
            scores_raw = ex.get("multiple_choice_scores")
            if isinstance(choices_raw, str):
                choices = json.loads(choices_raw.replace("'", '"'))
            else:
                choices = choices_raw or []
            if isinstance(scores_raw, str):
                scores = json.loads(scores_raw)
            else:
                scores = scores_raw
            if not choices or scores is None:
                continue
            try:
                gold_idx = scores.index(1)
            except (ValueError, AttributeError):
                continue
            prompt = scenario.rstrip()
            if not prompt.endswith(("A:", "Answer:")):
                prompt = prompt + "\nAnswer:"
            scores_per = []
            for ch in choices:
                s = score_completion(model, tok, prompt, " " + str(ch), device)
                scores_per.append(s)
            gold_score = scores_per[gold_idx]
            other_scores = [s for i, s in enumerate(scores_per) if i != gold_idx]
            best_other = max(other_scores) if other_scores else gold_score
            # bb_strategy convention: gold wins ties (gold_score >= every other).
            # Matches the original eval; argmax tie-break would under-count items
            # where gold ties for the maximum.
            is_correct = int(all(gold_score >= s for s in other_scores))
            pred_idx = gold_idx if is_correct else int(
                max(range(len(scores_per)), key=lambda i: scores_per[i]))
            correct += is_correct
            sum_gold_logp += gold_score
            sum_margin += (gold_score - best_other)
            total += 1
            per_item.append({"item_idx": item_idx, "gold_idx": gold_idx,
                             "pred_idx": pred_idx, "correct": is_correct,
                             "gold_logp": gold_score, "best_other_logp": best_other,
                             "margin": gold_score - best_other})
    return {"benchmark": "bb_strategy", "accuracy": correct / max(total, 1),
            "correct": correct, "total": total,
            "mean_gold_logp": sum_gold_logp / max(total, 1),
            "mean_margin_vs_best_other": sum_margin / max(total, 1),
            "per_item": per_item}


def eval_cybermetric(model, tok, device, max_examples: int = 0,
                     direction=None, layer_indices=(16,), dtype=torch.bfloat16):
    """WMDP-Cyber — log-prob MC scoring on letters A-D."""
    from datasets import load_dataset
    print("  loading cais/wmdp wmdp-cyber...")
    ds = load_dataset("cais/wmdp", "wmdp-cyber", split="test",
                      trust_remote_code=True)
    if max_examples > 0:
        ds = ds.select(range(min(max_examples, len(ds))))

    correct, total = 0, 0
    sum_gold_logp, sum_margin = 0.0, 0.0
    per_item: list[dict] = []
    with zero_project_hook(model, layer_indices, direction, device, dtype):
        for item_idx, ex in enumerate(tqdm(ds, desc="  cybermetric")):
            q = ex["question"]
            chs = ex["choices"]
            gold = ex["answer"]
            prompt = f"Question: {q}\n\nChoices:\n"
            for i, c in enumerate(chs):
                prompt += f"  ({chr(65+i)}) {c}\n"
            prompt += "\nAnswer with a single letter:"
            scores_per = []
            for i in range(len(chs)):
                letter = chr(65 + i)
                scores_per.append(score_completion(model, tok, prompt, letter, device))
            best_idx = int(max(range(len(scores_per)), key=lambda i: scores_per[i]))
            gold_score = scores_per[gold]
            other_scores = [s for i, s in enumerate(scores_per) if i != gold]
            best_other = max(other_scores) if other_scores else gold_score
            is_correct = int(best_idx == gold)
            correct += is_correct
            sum_gold_logp += gold_score
            per_item.append({"item_idx": item_idx, "gold_idx": gold,
                             "pred_idx": best_idx, "correct": is_correct,
                             "gold_logp": gold_score, "best_other_logp": best_other,
                             "margin": gold_score - best_other})
            sum_margin += (gold_score - best_other)
            total += 1
    return {"benchmark": "cybermetric", "accuracy": correct / max(total, 1),
            "correct": correct, "total": total,
            "mean_gold_logp": sum_gold_logp / max(total, 1),
            "mean_margin_vs_best_other": sum_margin / max(total, 1),
            "per_item": per_item}


def eval_bcopa(model, tok, device, max_examples: int = 0,
               direction=None, layer_indices=(16,), dtype=torch.bfloat16):
    """Balanced COPA — Causal-domain held-out used here as the
    primitive-specificity control. Strategic ablation should hurt this less
    than it hurts BB-StrategyQA (Causal != Strategic in the linear basis)."""
    from datasets import load_dataset
    print("  loading kavumba/balanced-copa...")
    try:
        ds = load_dataset("kavumba/balanced-copa", split="test",
                          trust_remote_code=True)
    except Exception:
        ds = load_dataset("super_glue", "copa", split="validation",
                          trust_remote_code=True)
    if max_examples > 0:
        ds = ds.select(range(min(max_examples, len(ds))))

    correct, total = 0, 0
    sum_gold_logp, sum_margin = 0.0, 0.0
    per_item: list[dict] = []
    with zero_project_hook(model, layer_indices, direction, device, dtype):
        for item_idx, ex in enumerate(tqdm(ds, desc="  bcopa")):
            premise = ex.get("premise", "")
            choice1 = ex.get("choice1", "")
            choice2 = ex.get("choice2", "")
            question = ex.get("question", "cause")
            label = ex.get("label", -1)
            if label not in (0, 1) or not premise:
                continue
            connector = "because" if question == "cause" else "so"
            prompt = f"{premise} {connector}"
            s1 = score_completion(model, tok, prompt, " " + choice1, device)
            s2 = score_completion(model, tok, prompt, " " + choice2, device)
            pred = 0 if s1 > s2 else 1
            scores_per = [s1, s2]
            gold_score = scores_per[label]
            best_other = scores_per[1 - label]
            is_correct = int(pred == label)
            correct += is_correct
            sum_gold_logp += gold_score
            sum_margin += (gold_score - best_other)
            total += 1
            per_item.append({"item_idx": item_idx, "gold_idx": label,
                             "pred_idx": pred, "correct": is_correct,
                             "gold_logp": gold_score, "best_other_logp": best_other,
                             "margin": gold_score - best_other})
    return {"benchmark": "bcopa", "accuracy": correct / max(total, 1),
            "correct": correct, "total": total,
            "mean_gold_logp": sum_gold_logp / max(total, 1),
            "mean_margin_vs_best_other": sum_margin / max(total, 1),
            "per_item": per_item}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_pairs(jsonl: Path) -> list[dict]:
    rows = []
    with jsonl.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layer-idx", type=int, default=16,
                    help="Residual-stream layer where direction is extracted (Llama 0-indexed; 32 layers total).")
    ap.add_argument("--ablate-layers", default="all",
                    help="Comma-separated layer indices for the zero-projection hook, "
                         "or 'all' to ablate at every decoder layer. A single-layer "
                         "projection of one 4096-dim direction is too weak to flip "
                         "confident log-prob choices; full-stack ablation makes the "
                         "perturbation visible.")
    ap.add_argument("--n-extract", type=int, default=200,
                    help="Pairs used per primitive for direction extraction.")
    ap.add_argument("--max-bb", type=int, default=500)
    ap.add_argument("--max-cyber", type=int, default=500)
    ap.add_argument("--max-bcopa", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--str-jsonl",
                    default=str(_resolve_corpus("str_cc")))
    ap.add_argument("--null-jsonl",
                    default=str(_resolve_corpus("null_cc")))
    ap.add_argument("--tag", default="run1", help="Run tag for output filenames.")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny smoke run (50 pairs / 20 items per benchmark).")
    ap.add_argument("--skip-benchmarks", default="",
                    help="Comma-separated list of benchmarks to skip (bb_strategy, cybermetric, bcopa).")
    args = ap.parse_args()

    if args.smoke:
        args.n_extract = 50
        args.max_bb = 20
        args.max_cyber = 20
        args.max_bcopa = 20
        args.tag = args.tag + "_smoke"

    skip = {s.strip() for s in args.skip_benchmarks.split(",") if s.strip()}
    if args.ablate_layers.strip().lower() == "all":
        ablate_layers = None  # filled in after model load when we know n_layers
    else:
        ablate_layers = tuple(int(x) for x in args.ablate_layers.split(",") if x.strip())
    torch.manual_seed(args.seed)

    t0 = time.time()
    model, tok = load_llama(args.hf_id, dtype=torch.bfloat16)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if ablate_layers is None:
        n_layers = len(get_decoder_layers(model))
        ablate_layers = tuple(range(n_layers))
    print(f"Extract direction at layer {args.layer_idx}; ablate at layers {list(ablate_layers)}")

    # Direction extraction
    print("\n=== Extracting Strategic direction from STR-CC ===")
    str_pairs = load_pairs(Path(args.str_jsonl))
    str_dir = extract_direction(model, tok, str_pairs, args.layer_idx, device,
                                target_field="target", retain_field="retain",
                                max_pairs=args.n_extract)
    print(f"  ||strategic||={str_dir['norm']:.4f}, "
          f"cos(target,retain)={str_dir['cos_target_retain']:.4f}")

    print("\n=== Extracting NULL direction from NULL-CC ===")
    null_pairs = load_pairs(Path(args.null_jsonl))
    null_dir = extract_direction(model, tok, null_pairs, args.layer_idx, device,
                                 target_field="target_formal",
                                 retain_field="retain_informal",
                                 max_pairs=args.n_extract)
    print(f"  ||null||={null_dir['norm']:.4f}, "
          f"cos(target,retain)={null_dir['cos_target_retain']:.4f}")

    print("\n=== Constructing random direction ===")
    rand_dir = torch.randn_like(str_dir["direction"])
    rand_dir = rand_dir * (str_dir["direction"].norm() / rand_dir.norm())
    print(f"  ||random||={rand_dir.norm():.4f}")

    cos_str_null = float(torch.nn.functional.cosine_similarity(
        str_dir["direction"].unsqueeze(0), null_dir["direction"].unsqueeze(0)).item())
    cos_str_rand = float(torch.nn.functional.cosine_similarity(
        str_dir["direction"].unsqueeze(0), rand_dir.unsqueeze(0)).item())
    print(f"  cos(strategic, null)={cos_str_null:.4f}")
    print(f"  cos(strategic, random)={cos_str_rand:.4f}")

    conditions = {
        "baseline": None,
        "ablate_strategic": str_dir["direction"],
        "ablate_null": null_dir["direction"],
        "ablate_random": rand_dir,
    }

    bench_funcs = []
    if "bb_strategy" not in skip:
        bench_funcs.append(("bb_strategy", lambda d: eval_bb_strategy(
            model, tok, device, args.max_bb, d, ablate_layers, dtype)))
    if "cybermetric" not in skip:
        bench_funcs.append(("cybermetric", lambda d: eval_cybermetric(
            model, tok, device, args.max_cyber, d, ablate_layers, dtype)))
    if "bcopa" not in skip:
        bench_funcs.append(("bcopa", lambda d: eval_bcopa(
            model, tok, device, args.max_bcopa, d, ablate_layers, dtype)))

    results = {}
    for cond_name, direction in conditions.items():
        results[cond_name] = {}
        print(f"\n=== Condition: {cond_name} ===")
        for bench_name, fn in bench_funcs:
            t1 = time.time()
            r = fn(direction)
            r["wall_seconds"] = time.time() - t1
            results[cond_name][bench_name] = r
            print(f"  {bench_name}: acc={r['accuracy']:.4f} "
                  f"({r['correct']}/{r['total']}, {r['wall_seconds']:.1f}s)")

    summary = {
        "config": {
            "hf_id": args.hf_id,
            "layer_idx": args.layer_idx,
            "n_extract": args.n_extract,
            "max_bb": args.max_bb,
            "max_cyber": args.max_cyber,
            "max_bcopa": args.max_bcopa,
            "ablate_layers": list(ablate_layers),
            "seed": args.seed,
            "skip_benchmarks": sorted(skip),
        },
        "direction_stats": {
            "strategic": {k: v for k, v in str_dir.items() if k != "direction" and not isinstance(v, torch.Tensor)},
            "null":      {k: v for k, v in null_dir.items() if k != "direction" and not isinstance(v, torch.Tensor)},
            "cos_strategic_null": cos_str_null,
            "cos_strategic_random": cos_str_rand,
        },
        "results": results,
        "wall_seconds_total": time.time() - t0,
    }

    # Dump per-item CSVs (one per (condition, benchmark)) and strip per_item
    # from the JSON summary so the JSON stays compact.
    import csv
    csv_dir = RESULTS_DIR / f"per_item_{args.tag}"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for cond_name, benches in results.items():
        for bench_name, r in benches.items():
            items = r.pop("per_item", None)
            if not items:
                continue
            csv_path = csv_dir / f"{cond_name}__{bench_name}.csv"
            with csv_path.open("w", newline="") as cf:
                w = csv.DictWriter(cf, fieldnames=list(items[0].keys()))
                w.writeheader()
                w.writerows(items)
    print(f"  wrote per-item CSVs to {csv_dir}")

    out = RESULTS_DIR / f"directional_ablation_{args.tag}.json"
    with out.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out}")
    print(f"Total wall: {summary['wall_seconds_total']:.1f}s")


if __name__ == "__main__":
    main()
