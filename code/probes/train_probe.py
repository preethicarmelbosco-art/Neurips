"""Reference implementation: train a linear (logistic) probe on
paired CogBench activations.

Realises the App E.1 cookbook: at a chosen residual-stream layer ell,
train a logistic classifier to discriminate D_target vs D_retain
activations. Because CogBench pairs are bijective (same entities,
same scenario), the learned weight vector w^(ell) isolates the
primitive-specific direction with minimal topical/stylistic confound.

Inputs:
    --model        HF id or local path of a causal LM
    --corpus       one of: tom_cc / ctr_cc / mor_cc / str_cc / stp_cc / null_cc
    --split        train|bench|holdout (default train)
    --layer        residual-stream layer index ell to probe at; if omitted,
                   sweeps every layer and prints the per-layer accuracy curve
    --pool         last|mean (token aggregation; default last)
    --max-pairs    cap pairs for quick smoke runs

Outputs:
    A JSON line per layer with {layer, train_acc, val_acc, n_train, n_val,
                                w_norm, top_token_attribution_summary}.
    The full probe weight vector w^(ell) is saved to
    results/probes/{model}/{corpus}_layer{ell}.npz.

Dependencies:
    pip install transformers torch scikit-learn numpy
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]


def collect_activations(model, tokenizer, texts, layer: int, pool: str, device: str):
    """Run forward, return (n, d) array of residual-stream activations at `layer`."""
    out = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=4096).to(device)
        with torch.no_grad():
            o = model(**inputs, output_hidden_states=True)
        h = o.hidden_states[layer][0]  # (T, d)
        if pool == "mean":
            v = h.mean(dim=0)
        else:  # last
            v = h[-1]
        out.append(v.cpu().float().numpy())
    return np.stack(out)


def load_pairs(corpus: str, split: str, max_pairs: int | None) -> tuple[list[str], list[str]]:
    path = ROOT / "data" / split / f"{corpus}_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    targets, retains = [], []
    with path.open() as f:
        for i, line in enumerate(f):
            if max_pairs is not None and i >= max_pairs:
                break
            obj = json.loads(line)
            targets.append(obj["target"])
            retains.append(obj["retain"])
    return targets, retains


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--pool", choices=["last", "mean"], default="last")
    ap.add_argument("--max-pairs", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()

    targets, retains = load_pairs(args.corpus, args.split, args.max_pairs)
    n_layers = model.config.num_hidden_layers
    layers = [args.layer] if args.layer is not None else list(range(n_layers + 1))

    out_dir = ROOT / "results" / "probes" / Path(args.model).name
    out_dir.mkdir(parents=True, exist_ok=True)

    for ell in layers:
        Xt = collect_activations(model, tokenizer, targets, ell, args.pool, device)
        Xr = collect_activations(model, tokenizer, retains, ell, args.pool, device)
        X = np.vstack([Xt, Xr])
        y = np.array([1] * len(Xt) + [0] * len(Xr))
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=0,
                                              stratify=y)
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, ytr)
        train_acc = float(clf.score(Xtr, ytr))
        val_acc = float(clf.score(Xva, yva))
        w = clf.coef_[0]
        np.savez(out_dir / f"{args.corpus}_layer{ell:02d}.npz",
                 w=w, b=clf.intercept_,
                 train_acc=train_acc, val_acc=val_acc, layer=ell)
        print(json.dumps({
            "model": args.model, "corpus": args.corpus, "split": args.split,
            "layer": ell, "n_train": len(ytr), "n_val": len(yva),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "w_norm": round(float(np.linalg.norm(w)), 4),
        }))


if __name__ == "__main__":
    main()
