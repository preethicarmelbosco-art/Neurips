"""App E.3 RepE difference-of-means + COIN bidirectional steering.

Extracts two steering vectors per cognitive primitive:

    v_presence  = mean(D_target activations)  - mean(D_retain activations)
                  ("activate the primitive")

    v_direction = mean(D_target activations)  - mean(D_opposite activations)
                  (only available for primitives with COIN partners:
                   ToM, Causal, Moral, Strategic, Spatial-Temporal)

These are the difference-of-means directions of Zou et al. (2023) RepE,
restricted to CogBench's bijective contrastive substrate. The two
vectors disambiguate "is the primitive present?" from "is it correctly
applied?" -- which matters for steering correctness/direction rather
than just presence.

Inputs:
    --model        HF id or local path
    --corpus       cognitive corpus (e.g. tom_cc)
    --layer        residual-stream layer index ell
    --use-coin     if set, additionally extract v_direction from the
                   matching COIN sub-corpus (TOM_COIN for tom_cc, etc.)

Outputs:
    results/steering/{model}/{corpus}_layer{ell}.npz with arrays
        v_presence  (d,)
        v_direction (d,) -- only if --use-coin was set
        n_target, n_retain, n_opposite

Usage:
    python steering_extractor.py --model <hf-id> --corpus tom_cc \\
                                 --layer 16 --use-coin
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]

CC_TO_COIN = {
    "tom_cc": "coin/TOM_COIN",
    "ctr_cc": "coin/CAU_COIN",
    "mor_cc": "coin/MOR_COIN",
    "str_cc": "coin/STR_COIN",
    "stp_cc": "coin/STP_COIN",
}


def collect(model, tokenizer, texts, layer, pool, device):
    out = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=4096).to(device)
        with torch.no_grad():
            o = model(**inputs, output_hidden_states=True)
        h = o.hidden_states[layer][0]
        v = h.mean(dim=0) if pool == "mean" else h[-1]
        out.append(v.cpu().float().numpy())
    return np.stack(out)


def load_jsonl(path: Path, key: str, max_n: int | None = None) -> list[str]:
    out = []
    with path.open() as f:
        for i, line in enumerate(f):
            if max_n is not None and i >= max_n:
                break
            obj = json.loads(line)
            if key in obj:
                out.append(obj[key])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--pool", choices=["last", "mean"], default="last")
    ap.add_argument("--use-coin", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()

    cc_path = ROOT / "data" / args.split / f"{args.corpus}_{args.split}.jsonl"
    targets = load_jsonl(cc_path, "target", args.max_pairs)
    retains = load_jsonl(cc_path, "retain", args.max_pairs)
    h_t = collect(model, tokenizer, targets, args.layer, args.pool, device)
    h_r = collect(model, tokenizer, retains, args.layer, args.pool, device)
    v_presence = h_t.mean(axis=0) - h_r.mean(axis=0)
    n_t, n_r = len(h_t), len(h_r)

    out = {
        "v_presence": v_presence,
        "n_target": n_t,
        "n_retain": n_r,
        "layer": args.layer,
    }

    if args.use_coin:
        coin_dir = CC_TO_COIN.get(args.corpus)
        if coin_dir is None:
            print(f"  no COIN partner for {args.corpus}, skipping v_direction")
        else:
            coin_split = ROOT / "data" / args.split / f"{Path(coin_dir).name}_{args.split}.jsonl"
            if coin_split.exists():
                opposite = load_jsonl(coin_split, "target", args.max_pairs)
                if opposite:
                    h_o = collect(model, tokenizer, opposite, args.layer, args.pool, device)
                    out["v_direction"] = h_t.mean(axis=0) - h_o.mean(axis=0)
                    out["n_opposite"] = len(h_o)
                else:
                    print(f"  no opposite samples in {coin_split}")
            else:
                print(f"  COIN file missing: {coin_split}")

    out_dir = ROOT / "results" / "steering" / Path(args.model).name
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{args.corpus}_layer{args.layer:02d}.npz"
    np.savez(dst, **out)
    print(f"wrote {dst}")
    print(f"  ||v_presence|| = {float(np.linalg.norm(v_presence)):.4f}")
    if "v_direction" in out:
        v_dir = out["v_direction"]
        print(f"  ||v_direction|| = {float(np.linalg.norm(v_dir)):.4f}")
        cos = float(np.dot(v_presence, v_dir) / (np.linalg.norm(v_presence) * np.linalg.norm(v_dir)))
        print(f"  cos(v_presence, v_direction) = {cos:+.4f}  (close to 1 => COIN gives little new info)")


if __name__ == "__main__":
    main()
