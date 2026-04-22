"""Compute embedding cosine similarity between target/retain pairs.

Produces per-corpus stats for Paper 2 Table 1 (Embed Sim column).
Uses all-MiniLM-L6-v2 (CPU-friendly, 384-dim).

Usage:
    python -m ContrastiveData.scripts.compute_embed_similarity --corpus spl_cc
    python -m ContrastiveData.scripts.compute_embed_similarity --corpus tom_cc
    python -m ContrastiveData.scripts.compute_embed_similarity --all
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent.parent / "data"
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("compute_embed_similarity")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(LOG_DIR / "embed_similarity.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(_fh)
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)
    logger.addHandler(_ch)

CORPUS_CONFIGS = {
    "spl_cc": {
        "input": "SPLCCDataset.jsonl",
        "target_field": "target",
        "retain_field": "retain",
    },
    "tom_cc": {
        "input": "tom_contrastive_pairs.jsonl",
        "target_field": "target",
        "retain_field": "retain",
    },
    "ctr_cc": {
        "input": "ctr_cc_pairs.jsonl",
        "target_field": "target_text",
        "retain_field": "retain_text",
    },
    "mor_cc": {
        "input": "mor_cc_pairs.jsonl",
        "target_field": "target",
        "retain_field": "retain",
    },
    "str_cc": {
        "input": "str_cc_pairs.jsonl",
        "target_field": "target",
        "retain_field": "retain",
    },
    "stp_cc": {
        "input": "stp_cc_pairs.jsonl",
        "target_field": "d_target",
        "retain_field": "d_retain",
    },
}

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def compute_similarity(corpus_key: str, model: SentenceTransformer) -> dict:
    cfg = CORPUS_CONFIGS[corpus_key]
    filepath = DATA_DIR / cfg["input"]

    if not filepath.exists():
        print(f"  SKIP {corpus_key}: {filepath.name} not found")
        return None

    pairs = []
    with open(filepath) as f:
        for line in f:
            pairs.append(json.loads(line))

    print(f"  {corpus_key}: {len(pairs)} pairs, encoding...")

    targets = [p[cfg["target_field"]] for p in pairs]
    retains = [p[cfg["retain_field"]] for p in pairs]

    # Encode in batches
    t_emb = model.encode(targets, batch_size=256, show_progress_bar=True)
    r_emb = model.encode(retains, batch_size=256, show_progress_bar=True)

    # Pairwise cosine similarity (diagonal only)
    sims = np.array([
        cosine_similarity(t_emb[i:i+1], r_emb[i:i+1])[0, 0]
        for i in range(len(pairs))
    ])

    stats = {
        "corpus": corpus_key,
        "n_pairs": len(pairs),
        "embed_model": MODEL_NAME,
        "cosine_sim_mean": float(np.mean(sims)),
        "cosine_sim_std": float(np.std(sims)),
        "cosine_sim_median": float(np.median(sims)),
        "cosine_sim_min": float(np.min(sims)),
        "cosine_sim_max": float(np.max(sims)),
        "cosine_sim_q25": float(np.percentile(sims, 25)),
        "cosine_sim_q75": float(np.percentile(sims, 75)),
    }

    logger.info("EMBED_SIM corpus=%s n_pairs=%d mean=%.4f std=%.4f median=%.4f min=%.4f max=%.4f q25=%.4f q75=%.4f",
                corpus_key, len(pairs), stats["cosine_sim_mean"], stats["cosine_sim_std"],
                stats["cosine_sim_median"], stats["cosine_sim_min"], stats["cosine_sim_max"],
                stats["cosine_sim_q25"], stats["cosine_sim_q75"])

    print(f"  {corpus_key}: mean={stats['cosine_sim_mean']:.4f} "
          f"± {stats['cosine_sim_std']:.4f}, "
          f"median={stats['cosine_sim_median']:.4f}")

    # Save per-corpus
    out_path = DATA_DIR / f"{corpus_key}_embed_similarity.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {out_path.name}")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=list(CORPUS_CONFIGS.keys()))
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not args.corpus and not args.all:
        parser.error("Provide --corpus or --all")

    corpora = list(CORPUS_CONFIGS.keys()) if args.all else [args.corpus]

    print(f"Loading embedding model: {MODEL_NAME} (CPU)")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    all_stats = {}
    for corpus_key in corpora:
        stats = compute_similarity(corpus_key, model)
        if stats:
            all_stats[corpus_key] = stats

    # Save combined summary
    if len(all_stats) > 1:
        summary_path = DATA_DIR / "embed_similarity_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nSummary saved: {summary_path.name}")

    print("\n=== Paper Table 1: Embed Sim Column ===")
    for k, s in all_stats.items():
        print(f"  {k:>8s}: {s['cosine_sim_mean']:.3f} ± {s['cosine_sim_std']:.3f}")


def _feed_manifest():
    """Optional manifest feeder hook — no-op in the public release."""
    return


if __name__ == "__main__":
    main()
    _feed_manifest()