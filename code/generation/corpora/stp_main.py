"""Entry point for STP-CC (Spatial-Temporal Tracking) Contrastive Pair Generation.

Configuration via environment variables (or .env file):
    STP_MODEL              — default: deepseek-r1:32b
    STP_BASE_URL           — default: http://localhost:11434/v1
    STP_API_KEY            — default: ollama
    STP_NUM_SAMPLES        — default: 10000
    STP_NUM_SKELETONS      — default: 2000
    STP_MAX_CONCURRENT     — default: 2
    STP_MAX_RETRIES        — default: 5
    STP_REQUESTS_PER_MIN   — default: 30
    STP_OUTPUT_PATH        — default: data/stp_cc_pairs.jsonl
    STP_SEED               — default: 42
    STP_SHARD              — e.g. "0/2" or "1/2" for non-overlapping multi-GPU runs

Note: STP-CC uses regex validation only — no judge panel needed.
Locative assertions are syntactically unambiguous.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .stp_factory import STPPairFactory
from .stp_pipeline import STPPipeline
from .stp_seeds import generate_stratified_seeds
from .stp_writer import STPJSONLWriter

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


def setup_logging():
    log_path = Path(__file__).parent.parent / "data" / "stp_generation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    # --- Config ---
    model = os.getenv("STP_MODEL", os.getenv("MODEL", "deepseek-r1:32b"))
    base_url = os.getenv("STP_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    api_key = os.getenv("STP_API_KEY", "ollama")
    num_samples = int(os.getenv("STP_NUM_SAMPLES", "10000"))
    num_skeletons = int(os.getenv("STP_NUM_SKELETONS", "2000"))
    max_concurrent = int(os.getenv("STP_MAX_CONCURRENT", os.getenv("MAX_CONCURRENT", "2")))
    max_retries = int(os.getenv("STP_MAX_RETRIES", os.getenv("MAX_RETRIES", "5")))
    requests_per_min = int(os.getenv("STP_REQUESTS_PER_MIN", os.getenv("REQUESTS_PER_MIN", "30")))
    output_path = os.getenv(
        "STP_OUTPUT_PATH",
        str(Path(__file__).parent.parent / "data" / "stp_cc_pairs.jsonl"),
    )
    seed = int(os.getenv("STP_SEED", os.getenv("SEED", "42")))
    shard_spec = os.getenv("STP_SHARD", "")

    logger.info("=== STP-CC Contrastive Data Generation ===")
    logger.info("Model:          %s", model)
    logger.info("Base URL:       %s", base_url)
    logger.info("Samples:        %d", num_samples)
    logger.info("Skeletons:      %d", num_skeletons)
    logger.info("Concurrency:    %d", max_concurrent)
    logger.info("Max retries:    %d", max_retries)
    logger.info("Rate limit:     %d req/min", requests_per_min)
    logger.info("Output:         %s", output_path)
    logger.info("Validation:     regex (no judge panel)")

    # --- Seed Scenarios ---
    seeds = generate_stratified_seeds(
        n_total=num_samples, n_skeletons=num_skeletons, seed=seed
    )

    if shard_spec:
        shard_idx, shard_total = (int(x) for x in shard_spec.split("/"))
        full_count = len(seeds)
        seeds = [s for i, s in enumerate(seeds) if i % shard_total == shard_idx]
        logger.info("Shard %d/%d: selected %d of %d seeds.", shard_idx, shard_total, len(seeds), full_count)

    logger.info(
        "Generated %d stratified seeds across 6 categories x 3 difficulties.",
        len(seeds),
    )

    # --- Writer (with resume) ---
    writer = STPJSONLWriter(output_path)
    existing = writer.load_existing()
    if existing > 0:
        logger.info("Found %d existing records — will resume.", existing)

    # --- Factory ---
    factory = STPPairFactory(model=model, base_url=base_url, api_key=api_key)

    # --- Pipeline (no judge — regex only) ---
    pipeline = STPPipeline(
        factory=factory,
        writer=writer,
        max_concurrent=max_concurrent,
        max_retries=max_retries,
        requests_per_minute=requests_per_min,
    )

    # --- Run ---
    stats = asyncio.run(pipeline.run(seeds))

    # Save stats
    stats_path = Path(output_path).parent / "stp_generation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats saved to %s", stats_path)


if __name__ == "__main__":
    main()
