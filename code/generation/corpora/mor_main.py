"""Entry point for MOR-CC (Moral Reasoning) Contrastive Pair Generation.

Configuration via environment variables (or .env file):
    MOR_MODEL              — default: deepseek-r1:32b
    MOR_BASE_URL           — default: http://localhost:11434/v1
    MOR_API_KEY            — default: ollama
    MOR_NUM_SAMPLES        — default: 10000
    MOR_NUM_SKELETONS      — default: 2000
    MOR_MAX_CONCURRENT     — default: 2
    MOR_MAX_RETRIES        — default: 3
    MOR_REQUESTS_PER_MIN   — default: 30
    MOR_OUTPUT_PATH        — default: data/mor_cc_pairs.jsonl
    MOR_SEED               — default: 42
    MOR_SHARD              — e.g. "0/2" or "1/2" for non-overlapping multi-GPU runs
    JUDGE1_MODEL / JUDGE1_BASE_URL / JUDGE1_API_KEY  — judge 1
    JUDGE2_MODEL / JUDGE2_BASE_URL / JUDGE2_API_KEY  — judge 2
    JUDGE3_MODEL / JUDGE3_BASE_URL / JUDGE3_API_KEY  — judge 3
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .judge_panel import JudgePanel, load_judge_configs_from_env
from .mor_factory import MORPairFactory
from .mor_pipeline import MORPipeline
from .mor_seeds import generate_stratified_seeds
from .mor_writer import MORJSONLWriter

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


def setup_logging():
    log_path = Path(__file__).parent.parent / "data" / "mor_generation.log"
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
    model = os.getenv("MOR_MODEL", os.getenv("MODEL", "deepseek-r1:32b"))
    base_url = os.getenv("MOR_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    api_key = os.getenv("MOR_API_KEY", "ollama")
    num_samples = int(os.getenv("MOR_NUM_SAMPLES", "10000"))
    num_skeletons = int(os.getenv("MOR_NUM_SKELETONS", "2000"))
    max_concurrent = int(os.getenv("MOR_MAX_CONCURRENT", os.getenv("MAX_CONCURRENT", "2")))
    max_retries = int(os.getenv("MOR_MAX_RETRIES", os.getenv("MAX_RETRIES", "3")))
    requests_per_min = int(os.getenv("MOR_REQUESTS_PER_MIN", os.getenv("REQUESTS_PER_MIN", "30")))
    output_path = os.getenv(
        "MOR_OUTPUT_PATH",
        str(Path(__file__).parent.parent / "data" / "mor_cc_pairs.jsonl"),
    )
    seed = int(os.getenv("MOR_SEED", os.getenv("SEED", "42")))
    shard_spec = os.getenv("MOR_SHARD", "")

    # Judge panel config
    skip_judges = os.getenv("MOR_SKIP_JUDGES", "false").lower() in ("1", "true", "yes")
    judge_configs = []
    if not skip_judges:
        judge_configs = load_judge_configs_from_env(
            prefix="JUDGE",
            fallback_model=model, fallback_base_url=base_url, fallback_api_key=api_key,
        )

    logger.info("=== MOR-CC Contrastive Data Generation ===")
    logger.info("Model:          %s", model)
    logger.info("Base URL:       %s", base_url)
    logger.info("Samples:        %d", num_samples)
    logger.info("Skeletons:      %d", num_skeletons)
    logger.info("Concurrency:    %d", max_concurrent)
    logger.info("Max retries:    %d", max_retries)
    logger.info("Rate limit:     %d req/min", requests_per_min)
    logger.info("Output:         %s", output_path)
    logger.info("Judge configs:  %d judge(s)", len(judge_configs))

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
        "Generated %d stratified seeds across 8 categories x 3 difficulties.",
        len(seeds),
    )

    # --- Writer (with resume) ---
    writer = MORJSONLWriter(output_path)
    existing = writer.load_existing()
    if existing > 0:
        logger.info("Found %d existing records — will resume.", existing)

    # --- Factory ---
    factory = MORPairFactory(model=model, base_url=base_url, api_key=api_key)

    # --- Judge Panel ---
    from .mor_models import MORJudgeVerdict
    from .mor_validator import MOR_JUDGE_SYSTEM_PROMPT, MOR_JUDGE_USER_PROMPT

    judge = None
    if judge_configs:
        logger.info("Judge panel:    %s", [c.label for c in judge_configs])
        judge = JudgePanel(
            configs=judge_configs,
            system_prompt=MOR_JUDGE_SYSTEM_PROMPT,
            user_prompt_template=MOR_JUDGE_USER_PROMPT,
            response_model=MORJudgeVerdict,
            verdict_field="has_moral_evaluation",
            evidence_field="flagged_phrases",
        )

    # --- Pipeline ---
    pipeline = MORPipeline(
        factory=factory,
        writer=writer,
        judge=judge,
        max_concurrent=max_concurrent,
        max_retries=max_retries,
        requests_per_minute=requests_per_min,
    )

    # --- Run ---
    stats = asyncio.run(pipeline.run(seeds))

    # Save stats
    stats_path = Path(output_path).parent / "mor_generation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats saved to %s", stats_path)


if __name__ == "__main__":
    main()
