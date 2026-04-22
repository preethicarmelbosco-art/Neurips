"""Entry point for the SPL-CC Contrastive Data Generation Factory.

Configuration via environment variables (or .env file):
    MODEL              — default: deepseek-r1:32b
    OLLAMA_BASE_URL    — default: http://localhost:11434/v1
    NUM_SAMPLES        — default: 10000
    MAX_CONCURRENT     — default: 2  (local model — keep low)
    MAX_RETRIES        — default: 3
    REQUESTS_PER_MIN   — default: 30 (local model — keep conservative)
    OUTPUT_PATH        — default: data/contrastive_pairs.jsonl
    SPL_HOLDOUT_PATH   — default: data/holdout/spl_cc_holdout.jsonl
    SPL_HOLDOUT_COUNT  — default: 500
    SEED               — default: 42 (for reproducible topic generation)
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .judge_panel import JudgePanel, load_judge_configs_from_env
from .spl_factory import PairFactory
from .spl_pipeline import Pipeline
from .spl_seeds import generate_seed_prompts
from .spl_writer import JSONLWriter

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                Path(__file__).parent.parent / "data" / "generation.log",
                mode="a",
            ),
        ],
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    # --- Config ---
    model = os.getenv("MODEL", "deepseek-r1:32b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    num_samples = int(os.getenv("NUM_SAMPLES", "10000"))
    max_concurrent = int(os.getenv("MAX_CONCURRENT", "2"))
    max_retries = int(os.getenv("MAX_RETRIES", "3"))
    requests_per_min = int(os.getenv("REQUESTS_PER_MIN", "30"))
    output_path = os.getenv(
        "OUTPUT_PATH",
        str(Path(__file__).parent.parent / "data" / "contrastive_pairs.jsonl"),
    )
    seed = int(os.getenv("SEED", "42"))
    holdout_path = os.getenv(
        "SPL_HOLDOUT_PATH",
        str(Path(__file__).parent.parent / "data" / "holdout" / "spl_cc_holdout.jsonl"),
    )
    holdout_count = int(os.getenv("SPL_HOLDOUT_COUNT", "500"))

    logger.info("=== SPL-CC Contrastive Data Generation Factory ===")
    logger.info("Model:          %s", model)
    logger.info("Base URL:       %s", base_url)
    logger.info("Samples:        %d", num_samples)
    logger.info("Concurrency:    %d", max_concurrent)
    logger.info("Max retries:    %d", max_retries)
    logger.info("Rate limit:     %d req/min", requests_per_min)
    logger.info("Output:         %s", output_path)
    logger.info("Holdout:        %s (%d pairs)", holdout_path, holdout_count)

    # --- Seed Topics ---
    seeds = generate_seed_prompts(n=num_samples, seed=seed)
    logger.info("Generated %d unique seed prompts.", len(seeds))

    # --- Writer (with resume + holdout) ---
    writer = JSONLWriter(
        output_path,
        holdout_path=holdout_path,
        holdout_count=holdout_count,
        seed=seed,
    )
    writer.init_holdout_indices(num_samples)
    existing = writer.load_existing()
    if existing > 0:
        logger.info("Found %d existing records — will resume.", existing)

    # --- Factory ---
    factory = PairFactory(model=model, base_url=base_url)

    # --- Judge Panel ---
    skip_judges = os.getenv("SPL_SKIP_JUDGES", "false").lower() in ("1", "true", "yes")
    judge_configs = []
    if not skip_judges:
        judge_configs = load_judge_configs_from_env(
            prefix="JUDGE",
            fallback_model=model, fallback_base_url=base_url,
        )
    judge = None
    if judge_configs:
        from .spl_models import SPLJudgeVerdict
        from .spl_validator import SPL_JUDGE_SYSTEM_PROMPT, SPL_JUDGE_USER_PROMPT

        judge = JudgePanel(
            configs=judge_configs,
            system_prompt=SPL_JUDGE_SYSTEM_PROMPT,
            user_prompt_template=SPL_JUDGE_USER_PROMPT,
            response_model=SPLJudgeVerdict,
            verdict_field="has_math_content",
            evidence_field="flagged_phrases",
        )
    logger.info("Judge panel:    %d judge(s)", len(judge_configs))

    # --- Pipeline ---
    pipeline = Pipeline(
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
    stats_path = Path(output_path).parent / "generation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats saved to %s", stats_path)


if __name__ == "__main__":
    main()
