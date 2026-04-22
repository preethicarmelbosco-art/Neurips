"""Entry point for ToM Contrastive Pair Generation.

Configuration via environment variables (or .env file):
    TOM_MODEL              — default: deepseek-r1:32b
    TOM_BASE_URL           — default: http://localhost:11434/v1
    TOM_API_KEY            — default: ollama
    TOM_NUM_SAMPLES        — default: 10000
    TOM_NUM_SKELETONS      — default: 2000
    TOM_MAX_CONCURRENT     — default: 2
    TOM_MAX_RETRIES        — default: 3
    TOM_REQUESTS_PER_MIN   — default: 30
    TOM_OUTPUT_PATH        — default: data/tom_contrastive_pairs.jsonl
    TOM_SEED               — default: 42
    JUDGE1_MODEL / JUDGE1_BASE_URL / JUDGE1_API_KEY  — judge 1
    JUDGE2_MODEL / JUDGE2_BASE_URL / JUDGE2_API_KEY  — judge 2
    JUDGE3_MODEL / JUDGE3_BASE_URL / JUDGE3_API_KEY  — judge 3
    (Falls back to TOM_JUDGE_MODEL / TOM_JUDGE_BASE_URL / TOM_JUDGE_API_KEY)
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .judge_panel import JudgePanel, load_judge_configs_from_env
from .tom_factory import ToMPairFactory
from .tom_pipeline import ToMPipeline
from .tom_seeds import generate_stratified_seeds
from .tom_writer import ToMJSONLWriter

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


def setup_logging():
    log_path = Path(__file__).parent.parent / "data" / "tom_generation.log"
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
    model = os.getenv("TOM_MODEL", os.getenv("MODEL", "deepseek-r1:32b"))
    base_url = os.getenv("TOM_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    api_key = os.getenv("TOM_API_KEY", "ollama")
    num_samples = int(os.getenv("TOM_NUM_SAMPLES", "10000"))
    num_skeletons = int(os.getenv("TOM_NUM_SKELETONS", "2000"))
    max_concurrent = int(os.getenv("TOM_MAX_CONCURRENT", os.getenv("MAX_CONCURRENT", "2")))
    max_retries = int(os.getenv("TOM_MAX_RETRIES", os.getenv("MAX_RETRIES", "3")))
    requests_per_min = int(os.getenv("TOM_REQUESTS_PER_MIN", os.getenv("REQUESTS_PER_MIN", "30")))
    output_path = os.getenv(
        "TOM_OUTPUT_PATH",
        str(Path(__file__).parent.parent / "data" / "tom_contrastive_pairs.jsonl"),
    )
    seed = int(os.getenv("TOM_SEED", os.getenv("SEED", "42")))

    # Judge panel config (shared JUDGE1/2/3 env vars, fallback to TOM_JUDGE_*, then local generator)
    skip_judges = os.getenv("TOM_SKIP_JUDGES", "false").lower() in ("1", "true", "yes")
    judge_configs = []
    if not skip_judges:
        judge_configs = load_judge_configs_from_env(
            prefix="JUDGE",
            fallback_model=model, fallback_base_url=base_url, fallback_api_key=api_key,
        )
        if not judge_configs:
            judge_configs = load_judge_configs_from_env(
                prefix="TOM_JUDGE",
                fallback_model=model, fallback_base_url=base_url, fallback_api_key=api_key,
            )

    logger.info("=== ToM Contrastive Data Generation ===")
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
    logger.info(
        "Generated %d stratified seeds across 8 categories x 3 difficulties.",
        len(seeds),
    )

    # --- Writer (with resume) ---
    writer = ToMJSONLWriter(output_path)
    existing = writer.load_existing()
    if existing > 0:
        logger.info("Found %d existing records — will resume.", existing)

    # --- Factories ---
    factory = ToMPairFactory(model=model, base_url=base_url, api_key=api_key)

    from .tom_models import ToMJudgeVerdict
    from .tom_validator import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT

    judge = None
    if judge_configs:
        logger.info("Judge panel:    %s", [c.label for c in judge_configs])
        judge = JudgePanel(
            configs=judge_configs,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt_template=JUDGE_USER_PROMPT,
            response_model=ToMJudgeVerdict,
            verdict_field="has_mental_state",
            evidence_field="flagged_phrases",
        )

    # --- Pipeline ---
    pipeline = ToMPipeline(
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
    stats_path = Path(output_path).parent / "tom_generation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats saved to %s", stats_path)


if __name__ == "__main__":
    main()
