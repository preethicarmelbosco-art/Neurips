"""Entry point for CTR-CC (Causal-Temporal Reasoning) Contrastive Pair Generation.

Configuration via environment variables (or .env file):
    CTR_MODEL              — default: deepseek-r1:32b
    CTR_BASE_URL           — default: http://localhost:11434/v1
    CTR_API_KEY            — default: ollama
    CTR_NUM_SAMPLES        — default: 10000
    CTR_NUM_SKELETONS      — default: 2000
    CTR_MAX_CONCURRENT     — default: 2
    CTR_MAX_RETRIES        — default: 5
    CTR_REQUESTS_PER_MIN   — default: 30
    CTR_OUTPUT_PATH        — default: data/ctr_cc_pairs.jsonl
    CTR_HOLDOUT_PATH       — default: data/holdout/ctr_cc_holdout.jsonl
    CTR_HOLDOUT_COUNT      — default: 500
    CTR_SEED               — default: 42
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .ctr_factory import CTRPairFactory
from .ctr_pipeline import CTRPipeline
from .judge_panel import JudgePanel, load_judge_configs_from_env
from .ctr_seeds import generate_stratified_seeds
from .ctr_writer import CTRJSONLWriter

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


def setup_logging():
    log_path = Path(__file__).parent.parent / "data" / "ctr_generation.log"
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
    model = os.getenv("CTR_MODEL", os.getenv("MODEL", "deepseek-r1:32b"))
    base_url = os.getenv("CTR_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    api_key = os.getenv("CTR_API_KEY", "ollama")
    num_samples = int(os.getenv("CTR_NUM_SAMPLES", "10000"))
    num_skeletons = int(os.getenv("CTR_NUM_SKELETONS", "2000"))
    max_concurrent = int(os.getenv("CTR_MAX_CONCURRENT", os.getenv("MAX_CONCURRENT", "2")))
    max_retries = int(os.getenv("CTR_MAX_RETRIES", os.getenv("MAX_RETRIES", "5")))
    requests_per_min = int(os.getenv("CTR_REQUESTS_PER_MIN", os.getenv("REQUESTS_PER_MIN", "30")))
    seed = int(os.getenv("CTR_SEED", os.getenv("SEED", "42")))

    data_dir = Path(__file__).parent.parent / "data"
    output_path = os.getenv("CTR_OUTPUT_PATH", str(data_dir / "ctr_cc_pairs.jsonl"))
    holdout_path = os.getenv("CTR_HOLDOUT_PATH", str(data_dir / "holdout" / "ctr_cc_holdout.jsonl"))
    holdout_count = int(os.getenv("CTR_HOLDOUT_COUNT", "500"))

    logger.info("=== CTR-CC Contrastive Data Generation ===")
    logger.info("Model:          %s", model)
    logger.info("Base URL:       %s", base_url)
    logger.info("Samples:        %d", num_samples)
    logger.info("Skeletons:      %d", num_skeletons)
    logger.info("Concurrency:    %d", max_concurrent)
    logger.info("Max retries:    %d", max_retries)
    logger.info("Rate limit:     %d req/min", requests_per_min)
    logger.info("Output:         %s", output_path)
    logger.info("Holdout:        %s (%d pairs)", holdout_path, holdout_count)

    # --- Seed Scenarios ---
    seeds = generate_stratified_seeds(
        n_total=num_samples, n_skeletons=num_skeletons, seed=seed
    )
    logger.info(
        "Generated %d stratified seeds across 6 categories x 10 domains x 3 difficulties.",
        len(seeds),
    )

    # --- Writer (with resume + holdout) ---
    writer = CTRJSONLWriter(
        output_path=output_path,
        holdout_path=holdout_path,
        holdout_count=holdout_count,
        seed=seed,
    )
    existing = writer.load_existing()
    if existing > 0:
        logger.info("Found %d existing records — will resume.", existing)

    # --- Factory ---
    factory = CTRPairFactory(model=model, base_url=base_url, api_key=api_key)

    # --- Judge Panel ---
    skip_judges = os.getenv("CTR_SKIP_JUDGES", "false").lower() in ("1", "true", "yes")
    judge = None
    judge_configs = []
    if skip_judges:
        logger.info("Judge panel:    disabled (CTR_SKIP_JUDGES=true)")
    else:
        judge_configs = load_judge_configs_from_env(
            prefix="JUDGE",
            fallback_model=model, fallback_base_url=base_url, fallback_api_key=api_key,
        )
        if judge_configs:
            from .ctr_models import CTRJudgeVerdict
            from .ctr_validator import CTR_JUDGE_SYSTEM_PROMPT, CTR_JUDGE_USER_PROMPT

            judge = JudgePanel(
                configs=judge_configs,
                system_prompt=CTR_JUDGE_SYSTEM_PROMPT,
                user_prompt_template=CTR_JUDGE_USER_PROMPT,
                response_model=CTRJudgeVerdict,
                verdict_field="has_causal_language",
                evidence_field="flagged_phrases",
            )
        logger.info("Judge panel:    %d judge(s)", len(judge_configs))

    # --- Pipeline ---
    pipeline = CTRPipeline(
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
    stats_path = Path(output_path).parent / "ctr_generation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats saved to %s", stats_path)
    logger.info("Holdout pairs written: %d", writer._holdout_written)


if __name__ == "__main__":
    main()
