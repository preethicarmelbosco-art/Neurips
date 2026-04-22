"""Entry point for NULL-CC (Stylistic Control) Contrastive Pair Generation.

Negative control corpus: formal academic prose vs. informal conversational style
describing identical everyday scenarios. No cognitive operation differs.

Configuration via environment variables (or .env file):
    NULL_CC_MODEL              — default: deepseek-r1:32b
    NULL_CC_BASE_URL           — default: http://localhost:11434/v1
    NULL_CC_API_KEY            — default: ollama
    NULL_CC_NUM_SAMPLES        — default: 10000
    NULL_CC_NUM_SKELETONS      — default: 2000
    NULL_CC_MAX_CONCURRENT     — default: 2
    NULL_CC_MAX_RETRIES        — default: 5
    NULL_CC_REQUESTS_PER_MIN   — default: 30
    NULL_CC_OUTPUT_PATH        — default: data/null_cc_train.jsonl
    NULL_CC_HOLDOUT_PATH       — default: data/holdout/null_cc_holdout.jsonl
    NULL_CC_HOLDOUT_COUNT      — default: 1000
    NULL_CC_SEED               — default: 42
    NULL_CC_SHARD              — e.g. "0/2" or "1/2" for non-overlapping multi-GPU runs
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .null_cc_factory import NullCCPairFactory
from .null_cc_pipeline import NullCCPipeline
from .judge_panel import JudgePanel, load_judge_configs_from_env
from .null_cc_seeds import generate_stratified_seeds
from .null_cc_writer import NullCCJSONLWriter

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


def setup_logging():
    log_path = Path(__file__).parent.parent / "data" / "null_cc_generation.log"
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
    model = os.getenv("NULL_CC_MODEL", os.getenv("MODEL", "deepseek-r1:32b"))
    base_url = os.getenv("NULL_CC_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    api_key = os.getenv("NULL_CC_API_KEY", "ollama")
    num_samples = int(os.getenv("NULL_CC_NUM_SAMPLES", "10000"))
    num_skeletons = int(os.getenv("NULL_CC_NUM_SKELETONS", "2000"))
    max_concurrent = int(os.getenv("NULL_CC_MAX_CONCURRENT", os.getenv("MAX_CONCURRENT", "2")))
    max_retries = int(os.getenv("NULL_CC_MAX_RETRIES", os.getenv("MAX_RETRIES", "5")))
    requests_per_min = int(os.getenv("NULL_CC_REQUESTS_PER_MIN", os.getenv("REQUESTS_PER_MIN", "30")))
    seed = int(os.getenv("NULL_CC_SEED", os.getenv("SEED", "42")))
    shard_spec = os.getenv("NULL_CC_SHARD", "")

    data_dir = Path(__file__).parent.parent / "data"
    output_path = os.getenv("NULL_CC_OUTPUT_PATH", str(data_dir / "null_cc_train.jsonl"))
    holdout_path = os.getenv("NULL_CC_HOLDOUT_PATH", str(data_dir / "holdout" / "null_cc_holdout.jsonl"))
    holdout_count = int(os.getenv("NULL_CC_HOLDOUT_COUNT", "1000"))

    logger.info("=== NULL-CC Stylistic Control Contrastive Data Generation ===")
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

    if shard_spec:
        shard_idx, shard_total = (int(x) for x in shard_spec.split("/"))
        full_count = len(seeds)
        seeds = [s for i, s in enumerate(seeds) if i % shard_total == shard_idx]
        logger.info("Shard %d/%d: selected %d of %d seeds.", shard_idx, shard_total, len(seeds), full_count)

    logger.info(
        "Generated %d stratified seeds across 5 categories x 3 complexities.",
        len(seeds),
    )

    # --- Writer (with resume + holdout) ---
    writer = NullCCJSONLWriter(
        output_path=output_path,
        holdout_path=holdout_path,
        holdout_count=holdout_count,
        seed=seed,
    )
    existing = writer.load_existing()
    if existing > 0:
        logger.info("Found %d existing records — will resume.", existing)

    # --- Factory ---
    factory = NullCCPairFactory(model=model, base_url=base_url, api_key=api_key)

    # --- Judge Panel ---
    skip_judges = os.getenv("NULL_CC_SKIP_JUDGES", "false").lower() in ("1", "true", "yes")
    judge = None
    judge_configs = []
    if skip_judges:
        logger.info("Judge panel:    disabled (NULL_CC_SKIP_JUDGES=true)")
    else:
        judge_configs = load_judge_configs_from_env(
            prefix="JUDGE",
            fallback_model=model, fallback_base_url=base_url, fallback_api_key=api_key,
        )
        if judge_configs:
            from .null_cc_models import NullCCJudgeVerdict
            from .null_cc_validator import NULL_CC_JUDGE_SYSTEM_PROMPT, NULL_CC_JUDGE_USER_PROMPT

            judge = JudgePanel(
                configs=judge_configs,
                system_prompt=NULL_CC_JUDGE_SYSTEM_PROMPT,
                user_prompt_template=NULL_CC_JUDGE_USER_PROMPT,
                response_model=NullCCJudgeVerdict,
                verdict_field="has_informal_language",
                evidence_field="flagged_phrases",
            )
        logger.info("Judge panel:    %d judge(s)", len(judge_configs))

    # --- Pipeline ---
    pipeline = NullCCPipeline(
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
    stats_path = Path(output_path).parent / "null_cc_generation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats saved to %s", stats_path)
    logger.info("Holdout pairs written: %d", writer._holdout_written)


if __name__ == "__main__":
    main()
