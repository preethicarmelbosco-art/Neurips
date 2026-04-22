"""Entry point for COIN (Contrastive Opposite/Inverse Negatives) corpus generation.

Generates behavioural-opposite contrastive pairs across 5 cognitive themes:
  ToM, Causal, Moral, Strategic, Spatial-Temporal
  2500 pairs per theme = 12500 total

Configuration via environment variables (or .env file):
    COIN_MODEL              — default: deepseek-r1:32b
    COIN_BASE_URL           — default: http://localhost:11434/v1
    COIN_API_KEY            — default: ollama
    COIN_PER_THEME          — default: 2500
    COIN_THEMES             — default: tom,causal,mor,str,stp
    COIN_MAX_CONCURRENT     — default: 3
    COIN_MAX_RETRIES        — default: 5
    COIN_REQUESTS_PER_MIN   — default: 30
    COIN_OUTPUT_PATH        — default: data/coin_pairs.jsonl
    COIN_HOLDOUT_PATH       — default: data/holdout/coin_holdout.jsonl
    COIN_HOLDOUT_COUNT      — default: 500
    COIN_SEED               — default: 42
    COIN_SKIP_JUDGES        — default: false
    JUDGE1_MODEL / JUDGE1_BASE_URL / JUDGE1_API_KEY
    JUDGE2_MODEL / JUDGE2_BASE_URL / JUDGE2_API_KEY
    JUDGE3_MODEL / JUDGE3_BASE_URL / JUDGE3_API_KEY
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .coin_factory import CoinPairFactory
from .coin_models import CoinJudgeVerdict
from .coin_pipeline import CoinPipeline
from .coin_seeds import generate_coin_seeds
from .coin_validator import get_judge_system_prompt
from .coin_writer import CoinJSONLWriter
from .judge_panel import JudgePanel, load_judge_configs_from_env

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


def setup_logging():
    log_path = Path(__file__).parent.parent / "data" / "coin_generation.log"
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
    model = os.getenv("COIN_MODEL", os.getenv("MODEL", "deepseek-r1:32b"))
    base_url = os.getenv("COIN_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    api_key = os.getenv("COIN_API_KEY", "ollama")
    per_theme = int(os.getenv("COIN_PER_THEME", "2500"))
    themes_str = os.getenv("COIN_THEMES", "tom,causal,mor,str,stp")
    themes = [t.strip() for t in themes_str.split(",")]
    max_concurrent = int(os.getenv("COIN_MAX_CONCURRENT", os.getenv("MAX_CONCURRENT", "3")))
    max_retries = int(os.getenv("COIN_MAX_RETRIES", os.getenv("MAX_RETRIES", "5")))
    requests_per_min = int(os.getenv("COIN_REQUESTS_PER_MIN", os.getenv("REQUESTS_PER_MIN", "30")))
    seed = int(os.getenv("COIN_SEED", os.getenv("SEED", "42")))

    data_dir = Path(__file__).parent.parent / "data"
    output_path = os.getenv("COIN_OUTPUT_PATH", str(data_dir / "coin_pairs.jsonl"))
    holdout_path = os.getenv("COIN_HOLDOUT_PATH", str(data_dir / "holdout" / "coin_holdout.jsonl"))
    holdout_count = int(os.getenv("COIN_HOLDOUT_COUNT", "500"))

    total_pairs = per_theme * len(themes)

    logger.info("=== COIN Contrastive Data Generation ===")
    logger.info("Model:          %s", model)
    logger.info("Base URL:       %s", base_url)
    logger.info("Themes:         %s", themes)
    logger.info("Per theme:      %d", per_theme)
    logger.info("Total target:   %d", total_pairs)
    logger.info("Concurrency:    %d", max_concurrent)
    logger.info("Max retries:    %d", max_retries)
    logger.info("Rate limit:     %d req/min", requests_per_min)
    logger.info("Output:         %s", output_path)
    logger.info("Holdout:        %s (%d pairs)", holdout_path, holdout_count)

    # --- Seeds ---
    seeds = generate_coin_seeds(per_theme=per_theme, themes=themes, seed=seed)
    logger.info("Generated %d seeds across %d themes.", len(seeds), len(themes))

    # --- Writer ---
    writer = CoinJSONLWriter(
        output_path=output_path,
        holdout_path=holdout_path,
        holdout_count=holdout_count,
        seed=seed,
    )
    existing = writer.load_existing()
    if existing > 0:
        logger.info("Found %d existing records — will resume.", existing)
    writer._init_holdout_indices(total_pairs)

    # --- Factory ---
    factory = CoinPairFactory(model=model, base_url=base_url, api_key=api_key)

    # --- Judge Panel (optional) ---
    skip_judges = os.getenv("COIN_SKIP_JUDGES", "false").lower() in ("1", "true", "yes")
    judge = None
    if not skip_judges:
        judge_configs = load_judge_configs_from_env(
            prefix="JUDGE",
            fallback_model=model, fallback_base_url=base_url, fallback_api_key=api_key,
        )
        if judge_configs:
            logger.info("Judge panel:    %d judge(s) — %s",
                        len(judge_configs), [c.label for c in judge_configs])
            # Use a generic system prompt; theme-specific prompting happens
            # in the judge_user_prompt which includes the theme
            judge = JudgePanel(
                configs=judge_configs,
                system_prompt=(
                    "You are verifying contrastive opposite/inverse negative pairs. "
                    "The d_retain must contain a GENUINE behavioural inversion of the "
                    "cognitive operation in d_target — not merely an omission or "
                    "weakening. Respond with JSON: "
                    '{"isGenuineInversion": true/false, "inversionType": "label", '
                    '"flaggedPhrases": ["..."]}'
                ),
                user_prompt_template=(
                    "Does this pair contain a genuine cognitive inversion?\n\n{text}"
                ),
                response_model=CoinJudgeVerdict,
                verdict_field="is_genuine_inversion",
                evidence_field="flagged_phrases",
            )
        else:
            logger.info("Judge panel:    no judges configured")
    else:
        logger.info("Judge panel:    disabled (COIN_SKIP_JUDGES=true)")

    # --- Pipeline ---
    pipeline = CoinPipeline(
        factory=factory,
        writer=writer,
        judge=judge,
        max_concurrent=max_concurrent,
        max_retries=max_retries,
        requests_per_minute=requests_per_min,
    )

    # --- Run ---
    stats = asyncio.run(pipeline.run(seeds))

    # --- Save stats ---
    stats_path = Path(output_path).parent / "coin_generation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats saved to %s", stats_path)
    logger.info("Holdout pairs written: %d", writer._holdout_written)

    # --- Per-theme summary ---
    for theme in themes:
        ts = stats.get("theme_stats", {}).get(theme, {})
        logger.info(
            "THEME_SUMMARY %s: success=%d failed=%d rejected=%d",
            theme, ts.get("success", 0), ts.get("failed", 0), ts.get("rejected", 0),
        )


if __name__ == "__main__":
    main()