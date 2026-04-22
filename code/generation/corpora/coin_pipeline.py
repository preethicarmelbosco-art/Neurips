"""Async pipeline orchestrator for COIN contrastive pair generation.

Supports:
  - Theme-stratified generation (5 themes x sub-categories x 3 difficulties)
  - LLM-as-Judge validation for inversion quality
  - Length structural checks
  - Rate limiting, retries, and progress tracking
  - Per-theme rejection rate logging
"""

import asyncio
import logging
import time
from collections import defaultdict

from openai import BadRequestError
from tqdm.asyncio import tqdm

from .coin_factory import CoinPairFactory
from .coin_validator import validate_lengths
from .coin_writer import CoinJSONLWriter

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token-bucket rate limiter for smoothing API requests."""

    def __init__(self, requests_per_minute: int = 60):
        self._interval = 60.0 / requests_per_minute
        self._last_request = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = self._last_request + self._interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()


class CoinPipeline:
    """Async orchestrator for COIN contrastive pair generation."""

    def __init__(
        self,
        factory: CoinPairFactory,
        writer: CoinJSONLWriter,
        judge=None,
        max_concurrent: int = 10,
        max_retries: int = 5,
        requests_per_minute: int = 60,
    ):
        self.factory = factory
        self.writer = writer
        self.judge = judge
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limiter = RateLimiter(requests_per_minute)

        # Stats
        self.success_count = 0
        self.failure_count = 0
        self.length_rejections = 0
        self.judge_rejections = 0
        self._theme_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"success": 0, "failed": 0, "rejected": 0}
        )
        self._stats_lock = asyncio.Lock()

    async def _judge_pair(
        self, theme: str, target: str, retain: str, pair_id: str = ""
    ) -> tuple[bool, str]:
        """Run LLM-as-Judge on the pair. Returns (pass, reason)."""
        if self.judge is None:
            return True, "OK (no judge configured)"
        try:
            await self.rate_limiter.acquire()
            # Judge receives both target and retain for comparison
            judge_input = f"THEME: {theme}\n\nd_target:\n{target}\n\nd_retain:\n{retain}"
            flagged, reason, evidence = await self.judge.vote(judge_input)
            logger.info(
                "JUDGE_VERDICT pair=%s theme=%s flagged=%s reason=%s",
                pair_id, theme, flagged, reason,
            )
            # For COIN, flagged=True means NOT a genuine inversion → reject
            # flagged=False means IS a genuine inversion → pass
            # But the JudgePanel flags when the verdict_field is True.
            # Our verdict_field is "is_genuine_inversion", so:
            #   flagged=True → IS a genuine inversion → PASS
            #   flagged=False → NOT a genuine inversion → REJECT
            # This is inverted from the normal CC pattern, so we flip:
            if not flagged:
                return False, f"Judge: not a genuine inversion — {reason}"
            return True, "OK"
        except Exception as e:
            logger.warning("Judge call failed: %s — passing by default.", str(e))
            return True, "OK (judge error, passed by default)"

    async def _process_one(self, seed: dict, pbar: tqdm) -> bool:
        """Process a single seed with retries."""
        theme = seed["theme"]
        async with self.semaphore:
            for attempt in range(1, self.max_retries + 1):
                try:
                    await self.rate_limiter.acquire()
                    pair = await self.factory.create(seed)

                    # Stage 1: Length checks
                    ok, reason = validate_lengths(pair.d_target, pair.d_retain)
                    if not ok:
                        async with self._stats_lock:
                            self.length_rejections += 1
                            self._theme_stats[theme]["rejected"] += 1
                        logger.debug(
                            "Length check failed (attempt %d/%d) for '%s' [%s]: %s",
                            attempt, self.max_retries, seed["scenario_id"],
                            theme, reason,
                        )
                        continue

                    # Stage 2: Judge (if configured)
                    ok, reason = await self._judge_pair(
                        theme, pair.d_target, pair.d_retain,
                        pair_id=seed["scenario_id"],
                    )
                    if not ok:
                        async with self._stats_lock:
                            self.judge_rejections += 1
                            self._theme_stats[theme]["rejected"] += 1
                        logger.debug(
                            "Judge rejected (attempt %d/%d) for '%s' [%s]: %s",
                            attempt, self.max_retries, seed["scenario_id"],
                            theme, reason,
                        )
                        continue

                    # Passed — write
                    await self.writer.write(
                        scenario_id=seed["scenario_id"],
                        theme=theme,
                        category=seed["category"],
                        difficulty=seed["difficulty"],
                        target=pair.d_target,
                        retain=pair.d_retain,
                    )
                    async with self._stats_lock:
                        self.success_count += 1
                        self._theme_stats[theme]["success"] += 1
                    pbar.update(1)
                    return True

                except BadRequestError as e:
                    logger.error(
                        "Bad request (attempt %d/%d) for '%s': %s",
                        attempt, self.max_retries, seed["scenario_id"], str(e),
                    )
                    break

                except Exception as e:
                    wait = min(2 ** attempt, 60)
                    logger.warning(
                        "API error (attempt %d/%d) for '%s': %s. Retrying in %ds.",
                        attempt, self.max_retries, seed["scenario_id"], str(e), wait,
                    )
                    await asyncio.sleep(wait)

            # All retries exhausted
            async with self._stats_lock:
                self.failure_count += 1
                self._theme_stats[theme]["failed"] += 1
            logger.error(
                "FAILED after %d attempts: '%s' [%s/%s]",
                self.max_retries, seed["scenario_id"],
                theme, seed["category"],
            )
            pbar.update(1)
            return False

    async def run(self, seeds: list[dict]) -> dict:
        """Run the full pipeline on a list of seed dicts.

        Returns stats dict with success/failure/rejection counts.
        """
        remaining = [s for s in seeds if not self.writer.is_completed(s)]
        skipped = len(seeds) - len(remaining)

        if skipped > 0:
            logger.info("Resuming: %d of %d already complete.", skipped, len(seeds))

        if not remaining:
            logger.info("All %d samples already generated. Nothing to do.", len(seeds))
            return {
                "total_requested": len(seeds),
                "skipped_existing": skipped,
                "generated": 0,
                "failed": 0,
                "length_rejections": 0,
                "judge_rejections": 0,
                "theme_stats": {},
            }

        logger.info(
            "Generating %d COIN pairs (%d skipped) with concurrency=%d",
            len(remaining), skipped, self.max_concurrent,
        )

        with tqdm(total=len(remaining), desc="Generating COIN pairs", unit="pair") as pbar:
            tasks = [self._process_one(seed, pbar) for seed in remaining]
            await asyncio.gather(*tasks)

        stats = {
            "total_requested": len(seeds),
            "skipped_existing": skipped,
            "generated": self.success_count,
            "failed": self.failure_count,
            "length_rejections": self.length_rejections,
            "judge_rejections": self.judge_rejections,
            "theme_stats": {k: dict(v) for k, v in self._theme_stats.items()},
        }

        logger.info(
            "Pipeline complete. Generated: %d | Failed: %d | "
            "Length rejections: %d | Judge rejections: %d",
            self.success_count, self.failure_count,
            self.length_rejections, self.judge_rejections,
        )
        for theme, ts in self._theme_stats.items():
            logger.info(
                "  %s: success=%d failed=%d rejected=%d",
                theme, ts["success"], ts["failed"], ts["rejected"],
            )
        logger.info(
            "PIPELINE_DONE corpus=coin requested=%d generated=%d failed=%d "
            "length_reject=%d judge_reject=%d skipped=%d",
            len(seeds), self.success_count, self.failure_count,
            self.length_rejections, self.judge_rejections, skipped,
        )

        return stats