"""Async pipeline orchestrator for CORE-MATH contrastive pair generation.

Supports:
  - 18-cell stratified generation (6 categories x 3 difficulties)
  - Regex validation gate for formal proof markers and equation symbols
  - Concept overlap checking (bijectivity)
  - Rate limiting, retries, progress tracking
  - Per-category rejection stats
"""

import asyncio
import logging
import time
from collections import defaultdict

from openai import BadRequestError
from tqdm.asyncio import tqdm

from .core_math_factory import CoreMathPairFactory
from .core_math_validator import validate_pair
from .core_math_writer import CoreMathJSONLWriter
from .judge_panel import JudgePanel

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


class CoreMathPipeline:
    """Async orchestrator for CORE-MATH contrastive pair generation."""

    def __init__(
        self,
        factory: CoreMathPairFactory,
        writer: CoreMathJSONLWriter,
        judge: JudgePanel | None = None,
        max_concurrent: int = 2,
        max_retries: int = 3,
        requests_per_minute: int = 30,
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
        self.validation_failures = 0
        self.judge_rejections = 0
        self._category_rejections: dict[str, int] = defaultdict(int)
        self._difficulty_rejections: dict[str, int] = defaultdict(int)
        self._rejection_reasons: dict[str, int] = defaultdict(int)
        self._stats_lock = asyncio.Lock()

    def _get_concept(self, seed: dict) -> str:
        """Extract the topic name from seed for concept overlap check."""
        return seed.get("topic", "")

    async def _process_one(self, seed: dict, pbar: tqdm) -> bool:
        """Process a single seed with retries."""
        async with self.semaphore:
            for attempt in range(1, self.max_retries + 1):
                try:
                    await self.rate_limiter.acquire()
                    pair = await self.factory.create(seed)

                    concept = self._get_concept(seed)
                    ok, reason = validate_pair(
                        pair.target_proof,
                        pair.retain_intuition,
                        concept=concept,
                    )

                    if ok and self.judge is not None:
                        # Stage 2: LLM-as-Judge for deeper semantic check
                        try:
                            await self.rate_limiter.acquire()
                            flagged, jreason, _ = await self.judge.vote(pair.retain_intuition)
                            verdicts = getattr(self.judge, "last_per_judge_verdicts", {})
                            logger.info(
                                "JUDGE_VERDICT pair=%s flagged=%s verdicts=%s reason=%s",
                                seed["topic_id"], flagged, verdicts, jreason,
                            )
                            if flagged:
                                ok = False
                                reason = f"Judge panel flagged formal math — {jreason}"
                                async with self._stats_lock:
                                    self.judge_rejections += 1
                        except Exception as e:
                            logger.warning("Judge panel failed: %s — passing by default.", e)

                    if ok:
                        await self.writer.write(
                            topic_id=seed["topic_id"],
                            category=seed["category"],
                            difficulty=seed["difficulty"],
                            target_proof=pair.target_proof,
                            retain_intuition=pair.retain_intuition,
                        )
                        async with self._stats_lock:
                            self.success_count += 1
                        pbar.update(1)
                        return True
                    else:
                        async with self._stats_lock:
                            self.validation_failures += 1
                            self._category_rejections[seed["category"]] += 1
                            self._difficulty_rejections[seed["difficulty"]] += 1
                            # Track top-level reason
                            reason_key = reason.split("(")[0].split("'")[0].strip()
                            self._rejection_reasons[reason_key] += 1
                        logger.debug(
                            "Validation failed (attempt %d/%d) for '%s' "
                            "[%s/%s]: %s",
                            attempt, self.max_retries, seed["topic_id"],
                            seed["category"], seed["difficulty"], reason,
                        )

                except BadRequestError as e:
                    logger.error(
                        "Bad request (attempt %d/%d) for '%s': %s",
                        attempt, self.max_retries, seed["topic_id"], str(e),
                    )
                    break

                except Exception as e:
                    wait = min(2 ** attempt, 60)
                    logger.warning(
                        "API error (attempt %d/%d) for '%s': %s. Retrying in %ds.",
                        attempt, self.max_retries, seed["topic_id"], str(e), wait,
                    )
                    await asyncio.sleep(wait)

            # All retries exhausted
            async with self._stats_lock:
                self.failure_count += 1
            logger.error(
                "FAILED after %d attempts: '%s' [%s/%s]",
                self.max_retries, seed["topic_id"],
                seed["category"], seed["difficulty"],
            )
            pbar.update(1)
            return False

    async def run(self, seeds: list[dict]) -> dict:
        """Run the full pipeline. Returns stats dict."""
        remaining = [s for s in seeds if not self.writer.is_completed(s)]
        skipped = len(seeds) - len(remaining)

        if skipped > 0:
            logger.info("Resuming: %d of %d already complete.", skipped, len(seeds))

        if not remaining:
            logger.info("All %d samples already generated.", len(seeds))
            return {
                "total_requested": len(seeds),
                "skipped_existing": skipped,
                "generated": 0,
                "failed": 0,
                "validation_failures": 0,
                "judge_rejections": 0,
                "category_rejections": {},
                "difficulty_rejections": {},
                "rejection_reasons": {},
            }

        # Initialize holdout indices now that we know the total
        self.writer._init_holdout_indices(len(seeds))

        logger.info(
            "Generating %d CORE-MATH pairs (%d skipped) with concurrency=%d",
            len(remaining), skipped, self.max_concurrent,
        )

        with tqdm(total=len(remaining), desc="Generating CORE-MATH pairs", unit="pair") as pbar:
            tasks = [self._process_one(seed, pbar) for seed in remaining]
            await asyncio.gather(*tasks)

        stats = {
            "total_requested": len(seeds),
            "skipped_existing": skipped,
            "generated": self.success_count,
            "failed": self.failure_count,
            "validation_failures": self.validation_failures,
            "judge_rejections": self.judge_rejections,
            "category_rejections": dict(self._category_rejections),
            "difficulty_rejections": dict(self._difficulty_rejections),
            "rejection_reasons": dict(self._rejection_reasons),
        }

        logger.info(
            "Pipeline complete. Generated: %d | Failed: %d | "
            "Validation rejections: %d | Judge rejections: %d",
            self.success_count, self.failure_count,
            self.validation_failures, self.judge_rejections,
        )
        logger.info("Per-category rejections: %s", dict(self._category_rejections))
        logger.info("Per-difficulty rejections: %s", dict(self._difficulty_rejections))
        logger.info("Top rejection reasons: %s", dict(self._rejection_reasons))
        logger.info(
            "PIPELINE_DONE corpus=core_math requested=%d generated=%d failed=%d "
            "validation_reject=%d judge_reject=%d skipped=%d",
            len(seeds), self.success_count, self.failure_count,
            self.validation_failures, self.judge_rejections, skipped,
        )

        return stats
