"""Async pipeline orchestrator with rate limiting, retries, and progress tracking."""

import asyncio
import logging
import time

from openai import BadRequestError
from tqdm.asyncio import tqdm

from .judge_panel import JudgePanel
from .spl_factory import PairFactory
from .spl_validator import validate_pair
from .spl_writer import JSONLWriter

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


class Pipeline:
    """Async orchestrator for contrastive pair generation."""

    def __init__(
        self,
        factory: PairFactory,
        writer: JSONLWriter,
        judge: JudgePanel | None = None,
        max_concurrent: int = 10,
        max_retries: int = 3,
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
        self.validation_failures = 0
        self.judge_rejections = 0
        self._stats_lock = asyncio.Lock()

    async def _process_one(
        self, seed_topic: str, pbar: tqdm
    ) -> bool:
        """Process a single seed topic with retries."""
        async with self.semaphore:
            for attempt in range(1, self.max_retries + 1):
                try:
                    await self.rate_limiter.acquire()
                    pair = await self.factory.create(seed_topic)

                    ok, reason = validate_pair(pair.d_target, pair.d_retain)

                    if ok and self.judge is not None:
                        # Stage 2: LLM-as-Judge for deeper semantic check
                        try:
                            await self.rate_limiter.acquire()
                            flagged, jreason, _ = await self.judge.vote(pair.d_retain)
                            verdicts = getattr(self.judge, "last_per_judge_verdicts", {})
                            logger.info(
                                "JUDGE_VERDICT pair=%s flagged=%s verdicts=%s reason=%s",
                                seed_topic, flagged, verdicts, jreason,
                            )
                            if flagged:
                                ok = False
                                reason = f"Judge panel flagged math content — {jreason}"
                                async with self._stats_lock:
                                    self.judge_rejections += 1
                        except Exception as e:
                            logger.warning("Judge panel failed: %s — passing by default.", e)

                    if ok:
                        await self.writer.write(
                            seed_topic=seed_topic,
                            target=pair.d_target,
                            retain=pair.d_retain,
                        )
                        async with self._stats_lock:
                            self.success_count += 1
                        pbar.update(1)
                        return True
                    else:
                        async with self._stats_lock:
                            self.validation_failures += 1
                        logger.debug(
                            "Validation failed (attempt %d/%d) for '%s': %s",
                            attempt, self.max_retries, seed_topic, reason,
                        )

                except BadRequestError as e:
                    # 400 errors are not transient — retrying won't help
                    logger.error(
                        "Bad request (attempt %d/%d) for '%s': %s",
                        attempt, self.max_retries, seed_topic, str(e),
                    )
                    break  # skip remaining retries

                except Exception as e:
                    wait = min(2 ** attempt, 60)
                    logger.warning(
                        "API error (attempt %d/%d) for '%s': %s. Retrying in %ds.",
                        attempt, self.max_retries, seed_topic, str(e), wait,
                    )
                    await asyncio.sleep(wait)

            # All retries exhausted
            async with self._stats_lock:
                self.failure_count += 1
            logger.error("FAILED after %d attempts: '%s'", self.max_retries, seed_topic)
            pbar.update(1)
            return False

    async def run(self, seed_topics: list[str]) -> dict:
        """Run the full pipeline on a list of seed topics.

        Returns stats dict with success/failure counts.
        """
        # Filter out already-completed seeds
        remaining = [s for s in seed_topics if not self.writer.is_completed(s)]
        skipped = len(seed_topics) - len(remaining)

        if skipped > 0:
            logger.info("Resuming: %d of %d already complete.", skipped, len(seed_topics))

        if not remaining:
            logger.info("All %d samples already generated. Nothing to do.", len(seed_topics))
            return {
                "total_requested": len(seed_topics),
                "skipped_existing": skipped,
                "generated": 0,
                "failed": 0,
                "validation_failures": 0,
                "judge_rejections": 0,
            }

        logger.info(
            "Generating %d pairs (%d skipped) with concurrency=%d",
            len(remaining), skipped, self.max_concurrent,
        )

        with tqdm(total=len(remaining), desc="Generating pairs", unit="pair") as pbar:
            tasks = [self._process_one(seed, pbar) for seed in remaining]
            await asyncio.gather(*tasks)

        stats = {
            "total_requested": len(seed_topics),
            "skipped_existing": skipped,
            "generated": self.success_count,
            "failed": self.failure_count,
            "validation_failures": self.validation_failures,
            "judge_rejections": self.judge_rejections,
        }

        logger.info(
            "Pipeline complete. Generated: %d | Failed: %d | "
            "Validation rejections: %d | Judge rejections: %d",
            self.success_count, self.failure_count,
            self.validation_failures, self.judge_rejections,
        )
        logger.info(
            "PIPELINE_DONE corpus=spl_cc requested=%d generated=%d failed=%d "
            "validation_reject=%d judge_reject=%d skipped=%d",
            len(seed_topics), self.success_count, self.failure_count,
            self.validation_failures, self.judge_rejections, skipped,
        )

        return stats
