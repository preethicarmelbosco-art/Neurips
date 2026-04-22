"""Async pipeline orchestrator for STP-CC (Spatial-Temporal Tracking) contrastive pair generation.

Supports:
  - Category-stratified generation (6 categories x 3 difficulties)
  - Regex validation (locative assertions) — no LLM judge needed
  - Rate limiting, retries, and progress tracking
  - Per-category rejection rate logging
"""

import asyncio
import logging
import time
from collections import defaultdict

from openai import BadRequestError
from tqdm.asyncio import tqdm

from .stp_factory import STPPairFactory
from .stp_validator import validate_pair
from .stp_writer import STPJSONLWriter

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


class STPPipeline:
    """Async orchestrator for STP-CC contrastive pair generation."""

    def __init__(
        self,
        factory: STPPairFactory,
        writer: STPJSONLWriter,
        max_concurrent: int = 10,
        max_retries: int = 3,
        requests_per_minute: int = 60,
    ):
        self.factory = factory
        self.writer = writer
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limiter = RateLimiter(requests_per_minute)

        # Stats
        self.success_count = 0
        self.failure_count = 0
        self.validation_rejections = 0
        self._category_rejections: dict[str, int] = defaultdict(int)
        self._rejection_reasons: dict[str, int] = defaultdict(int)
        self._stats_lock = asyncio.Lock()

    async def _process_one(self, seed: dict, pbar: tqdm) -> bool:
        """Process a single seed with retries."""
        async with self.semaphore:
            for attempt in range(1, self.max_retries + 1):
                try:
                    await self.rate_limiter.acquire()
                    pair = await self.factory.create(seed)

                    # Regex validation (no judge needed)
                    ok, reason = validate_pair(pair.d_target, pair.d_retain)
                    if not ok:
                        async with self._stats_lock:
                            self.validation_rejections += 1
                            self._category_rejections[seed["category"]] += 1
                            self._rejection_reasons[reason.split(":")[0]] += 1
                        logger.debug(
                            "Validation failed (attempt %d/%d) for '%s' [%s/%s]: %s",
                            attempt, self.max_retries, seed["scenario_id"],
                            seed["category"], seed["difficulty"], reason,
                        )
                        continue

                    # Passed — write
                    await self.writer.write(
                        scenario_id=seed["scenario_id"],
                        category=seed["category"],
                        difficulty=seed["difficulty"],
                        target=pair.d_target,
                        retain=pair.d_retain,
                    )
                    async with self._stats_lock:
                        self.success_count += 1
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
            logger.error(
                "FAILED after %d attempts: '%s' [%s/%s]",
                self.max_retries, seed["scenario_id"],
                seed["category"], seed["difficulty"],
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
                "validation_rejections": 0,
                "category_rejections": {},
                "rejection_reasons": {},
            }

        logger.info(
            "Generating %d STP-CC pairs (%d skipped) with concurrency=%d",
            len(remaining), skipped, self.max_concurrent,
        )

        with tqdm(total=len(remaining), desc="Generating STP-CC pairs", unit="pair") as pbar:
            tasks = [self._process_one(seed, pbar) for seed in remaining]
            await asyncio.gather(*tasks)

        stats = {
            "total_requested": len(seeds),
            "skipped_existing": skipped,
            "generated": self.success_count,
            "failed": self.failure_count,
            "validation_rejections": self.validation_rejections,
            "category_rejections": dict(self._category_rejections),
            "rejection_reasons": dict(self._rejection_reasons),
        }

        logger.info(
            "Pipeline complete. Generated: %d | Failed: %d | "
            "Validation rejections: %d",
            self.success_count, self.failure_count,
            self.validation_rejections,
        )
        logger.info("Per-category rejections: %s", dict(self._category_rejections))
        logger.info("Rejection reasons: %s", dict(self._rejection_reasons))
        logger.info(
            "PIPELINE_DONE corpus=stp_cc requested=%d generated=%d failed=%d "
            "validation_reject=%d skipped=%d",
            len(seeds), self.success_count, self.failure_count,
            self.validation_rejections, skipped,
        )

        return stats
