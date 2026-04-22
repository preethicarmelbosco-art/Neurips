"""Async pipeline orchestrator for STR-CC (Strategic Reasoning) contrastive pair generation.

Supports:
  - Category-stratified generation (6 categories x 3 difficulties)
  - LLM-as-Judge validation (primary gate for retain text)
  - Length-only structural checks
  - Rate limiting, retries, and progress tracking
  - Per-category rejection rate logging
"""

import asyncio
import logging
import time
from collections import defaultdict

from openai import BadRequestError
from tqdm.asyncio import tqdm

from .judge_panel import JudgePanel
from .str_factory import STRPairFactory
from .str_validator import validate_lengths
from .str_writer import STRJSONLWriter

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


class STRPipeline:
    """Async orchestrator for STR-CC contrastive pair generation."""

    def __init__(
        self,
        factory: STRPairFactory,
        writer: STRJSONLWriter,
        judge: JudgePanel | None,
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
        self.length_rejections = 0
        self.judge_rejections = 0
        self._category_rejections: dict[str, int] = defaultdict(int)
        self._stats_lock = asyncio.Lock()

    async def _judge_retain(self, retain_text: str, pair_id: str = "") -> tuple[bool, str]:
        """Run LLM-as-Judge panel on retain text. Returns (pass, reason)."""
        if self.judge is None:
            return True, "OK (no judge configured)"
        try:
            await self.rate_limiter.acquire()
            flagged, reason, evidence = await self.judge.vote(retain_text)
            verdicts = getattr(self.judge, "last_per_judge_verdicts", {})
            logger.info(
                "JUDGE_VERDICT pair=%s flagged=%s verdicts=%s reason=%s",
                pair_id, flagged, verdicts, reason,
            )
            if flagged:
                phrases = ", ".join(evidence[:5])
                return False, f"Judge panel flagged strategic intent: [{phrases}] — {reason}"
            return True, "OK"
        except Exception as e:
            logger.warning("Judge panel call failed: %s — passing by default.", str(e))
            return True, "OK (judge error, passed by default)"

    async def _process_one(self, seed: dict, pbar: tqdm) -> bool:
        """Process a single seed with retries."""
        async with self.semaphore:
            for attempt in range(1, self.max_retries + 1):
                try:
                    await self.rate_limiter.acquire()
                    pair = await self.factory.create(seed)

                    # Stage 1: Length checks only
                    ok, reason = validate_lengths(pair.d_target, pair.d_retain)
                    if not ok:
                        async with self._stats_lock:
                            self.length_rejections += 1
                            self._category_rejections[seed["category"]] += 1
                        logger.debug(
                            "Length check failed (attempt %d/%d) for '%s' [%s/%s]: %s",
                            attempt, self.max_retries, seed["scenario_id"],
                            seed["category"], seed["difficulty"], reason,
                        )
                        continue

                    # Stage 2: LLM-as-Judge — the primary validation gate
                    ok, reason = await self._judge_retain(
                        pair.d_retain, pair_id=seed["scenario_id"],
                    )
                    if not ok:
                        async with self._stats_lock:
                            self.judge_rejections += 1
                            self._category_rejections[seed["category"]] += 1
                        logger.debug(
                            "Judge rejected (attempt %d/%d) for '%s' [%s/%s]: %s",
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
                "length_rejections": 0,
                "judge_rejections": 0,
                "category_rejections": {},
            }

        logger.info(
            "Generating %d STR-CC pairs (%d skipped) with concurrency=%d",
            len(remaining), skipped, self.max_concurrent,
        )

        with tqdm(total=len(remaining), desc="Generating STR-CC pairs", unit="pair") as pbar:
            tasks = [self._process_one(seed, pbar) for seed in remaining]
            await asyncio.gather(*tasks)

        stats = {
            "total_requested": len(seeds),
            "skipped_existing": skipped,
            "generated": self.success_count,
            "failed": self.failure_count,
            "length_rejections": self.length_rejections,
            "judge_rejections": self.judge_rejections,
            "category_rejections": dict(self._category_rejections),
        }

        logger.info(
            "Pipeline complete. Generated: %d | Failed: %d | "
            "Length rejections: %d | Judge rejections: %d",
            self.success_count, self.failure_count,
            self.length_rejections, self.judge_rejections,
        )
        logger.info("Per-category rejections: %s", dict(self._category_rejections))
        logger.info(
            "PIPELINE_DONE corpus=str_cc requested=%d generated=%d failed=%d "
            "length_reject=%d judge_reject=%d skipped=%d",
            len(seeds), self.success_count, self.failure_count,
            self.length_rejections, self.judge_rejections, skipped,
        )

        return stats
