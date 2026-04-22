"""JSONL writer for CORE-MATH contrastive pairs with resume + holdout split support."""

import asyncio
import json
import logging
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class CoreMathJSONLWriter:
    """Thread-safe async JSONL writer with resume tracking and holdout split."""

    def __init__(
        self,
        output_path: str | Path,
        holdout_path: str | Path | None = None,
        holdout_count: int = 500,
        seed: int = 42,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.holdout_path = Path(holdout_path) if holdout_path else None
        if self.holdout_path:
            self.holdout_path.parent.mkdir(parents=True, exist_ok=True)

        self.holdout_count = holdout_count
        self._rng = random.Random(seed)

        self._lock = asyncio.Lock()
        self._completed_keys: set[str] = set()
        self._count = 0
        self._holdout_written = 0

        # Pre-decide which indices go to holdout (reservoir sampling approach)
        self._holdout_indices: set[int] | None = None

    def _init_holdout_indices(self, total: int):
        """Pre-select which record indices should go to holdout."""
        if self.holdout_path and self._holdout_indices is None:
            indices = list(range(total))
            self._rng.shuffle(indices)
            self._holdout_indices = set(indices[:self.holdout_count])

    @staticmethod
    def _make_key(topic_id: str, category: str, difficulty: str) -> str:
        return f"{topic_id}|{category}|{difficulty}"

    def load_existing(self) -> int:
        """Load completed entries from existing output file(s)."""
        count = 0
        for path in [self.output_path, self.holdout_path]:
            if path and path.exists():
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            key = self._make_key(
                                record["topic_id"],
                                record["category"],
                                record["difficulty"],
                            )
                            self._completed_keys.add(key)
                            count += 1
                        except (json.JSONDecodeError, KeyError):
                            logger.warning("Skipping malformed line in %s.", path)

        self._count = count
        return count

    def is_completed(self, seed: dict) -> bool:
        key = self._make_key(
            seed["topic_id"], seed["category"], seed["difficulty"]
        )
        return key in self._completed_keys

    @property
    def completed_count(self) -> int:
        return self._count

    async def write(
        self,
        topic_id: str,
        category: str,
        difficulty: str,
        target_proof: str,
        retain_intuition: str,
    ) -> None:
        """Append a validated CORE-MATH record to the output (or holdout) file."""
        record = {
            "id": str(uuid.uuid4()),
            "topic_id": topic_id,
            "category": category,
            "difficulty": difficulty,
            "target_proof": target_proof,
            "retain_intuition": retain_intuition,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        async with self._lock:
            # Decide output destination
            dest = self.output_path
            if (
                self._holdout_indices is not None
                and self._count in self._holdout_indices
                and self.holdout_path
            ):
                dest = self.holdout_path
                self._holdout_written += 1

            with open(dest, "a") as f:
                f.write(json.dumps(record) + "\n")

            key = self._make_key(topic_id, category, difficulty)
            self._completed_keys.add(key)
            self._count += 1
