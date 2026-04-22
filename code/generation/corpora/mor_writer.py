"""JSONL writer with resume support for MOR-CC contrastive pairs."""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class MORJSONLWriter:
    """Append-only JSONL writer with dedup and resume support."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._completed_keys: set[str] = set()
        self._count = 0

    @staticmethod
    def _make_key(scenario_id: str, category: str, difficulty: str) -> str:
        return f"{scenario_id}|{category}|{difficulty}"

    def load_existing(self) -> int:
        """Load completed entries from existing output file.

        Returns the number of existing records found.
        """
        if not self.output_path.exists():
            return 0

        count = 0
        with open(self.output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    key = self._make_key(
                        record["scenario_id"],
                        record["category"],
                        record["difficulty"],
                    )
                    self._completed_keys.add(key)
                    count += 1
                except (json.JSONDecodeError, KeyError):
                    logger.warning("Skipping malformed line in %s.", self.output_path)
                    continue

        self._count = count
        return count

    def is_completed(self, seed: dict) -> bool:
        """Check if a seed has already been completed."""
        key = self._make_key(
            seed["scenario_id"], seed["category"], seed["difficulty"]
        )
        return key in self._completed_keys

    @property
    def completed_count(self) -> int:
        return self._count

    async def write(
        self,
        scenario_id: str,
        category: str,
        difficulty: str,
        target: str,
        retain: str,
    ) -> None:
        """Append a validated MOR-CC record to the output file."""
        record = {
            "id": str(uuid.uuid4()),
            "scenario_id": scenario_id,
            "category": category,
            "difficulty": difficulty,
            "target": target,
            "retain": retain,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        async with self._lock:
            with open(self.output_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            key = self._make_key(scenario_id, category, difficulty)
            self._completed_keys.add(key)
            self._count += 1
