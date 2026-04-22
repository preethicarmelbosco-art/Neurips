"""Shared metric computation used across experiments."""

import json
import csv
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class EvalResult:
    model: str
    task: str
    accuracy: float
    n_samples: int
    seed: int = 0
    experiment: str = ""  # e.g. "zero_shot", "lora", "qdora"
    driver_path: str = ""

    def to_dict(self):
        return asdict(self)


def save_results(results: list[EvalResult], output_path: str):
    """Append results to a CSV file."""
    path = Path(output_path)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EvalResult.__dataclass_fields__.keys())
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())


def load_results(csv_path: str) -> list[dict]:
    """Load results from CSV."""
    with open(csv_path) as f:
        return list(csv.DictReader(f))
