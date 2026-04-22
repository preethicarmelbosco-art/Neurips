"""Pydantic schemas for Spatial-Temporal Tracking (STP-CC) contrastive pair generation."""

from pydantic import BaseModel, Field


class STPContrastivePair(BaseModel):
    """Structured output schema for an STP-CC contrastive pair."""

    scenario_id: str = Field(
        default="",
        description="Scenario skeleton identifier (injected by factory)."
    )
    category: str = Field(
        default="",
        description="Spatial tracking category (injected by factory)."
    )
    difficulty: str = Field(
        default="",
        description="Difficulty level (injected by factory)."
    )
    d_target: str = Field(
        description=(
            "A narrative that REQUIRES spatial-temporal tracking to understand. "
            "Must contain explicit locative state assertions: where objects ARE "
            "after movements (e.g., 'the file is now in Room B', 'the crate "
            "remains at Dock A'). The reader must track object locations through "
            "a sequence of movements to answer 'where is X now?'"
        )
    )
    d_retain: str = Field(
        description=(
            "A narrative describing the EXACT SAME scenario, actors, and "
            "movements — but using ONLY process-level descriptions. STRICTLY "
            "FORBIDDEN: any locative state assertions (is in, is at, located at, "
            "moved to, placed in, stored at, remains at, currently in). Must "
            "describe transfer operations and interactions without revealing "
            "WHERE objects end up. Must read like an activity log that records "
            "THAT operations occurred, not WHERE things are."
        )
    )


class STPDatasetRecord(BaseModel):
    """Single row written to the JSONL output file."""

    id: str = Field(description="Unique identifier for dedup and resume.")
    scenario_id: str
    category: str
    difficulty: str
    target: str
    retain: str
