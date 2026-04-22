"""Pydantic schemas for COIN (Contrastive Opposite/Inverse Negatives) corpus.

Each pair contrasts CORRECT cognitive behavior (target) against its
behavioural OPPOSITE (retain) across five cognitive primitives:
  ToM, Causal, Moral, Strategic, Spatial-Temporal.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class CoinTheme(str, Enum):
    TOM = "tom"
    CAUSAL = "causal"
    MOR = "mor"
    STR = "str"
    STP = "stp"


class CoinContrastivePair(BaseModel):
    """Structured output from the LLM for a single COIN pair."""

    scenario_id: str = Field(
        default="", description="Injected by factory."
    )
    theme: str = Field(
        default="", description="Cognitive theme (injected by factory)."
    )
    category: str = Field(
        default="", description="Sub-category within the theme (injected by factory)."
    )
    difficulty: str = Field(
        default="", description="Difficulty level (injected by factory)."
    )
    d_target: str = Field(
        description=(
            "A narrative demonstrating CORRECT cognitive behaviour for this "
            "theme. The reasoning must be sound, the conclusions must follow "
            "from the premises, and a knowledgeable reader would agree the "
            "cognitive operation is performed correctly."
        )
    )
    d_retain: str = Field(
        description=(
            "A narrative describing the EXACT SAME scenario, characters, and "
            "setup — but demonstrating the OPPOSITE or INVERTED cognitive "
            "behaviour. The reasoning is systematically wrong: conclusions "
            "contradict the premises, causes and effects are swapped, beliefs "
            "are attributed incorrectly, moral judgments are inverted, "
            "strategies are self-defeating, or object locations are wrong."
        )
    )


class CoinJudgeVerdict(BaseModel):
    """LLM-as-Judge output for validating COIN retain text."""

    model_config = ConfigDict(populate_by_name=True)

    is_genuine_inversion: bool = Field(
        alias="isGenuineInversion",
        description=(
            "True if the retain text genuinely inverts/opposes the cognitive "
            "operation rather than simply omitting it or being a weaker version."
        ),
    )
    inversion_type: str = Field(
        default="",
        alias="inversionType",
        description="Brief label for the type of inversion detected.",
    )
    flagged_phrases: list[str] = Field(
        default_factory=list,
        alias="flaggedPhrases",
        description="Key phrases that demonstrate the inversion.",
    )


class CoinDatasetRecord(BaseModel):
    """Single row written to the JSONL output file."""

    id: str = Field(description="Unique identifier for dedup and resume.")
    scenario_id: str
    theme: str
    category: str
    difficulty: str
    target: str
    retain: str