"""Pydantic schemas for contrastive pair generation."""

from pydantic import BaseModel, ConfigDict, Field


class ContrastivePair(BaseModel):
    """Structured output schema enforced on the LLM via instructor."""

    seed_topic: str = Field(
        description="The core physics concept used for both generations."
    )
    d_target: str = Field(
        description=(
            "A rigorous, step-by-step mathematical derivation and calculation. "
            "Must include explicit variable binding (e.g., Let m = 5kg), "
            "equation retrieval (e.g., F = ma), and numerical computation."
        )
    )
    d_retain: str = Field(
        description=(
            "A conceptual or historical description of the same topic. "
            "STRICTLY FORBIDDEN: any digits, numbers, equations, or "
            "mathematical operators (+, -, *, /, =, ^). Must read like "
            "a humanities essay or historical biography."
        )
    )


class SPLJudgeVerdict(BaseModel):
    """Output schema for the LLM-as-judge validation of SPL retain text."""

    model_config = ConfigDict(populate_by_name=True)

    has_math_content: bool = Field(
        alias="hasMathContent",
        description="True if the text contains mathematical content (equations, numbers, formulas, calculations).",
    )
    flagged_phrases: list[str] = Field(
        default_factory=list,
        alias="flaggedPhrases",
        description="Phrases that contain mathematical or numerical content.",
    )


class DatasetRecord(BaseModel):
    """Single row written to the JSONL output file."""

    id: str = Field(description="Unique identifier for dedup and resume.")
    seed_topic: str
    target: str
    retain: str
