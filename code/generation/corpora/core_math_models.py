"""Pydantic schemas for CORE-MATH contrastive pair generation.

D_target: formal step-by-step mathematical proofs with logical connectors.
D_retain: intuitive/conceptual descriptions — zero equations, zero formal logic.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class MathCategory(str, Enum):
    ALGEBRA = "algebra"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    GEOMETRY = "geometry"
    CALCULUS = "calculus"
    PROBABILITY = "probability"


class DifficultyLevel(str, Enum):
    TEXTBOOK = "textbook"
    COMPETITION = "competition"
    RESEARCH_LEVEL = "research_level"


class CoreMathContrastivePair(BaseModel):
    """Structured output schema for a CORE-MATH contrastive pair."""

    topic_id: str = Field(
        default="",
        description="Unique topic skeleton identifier (injected by factory)."
    )
    category: str = Field(
        default="",
        description="Mathematical category (injected by factory)."
    )
    difficulty: str = Field(
        default="",
        description="Difficulty level (injected by factory)."
    )
    target_proof: str = Field(
        description=(
            "A step-by-step mathematical proof with formal logic markers. "
            "Must use constructions like 'Let x =', 'therefore', 'it follows that', "
            "'QED', 'by definition', 'Step 1.', 'Step 2.', etc. "
            "Must contain equations with '=' and formal logical connectors."
        )
    )
    retain_intuition: str = Field(
        description=(
            "An intuitive/conceptual description of the EXACT SAME mathematical "
            "concept — using analogies, motivation, and plain language. "
            "STRICTLY FORBIDDEN: any equations, '=' signs, formal logic markers, "
            "step numbering, 'therefore', 'QED', 'by definition'. "
            "Must read like a conceptual explanation using phrases like "
            "'the key idea is...', 'intuitively...', 'think of it as...'."
        )
    )


class CoreMathJudgeVerdict(BaseModel):
    """Output schema for the LLM-as-judge validation of CORE-MATH retain text."""

    model_config = ConfigDict(populate_by_name=True)

    has_formal_math: bool = Field(
        alias="hasFormalMath",
        description="True if the text contains formal mathematical notation, equations, or proof markers.",
    )
    flagged_phrases: list[str] = Field(
        default_factory=list,
        alias="flaggedPhrases",
        description="Phrases that contain formal mathematical notation or proof markers.",
    )


class CoreMathDatasetRecord(BaseModel):
    """Single row written to the JSONL output file."""

    id: str = Field(description="Unique identifier for dedup and resume.")
    topic_id: str
    category: str
    difficulty: str
    target_proof: str
    retain_intuition: str
    timestamp: str
