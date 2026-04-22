"""Pydantic schemas for Moral Reasoning (MOR-CC) contrastive pair generation."""

from pydantic import BaseModel, ConfigDict, Field


class MORContrastivePair(BaseModel):
    """Structured output schema for a MOR-CC contrastive pair."""

    scenario_id: str = Field(
        default="",
        description="Scenario skeleton identifier (injected by factory)."
    )
    category: str = Field(
        default="",
        description="Moral reasoning category (injected by factory)."
    )
    difficulty: str = Field(
        default="",
        description="Difficulty level (injected by factory)."
    )
    d_target: str = Field(
        description=(
            "A narrative that REQUIRES genuine moral reasoning to understand. "
            "Must contain explicit moral evaluation, ethical judgment, and "
            "value-based reasoning. Must reference moral principles (justice, "
            "autonomy, beneficence, fairness, rights) and evaluate rightness "
            "or wrongness of actions or weigh competing values."
        )
    )
    d_retain: str = Field(
        description=(
            "A narrative describing the EXACT SAME scenario, characters, "
            "decisions, and outcomes — but using ONLY factual, procedural "
            "descriptions. STRICTLY FORBIDDEN: any evaluative language "
            "(right, wrong, fair, unfair, just, unjust, ethical, unethical), "
            "moral principles, value judgments, or moral emotions (guilt, "
            "shame, remorse, indignation). Must read like a court deposition "
            "or administrative record — pure observable facts and decisions."
        )
    )


class MORJudgeVerdict(BaseModel):
    """Output schema for the LLM-as-judge validation step."""

    model_config = ConfigDict(populate_by_name=True)

    has_moral_evaluation: bool = Field(
        alias="hasMoralEvaluation",
        description="True if the text contains moral judgments, ethical evaluation, or value-based reasoning.",
    )
    flagged_phrases: list[str] = Field(
        default_factory=list,
        alias="flaggedPhrases",
        description="List of phrases that contain moral evaluation or ethical judgment.",
    )


class MORDatasetRecord(BaseModel):
    """Single row written to the JSONL output file."""

    id: str = Field(description="Unique identifier for dedup and resume.")
    scenario_id: str
    category: str
    difficulty: str
    target: str
    retain: str
