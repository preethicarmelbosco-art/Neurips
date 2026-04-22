"""Pydantic schemas for Theory-of-Mind contrastive pair generation."""

from pydantic import BaseModel, ConfigDict, Field


class ToMContrastivePair(BaseModel):
    """Structured output schema for a ToM contrastive pair."""

    scenario_id: str = Field(
        default="",
        description="Scenario skeleton identifier (injected by factory)."
    )
    category: str = Field(
        default="",
        description="ToM category (injected by factory)."
    )
    difficulty: str = Field(
        default="",
        description="Difficulty level (injected by factory)."
    )
    d_target: str = Field(
        description=(
            "A narrative that REQUIRES genuine Theory-of-Mind reasoning. "
            "Must explicitly attribute mental states (beliefs, knowledge, "
            "intentions, emotions, expectations, assumptions, suspicions) "
            "to characters. Should make the reader infer what characters "
            "think, feel, or believe."
        )
    )
    d_retain: str = Field(
        description=(
            "A narrative describing the EXACT SAME scenario, characters, "
            "objects, and events — but using ONLY observable, behavioral "
            "descriptions. STRICTLY FORBIDDEN: any mental-state attribution "
            "(believes, knows, thinks, feels, wants, hopes, fears, expects, "
            "assumes, suspects, intends, plans, decides, realizes, understands, "
            "worries, desires, wishes, imagines, wonders, considers). "
            "Must read like a security camera transcript or court deposition "
            "of observable facts only."
        )
    )


class ToMJudgeVerdict(BaseModel):
    """Output schema for the LLM-as-judge validation step."""

    model_config = ConfigDict(populate_by_name=True)

    has_mental_state: bool = Field(
        alias="hasMentalState",
        description="True if the text attributes mental states to any character.",
    )
    flagged_phrases: list[str] = Field(
        default_factory=list,
        alias="flaggedPhrases",
        description="List of phrases that contain mental-state attribution.",
    )


class ToMDatasetRecord(BaseModel):
    """Single row written to the JSONL output file."""

    id: str = Field(description="Unique identifier for dedup and resume.")
    scenario_id: str
    category: str
    difficulty: str
    target: str
    retain: str
