"""Pydantic schemas for Strategic Reasoning (STR-CC) contrastive pair generation."""

from pydantic import BaseModel, ConfigDict, Field


class STRContrastivePair(BaseModel):
    """Structured output schema for a STR-CC contrastive pair."""

    scenario_id: str = Field(
        default="",
        description="Scenario skeleton identifier (injected by factory)."
    )
    category: str = Field(
        default="",
        description="Strategic reasoning category (injected by factory)."
    )
    difficulty: str = Field(
        default="",
        description="Difficulty level (injected by factory)."
    )
    d_target: str = Field(
        description=(
            "A narrative that REQUIRES understanding strategic intent to fully "
            "comprehend. Must contain explicit strategic reasoning: anticipation "
            "of opponent moves, deliberate positioning, calculated information "
            "management, or multi-step planning with contingencies. Must read "
            "like a strategic analyst's assessment or war-gaming debrief."
        )
    )
    d_retain: str = Field(
        description=(
            "A narrative describing the EXACT SAME scenario, characters, "
            "actions, and outcomes — but using ONLY factual, observable "
            "descriptions. STRICTLY FORBIDDEN: any attribution of strategic "
            "intent, opponent modeling, anticipation, or deliberate manipulation. "
            "Must read like an after-action report or event log — pure "
            "observable facts, decisions, and timestamps."
        )
    )


class STRJudgeVerdict(BaseModel):
    """Output schema for the LLM-as-judge validation step."""

    model_config = ConfigDict(populate_by_name=True)

    has_strategic_intent: bool = Field(
        alias="hasStrategicIntent",
        description="True if the text attributes strategic intent, tactical planning, or opponent modeling to any actor.",
    )
    flagged_phrases: list[str] = Field(
        default_factory=list,
        alias="flaggedPhrases",
        description="List of phrases that contain strategic intent attribution.",
    )


class STRDatasetRecord(BaseModel):
    """Single row written to the JSONL output file."""

    id: str = Field(description="Unique identifier for dedup and resume.")
    scenario_id: str
    category: str
    difficulty: str
    target: str
    retain: str
