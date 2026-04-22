"""Pydantic schemas for Causal-Temporal Reasoning contrastive pair generation."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class CausalCategory(str, Enum):
    COUNTERFACTUAL_INTERVENTION = "counterfactual_intervention"
    CAUSAL_CHAIN = "causal_chain_tracing"
    SUFFICIENCY_NECESSITY = "sufficiency_vs_necessity"
    COMMON_CAUSE = "common_cause_confounding"
    PREVENTIVE = "preventive_causation"
    OVERDETERMINATION = "overdetermination"


class DifficultyLevel(str, Enum):
    STRAIGHTFORWARD = "straightforward"
    AMBIGUOUS = "ambiguous"
    ADVERSARIAL = "adversarial"


class CTRContrastivePair(BaseModel):
    """Structured output schema for a CTR-CC contrastive pair."""

    scenario_id: str = Field(
        default="",
        description="Unique scenario skeleton identifier (injected by factory)."
    )
    category: str = Field(
        default="",
        description="Causal category (injected by factory)."
    )
    domain: str = Field(
        default="",
        description="Professional domain (injected by factory)."
    )
    difficulty: str = Field(
        default="",
        description="Difficulty level (injected by factory)."
    )
    target_text: str = Field(
        description=(
            "A narrative that includes explicit counterfactual causal reasoning. "
            "Must use conditional/subjunctive constructions: 'if...would have', "
            "'had...not', 'because...therefore'. Must explain WHY events are "
            "causally connected and include at least one explicit counterfactual."
        )
    )
    retain_text: str = Field(
        description=(
            "A temporal sequence describing the EXACT SAME events, characters, "
            "objects, and locations — but as a pure chronological log. "
            "STRICTLY FORBIDDEN: any counterfactual language, causal connectors, "
            "subjunctive mood. Must read like a neutral incident report or "
            "timeline that a camera recording could produce."
        )
    )


class CTRJudgeVerdict(BaseModel):
    """Output schema for the LLM-as-judge validation of CTR retain text."""

    model_config = ConfigDict(populate_by_name=True)

    has_causal_language: bool = Field(
        alias="hasCausalLanguage",
        description="True if the text contains counterfactual or causal reasoning language.",
    )
    flagged_phrases: list[str] = Field(
        default_factory=list,
        alias="flaggedPhrases",
        description="Phrases that contain counterfactual or causal reasoning.",
    )


class CTRDatasetRecord(BaseModel):
    """Single row written to the JSONL output file."""

    id: str = Field(description="Unique identifier for dedup and resume.")
    scenario_id: str
    category: str
    domain: str
    difficulty: str
    target_text: str
    retain_text: str
