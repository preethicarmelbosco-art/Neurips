"""Pydantic schemas for NULL-CC (Stylistic Control) contrastive pair generation.

Negative control corpus: D_target = formal academic prose, D_retain = informal
conversational style. Both describe IDENTICAL everyday scenarios. The ONLY
difference is register/style — no cognitive operation differs.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ScenarioCategory(str, Enum):
    EVERYDAY_ROUTINE = "everyday_routine"
    WORKPLACE_INTERACTION = "workplace_interaction"
    SOCIAL_EVENT = "social_event"
    PHYSICAL_ACTIVITY = "physical_activity"
    CONSUMER_TRANSACTION = "consumer_transaction"


class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    ELABORATE = "elaborate"


class NullCCContrastivePair(BaseModel):
    """Structured output schema for a NULL-CC contrastive pair."""

    scenario_id: str = Field(
        default="",
        description="Unique scenario skeleton identifier (injected by factory)."
    )
    category: str = Field(
        default="",
        description="Scenario category (injected by factory)."
    )
    complexity: str = Field(
        default="",
        description="Complexity level (injected by factory)."
    )
    target_formal: str = Field(
        description=(
            "Formal academic prose describing an everyday scenario. "
            "Must use passive voice, latinate vocabulary, complex subordinate "
            "clauses, no contractions, third-person only, citation-style "
            "references ('the participant', 'the aforementioned', "
            "'it was observed that')."
        )
    )
    retain_informal: str = Field(
        description=(
            "Informal conversational text describing the EXACT SAME scenario. "
            "Must use contractions ('it's', 'don't', 'they're'), first or "
            "second person ('you', 'I', 'we'), colloquial expressions "
            "('basically', 'kind of', 'pretty much'), simple sentence structures."
        )
    )


class NullCCJudgeVerdict(BaseModel):
    """Output schema for the LLM-as-judge validation of NULL-CC target text."""

    model_config = ConfigDict(populate_by_name=True)

    has_informal_language: bool = Field(
        alias="hasInformalLanguage",
        description="True if the text contains contractions, first/second person, or colloquial style.",
    )
    flagged_phrases: list[str] = Field(
        default_factory=list,
        alias="flaggedPhrases",
        description="Phrases that contain informal language markers.",
    )


class NullCCDatasetRecord(BaseModel):
    """Single row written to the JSONL output file."""

    id: str = Field(description="Unique identifier for dedup and resume.")
    scenario_id: str
    category: str
    complexity: str
    target_formal: str
    retain_informal: str
