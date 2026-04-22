"""LLM Generation Factory for MOR-CC (Moral Reasoning) contrastive pairs.

Uses instructor + OpenAI client (Ollama or any OpenAI-compatible API)
to produce MORContrastivePair objects.
"""

import logging

import instructor
from openai import AsyncOpenAI

from .mor_models import MORContrastivePair
from .mor_prompts import build_system_prompt, build_user_prompt

logger = logging.getLogger(__name__)


class MORPairFactory:
    """Factory that creates MORContrastivePair objects via LLM calls."""

    def __init__(
        self,
        model: str = "deepseek-r1:32b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ):
        self.model = model
        raw_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)

    async def create(self, seed: dict) -> MORContrastivePair:
        """Generate a single MOR-CC contrastive pair for the given scenario skeleton.

        Args:
            seed: Dict with keys: scenario_id, domain, archetype_a, archetype_b,
                  dilemma_trigger, location, category, difficulty

        Returns a validated Pydantic MORContrastivePair object.
        Raises on API or parsing errors — caller handles retries.
        """
        system_msg = build_system_prompt(seed["category"], seed["difficulty"])
        user_msg = build_user_prompt(seed)

        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=8192,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            response_model=MORContrastivePair,
            max_retries=0,
        )

        # Inject metadata from seed
        response.scenario_id = seed["scenario_id"]
        response.category = seed["category"]
        response.difficulty = seed["difficulty"]

        return response
