"""LLM Generation Factory for STR-CC (Strategic Reasoning) contrastive pairs.

Uses instructor + OpenAI client (Ollama or any OpenAI-compatible API)
to produce STRContrastivePair objects.
"""

import logging

import instructor
from openai import AsyncOpenAI

from .str_models import STRContrastivePair
from .str_prompts import build_system_prompt, build_user_prompt

logger = logging.getLogger(__name__)


class STRPairFactory:
    """Factory that creates STRContrastivePair objects via LLM calls."""

    def __init__(
        self,
        model: str = "deepseek-r1:32b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ):
        self.model = model
        raw_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)

    async def create(self, seed: dict) -> STRContrastivePair:
        """Generate a single STR-CC contrastive pair for the given scenario skeleton.

        Args:
            seed: Dict with keys: scenario_id, domain, actor_a, actor_b,
                  asset, trigger, location, category, difficulty

        Returns a validated Pydantic STRContrastivePair object.
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
            response_model=STRContrastivePair,
            max_retries=0,
        )

        # Inject metadata from seed
        response.scenario_id = seed["scenario_id"]
        response.category = seed["category"]
        response.difficulty = seed["difficulty"]

        return response
