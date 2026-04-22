"""LLM Generation Factory for CTR-CC contrastive pairs.

Uses instructor + OpenAI client (Ollama / local DeepSeek) to produce
CTRContrastivePair objects.
"""

import logging

import instructor
from openai import AsyncOpenAI

from .ctr_models import CTRContrastivePair
from .ctr_prompts import build_system_prompt, build_user_prompt

logger = logging.getLogger(__name__)


class CTRPairFactory:
    """Factory that creates CTRContrastivePair objects via LLM calls."""

    def __init__(
        self,
        model: str = "deepseek-r1:32b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ):
        self.model = model
        raw_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)

    async def create(self, seed: dict) -> CTRContrastivePair:
        """Generate a single CTR-CC contrastive pair for the given scenario.

        Args:
            seed: Dict with keys from generate_stratified_seeds()

        Returns a validated Pydantic CTRContrastivePair object.
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
            response_model=CTRContrastivePair,
            max_retries=0,
        )

        # Inject metadata from seed
        response.scenario_id = seed["scenario_id"]
        response.category = seed["category"]
        response.domain = seed["domain"]
        response.difficulty = seed["difficulty"]

        return response
