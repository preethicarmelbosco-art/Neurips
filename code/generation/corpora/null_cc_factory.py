"""LLM Generation Factory for NULL-CC contrastive pairs.

Uses instructor + OpenAI client (Ollama / local DeepSeek) to produce
NullCCContrastivePair objects.
"""

import logging

import instructor
from openai import AsyncOpenAI

from .null_cc_models import NullCCContrastivePair
from .null_cc_prompts import build_system_prompt, build_user_prompt

logger = logging.getLogger(__name__)


class NullCCPairFactory:
    """Factory that creates NullCCContrastivePair objects via LLM calls."""

    def __init__(
        self,
        model: str = "deepseek-r1:32b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ):
        self.model = model
        raw_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)

    async def create(self, seed: dict) -> NullCCContrastivePair:
        """Generate a single NULL-CC contrastive pair for the given scenario.

        Args:
            seed: Dict with keys from generate_stratified_seeds()

        Returns a validated Pydantic NullCCContrastivePair object.
        Raises on API or parsing errors — caller handles retries.
        """
        system_msg = build_system_prompt(seed["category"], seed["complexity"])
        user_msg = build_user_prompt(seed)

        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            response_model=NullCCContrastivePair,
            max_retries=0,
        )

        # Inject metadata from seed
        response.scenario_id = seed["scenario_id"]
        response.category = seed["category"]
        response.complexity = seed["complexity"]

        return response
