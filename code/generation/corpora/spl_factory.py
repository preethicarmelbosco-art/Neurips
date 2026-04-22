"""LLM Generation Factory — uses instructor + OpenAI client (Ollama) to produce ContrastivePairs."""

import logging

import instructor
from openai import AsyncOpenAI

from .spl_models import ContrastivePair
from .spl_prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class PairFactory:
    """Factory that creates ContrastivePair objects via LLM calls."""

    def __init__(
        self,
        model: str = "deepseek-r1:32b",
        base_url: str = "http://localhost:11434/v1",
    ):
        self.model = model
        raw_client = AsyncOpenAI(base_url=base_url, api_key="ollama")
        self.client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)

    async def create(self, seed_topic: str) -> ContrastivePair:
        """Generate a single contrastive pair for the given seed topic.

        Returns a validated Pydantic ContrastivePair object.
        Raises on API or parsing errors — caller handles retries.
        """
        user_msg = USER_PROMPT_TEMPLATE.format(seed_topic=seed_topic)

        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=8192,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_model=ContrastivePair,
            max_retries=0,
        )

        return response
