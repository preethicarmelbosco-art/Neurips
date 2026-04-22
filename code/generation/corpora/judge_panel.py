"""Shared multi-judge panel with majority voting.

Supports 1–N judges queried in parallel. Each judge is an OpenAI-compatible
endpoint (local Ollama, remote vLLM, cloud API, etc.). The panel returns a
verdict based on majority vote of boolean outcomes.

Configuration is via JudgeConfig dataclasses — one per judge.
"""

import asyncio
import logging
from dataclasses import dataclass

import httpx
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeConfig:
    """Connection details for a single LLM judge."""

    model: str
    base_url: str
    api_key: str = "ollama"
    label: str = ""  # human-readable name for logging

    def __post_init__(self):
        if not self.label:
            object.__setattr__(self, "label", f"{self.model}@{self.base_url}")


class JudgePanel:
    """Multi-judge panel that queries N judges and majority-votes.

    Args:
        configs: List of JudgeConfig (1–N judges).
        system_prompt: The system prompt sent to every judge.
        user_prompt_template: A format string with ``{text}`` placeholder.
        response_model: A Pydantic BaseModel with a boolean field named by
            ``verdict_field`` and optionally a list-of-str evidence field.
        verdict_field: Name of the boolean field in response_model to vote on.
        evidence_field: Name of the list[str] field to aggregate (optional).
    """

    def __init__(
        self,
        configs: list[JudgeConfig],
        system_prompt: str,
        user_prompt_template: str,
        response_model: type[BaseModel],
        verdict_field: str = "has_mental_state",
        evidence_field: str | None = "flagged_phrases",
    ):
        if not configs:
            raise ValueError("JudgePanel requires at least one JudgeConfig.")

        self.configs = configs
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.response_model = response_model
        self.verdict_field = verdict_field
        self.evidence_field = evidence_field

        # Build one instructor-wrapped async client per judge.
        # Disable HTTP keep-alive to avoid stale connection pool issues under nohup.
        self._judges: list[tuple[JudgeConfig, instructor.AsyncInstructor]] = []
        for cfg in configs:
            http_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=0,  # disable keep-alive
                ),
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
            raw = AsyncOpenAI(
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                http_client=http_client,
            )
            client = instructor.from_openai(raw, mode=instructor.Mode.JSON)
            self._judges.append((cfg, client))

        self.judge_count = len(self._judges)
        self.majority = self.judge_count // 2 + 1

        logger.info(
            "JudgePanel initialised with %d judge(s), majority=%d: %s",
            self.judge_count,
            self.majority,
            [c.label for c in configs],
        )

    async def _query_one(
        self, cfg: JudgeConfig, client: instructor.AsyncInstructor, text: str
    ) -> BaseModel | None:
        """Query a single judge. Returns the verdict or None on failure."""
        try:
            response = await client.chat.completions.create(
                model=cfg.model,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": self.user_prompt_template.format(text=text),
                    },
                ],
                response_model=self.response_model,
                max_retries=0,
            )
            return response
        except Exception as e:
            logger.warning("Judge %s failed: %s", cfg.label, e)
            return None

    async def vote(self, text: str) -> tuple[bool, str, list[str]]:
        """Query all judges in parallel and return majority verdict.

        Returns:
            (flagged, reason, evidence)
            - flagged: True if the majority voted True on verdict_field
            - reason: human-readable summary
            - evidence: aggregated evidence strings from all judges that voted True
        """
        tasks = [
            self._query_one(cfg, client, text) for cfg, client in self._judges
        ]
        results = await asyncio.gather(*tasks)

        true_votes = 0
        false_votes = 0
        failures = 0
        all_evidence: list[str] = []
        per_judge: list[str] = []
        per_judge_verdicts: dict[str, str] = {}

        for cfg_client, result in zip(self._judges, results):
            cfg = cfg_client[0]
            if result is None:
                failures += 1
                per_judge.append(f"{cfg.label}=ERROR")
                per_judge_verdicts[cfg.label] = "ERROR"
                continue

            verdict_val = getattr(result, self.verdict_field)
            if verdict_val:
                true_votes += 1
                per_judge.append(f"{cfg.label}=TRUE")
                per_judge_verdicts[cfg.label] = "TRUE"
                if self.evidence_field:
                    evidence = getattr(result, self.evidence_field, []) or []
                    all_evidence.extend(evidence[:5])
            else:
                false_votes += 1
                per_judge.append(f"{cfg.label}=FALSE")
                per_judge_verdicts[cfg.label] = "FALSE"

        flagged = true_votes >= self.majority
        reason = (
            f"Majority vote {true_votes}/{self.judge_count} "
            f"(need {self.majority}): [{', '.join(per_judge)}]"
        )

        logger.info("JudgePanel verdict: flagged=%s — %s", flagged, reason)
        logger.debug("JUDGE_VOTE true=%d false=%d errors=%d per_judge=%s",
                     true_votes, false_votes, failures,
                     per_judge_verdicts)
        self.last_per_judge_verdicts = per_judge_verdicts
        return flagged, reason, all_evidence


def load_judge_configs_from_env(
    prefix: str = "JUDGE",
    max_judges: int = 3,
    fallback_model: str | None = None,
    fallback_base_url: str = "http://localhost:11434/v1",
    fallback_api_key: str = "ollama",
) -> list[JudgeConfig]:
    """Load judge configs from environment variables.

    Looks for:
        {prefix}1_MODEL, {prefix}1_BASE_URL, {prefix}1_API_KEY
        {prefix}2_MODEL, {prefix}2_BASE_URL, {prefix}2_API_KEY
        ...up to max_judges

    Falls back to legacy single-judge vars if no numbered vars found:
        {prefix}_MODEL, {prefix}_BASE_URL, {prefix}_API_KEY

    If still nothing found, uses fallback_model/base_url/api_key as a
    single local judge (typically the same model used for generation).
    """
    import os

    configs: list[JudgeConfig] = []

    for i in range(1, max_judges + 1):
        model = os.getenv(f"{prefix}{i}_MODEL")
        if not model:
            continue
        base_url = os.getenv(f"{prefix}{i}_BASE_URL", "http://localhost:11434/v1")
        api_key = os.getenv(f"{prefix}{i}_API_KEY", "ollama")
        label = os.getenv(f"{prefix}{i}_LABEL", f"judge-{i}")
        configs.append(JudgeConfig(
            model=model, base_url=base_url, api_key=api_key, label=label,
        ))

    # Fallback: legacy single-judge config
    if not configs:
        model = os.getenv(f"{prefix}_MODEL")
        if model:
            base_url = os.getenv(f"{prefix}_BASE_URL", "http://localhost:11434/v1")
            api_key = os.getenv(f"{prefix}_API_KEY", "ollama")
            configs.append(JudgeConfig(
                model=model, base_url=base_url, api_key=api_key, label="judge-1",
            ))

    # Final fallback: use the generator model as a single local judge
    if not configs and fallback_model:
        configs.append(JudgeConfig(
            model=fallback_model,
            base_url=fallback_base_url,
            api_key=fallback_api_key,
            label="local-fallback",
        ))

    return configs