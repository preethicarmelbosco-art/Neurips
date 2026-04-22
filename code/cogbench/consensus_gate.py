"""Consensus Gate — generalized majority-voting validator for contrastive pairs.

Wraps JudgePanel to provide structured per-judge verdict tracking,
Table 2b dissent statistics, and JSONL-persistable verdict records.

Works with any corpus's JudgeVerdict model (ToM, MOR, STR, CTR, SPL, STP)
as long as it has a boolean verdict field and a list[str] evidence field.

Paper reference: Section 3.4 — Three-Judge Consensus Gate.
Table 2b columns: 3/3 Agree Clean, 2/3 Clean (1 dissent), Majority Reject.

Usage (inline in pipeline):
    gate = ConsensusGate.from_env(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt_template=JUDGE_USER_PROMPT,
        response_model=ToMJudgeVerdict,
        verdict_field="has_mental_state",
    )
    result = await gate.evaluate(retain_text)
    if result.passed:
        # accept pair
    pair["_consensus_gate"] = result.to_dict()

Usage (batch revalidation):
    gate = ConsensusGate(panel=existing_panel)
    results = await gate.evaluate_batch(pairs, text_field="retain")
    stats = ConsensusStats.from_results(results)
    print(stats.table2b_summary())
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from .judge_panel import JudgeConfig, JudgePanel, load_judge_configs_from_env

logger = logging.getLogger(__name__)


@dataclass
class JudgeVote:
    """A single judge's vote on one text."""

    label: str
    flagged: bool
    evidence: list[str] = field(default_factory=list)
    error: bool = False

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"vote": self.flagged}
        if self.evidence:
            d["evidence"] = self.evidence
        if self.error:
            d["error"] = True
        return d


@dataclass
class ConsensusVerdict:
    """Structured result of the consensus gate for one text.

    Attributes:
        passed: True if the text passed (majority of judges did NOT flag it).
        flagged: True if majority flagged it (i.e., not passed).
        votes: Per-judge vote details.
        n_flagged: Number of judges that flagged the text.
        n_clean: Number of judges that did not flag.
        n_error: Number of judges that failed.
        majority_needed: Threshold for flagging.
        evidence: Aggregated evidence from flagging judges.
    """

    passed: bool
    flagged: bool
    votes: dict[str, JudgeVote] = field(default_factory=dict)
    n_flagged: int = 0
    n_clean: int = 0
    n_error: int = 0
    majority_needed: int = 2
    evidence: list[str] = field(default_factory=list)

    @property
    def agreement_category(self) -> str:
        """Classify for Table 2b: 'all_clean', 'one_dissent', 'majority_reject'."""
        if self.n_flagged == 0:
            return "all_clean"
        elif self.n_flagged < self.majority_needed:
            return "one_dissent"
        else:
            return "majority_reject"

    @property
    def dissenting_judges(self) -> list[str]:
        """Labels of judges that disagreed with the majority."""
        if self.flagged:
            # Majority flagged → dissenters are the clean ones
            return [k for k, v in self.votes.items() if not v.flagged and not v.error]
        else:
            # Majority clean → dissenters are the flaggers
            return [k for k, v in self.votes.items() if v.flagged]

    def to_dict(self) -> dict:
        """Serialize for JSONL persistence."""
        return {
            "verdict": "REJECT" if self.flagged else "PASS",
            "n_flagged": self.n_flagged,
            "n_clean": self.n_clean,
            "agreement": self.agreement_category,
            "per_judge": {k: v.to_dict() for k, v in self.votes.items()},
            "evidence": self.evidence[:10],
        }


@dataclass
class ConsensusStats:
    """Aggregated statistics across many consensus verdicts (Table 2b)."""

    total: int = 0
    all_clean: int = 0       # 3/3 agree clean
    one_dissent: int = 0     # 2/3 clean, 1 flags
    majority_reject: int = 0  # 2+ flag → rejected
    judge_dissent_counts: Counter = field(default_factory=Counter)
    # Table 2b: per-judge flag counts (not just dissent, but total flags)
    judge_flag_counts: Counter = field(default_factory=Counter)
    # Table 1: number of pairs that passed (accepted into corpus)
    n_passed: int = 0
    # Borderline pairs (exactly at majority threshold)
    n_borderline: int = 0
    # Per-difficulty tracking (for Appendix difficulty tables)
    difficulty_stats: dict[str, dict] = field(default_factory=dict)
    category_stats: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_results(
        cls,
        results: list[tuple[dict, ConsensusVerdict]],
        category_field: str = "category",
    ) -> ConsensusStats:
        """Build stats from a list of (pair, verdict) tuples."""
        stats = cls()
        cat_total: dict[str, int] = defaultdict(int)
        cat_rejected: dict[str, int] = defaultdict(int)
        diff_total: dict[str, int] = defaultdict(int)
        diff_rejected: dict[str, int] = defaultdict(int)

        for pair, verdict in results:
            stats.total += 1
            cat = verdict.agreement_category
            if cat == "all_clean":
                stats.all_clean += 1
            elif cat == "one_dissent":
                stats.one_dissent += 1
            elif cat == "majority_reject":
                stats.majority_reject += 1

            if verdict.passed:
                stats.n_passed += 1

            # Borderline: exactly at majority threshold (e.g., 2 flagged out of 3)
            if verdict.n_flagged == verdict.majority_needed:
                stats.n_borderline += 1

            # Track which judges dissent (from majority)
            for judge_label in verdict.dissenting_judges:
                stats.judge_dissent_counts[judge_label] += 1

            # Track total flags per judge (Table 2b: which judge flags most)
            for judge_label, vote in verdict.votes.items():
                if vote.flagged:
                    stats.judge_flag_counts[judge_label] += 1

            # Per-category tracking
            category = pair.get(category_field, pair.get("seed_topic", "unknown"))
            cat_total[category] += 1
            if verdict.flagged:
                cat_rejected[category] += 1

            # Per-difficulty tracking (for Appendix tables A1, A2)
            difficulty = pair.get("difficulty", "unknown")
            diff_total[difficulty] += 1
            if verdict.flagged:
                diff_rejected[difficulty] += 1

        stats.category_stats = {
            cat: {
                "total": cat_total[cat],
                "rejected": cat_rejected[cat],
                "rate": round(cat_rejected[cat] / cat_total[cat], 4) if cat_total[cat] > 0 else 0,
            }
            for cat in sorted(cat_total.keys())
        }
        stats.difficulty_stats = {
            diff: {
                "total": diff_total[diff],
                "rejected": diff_rejected[diff],
                "rate": round(diff_rejected[diff] / diff_total[diff], 4) if diff_total[diff] > 0 else 0,
            }
            for diff in sorted(diff_total.keys())
        }
        return stats

    @property
    def rejection_rate(self) -> float:
        return self.majority_reject / self.total if self.total > 0 else 0

    @property
    def single_dissent_rate(self) -> float:
        return self.one_dissent / self.total if self.total > 0 else 0

    def to_dict(self) -> dict:
        """Serialize for JSON output (Tables 1, 2b)."""
        d = {
            # Table 1: Corpus statistics
            "total_pairs_judged": self.total,
            "pairs_accepted": self.n_passed,
            "pairs_rejected": self.majority_reject,
            "rejection_rate": round(self.rejection_rate, 4),
            # Table 2b: Cross-model consensus
            "3_of_3_agree_clean": self.all_clean,
            "2_of_3_agree_clean_1_dissent": self.one_dissent,
            "majority_reject_2plus_flag": self.majority_reject,
            "single_dissent_rate": round(self.single_dissent_rate, 4),
            "n_borderline": self.n_borderline,
            # Table 2b: Per-judge breakdown
            "per_judge_dissent_counts": dict(self.judge_dissent_counts),
            "per_judge_flag_counts": dict(self.judge_flag_counts),
            # Per-category breakdown (for targeted analysis)
            "per_category": self.category_stats,
            # Appendix: Per-difficulty breakdown
            "per_difficulty": self.difficulty_stats,
        }
        # Per-judge flag rate (Table 2b: which judge flags most often)
        if self.total > 0 and self.judge_flag_counts:
            d["per_judge_flag_rate"] = {
                j: round(c / self.total, 4) for j, c in self.judge_flag_counts.items()
            }
        return d

    def log_stats(self, corpus_name: str = "unknown"):
        """Log Table 2b stats to the logger for persistent extraction."""
        logger.info("TABLE2B_STATS corpus=%s total=%d accepted=%d rejected=%d "
                    "all_clean=%d one_dissent=%d majority_reject=%d "
                    "rejection_rate=%.4f single_dissent_rate=%.4f borderline=%d",
                    corpus_name, self.total, self.n_passed, self.majority_reject,
                    self.all_clean, self.one_dissent, self.majority_reject,
                    self.rejection_rate, self.single_dissent_rate, self.n_borderline)
        for judge, count in self.judge_flag_counts.items():
            rate = count / self.total if self.total > 0 else 0
            logger.info("TABLE2B_JUDGE corpus=%s judge=%s flags=%d rate=%.4f",
                        corpus_name, judge, count, rate)
        for cat, cs in self.category_stats.items():
            logger.debug("TABLE2B_CATEGORY corpus=%s category=%s total=%d rejected=%d rate=%.4f",
                         corpus_name, cat, cs["total"], cs["rejected"], cs["rate"])

    def save_json(self, output_path: Path, corpus_name: str = "unknown"):
        """Save Table 2b stats to JSON file for paper extraction."""
        import json
        from pathlib import Path as _Path
        data = self.to_dict()
        data["corpus"] = corpus_name
        _Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("TABLE2B_SAVED corpus=%s path=%s", corpus_name, output_path)

    def table2b_summary(self) -> str:
        """Human-readable Table 2b summary."""
        lines = [
            f"Total pairs judged: {self.total}",
            f"  Accepted:           {self.n_passed} ({100*self.n_passed/max(self.total,1):.1f}%)",
            f"  3/3 agree clean:    {self.all_clean} ({100*self.all_clean/max(self.total,1):.1f}%)",
            f"  2/3 clean (1 diss): {self.one_dissent} ({100*self.single_dissent_rate:.1f}%)",
            f"  Majority reject:    {self.majority_reject} ({100*self.rejection_rate:.1f}%)",
            f"  Borderline (at threshold): {self.n_borderline}",
        ]
        if self.judge_flag_counts:
            lines.append(f"  Per-judge flag counts: {dict(self.judge_flag_counts)}")
            if self.total > 0:
                rates = {j: f"{100*c/self.total:.1f}%" for j, c in self.judge_flag_counts.items()}
                lines.append(f"  Per-judge flag rates:  {rates}")
        if self.judge_dissent_counts:
            lines.append(f"  Per-judge dissents:   {dict(self.judge_dissent_counts)}")
        if self.difficulty_stats:
            lines.append(f"  Per-difficulty rejection rates:")
            for diff, ds in sorted(self.difficulty_stats.items()):
                lines.append(f"    {diff}: {ds['rejected']}/{ds['total']} ({100*ds['rate']:.1f}%)")
        return "\n".join(lines)


class ConsensusGate:
    """Generalized consensus gate wrapping a JudgePanel.

    Provides structured verdicts (not just string reasons) and
    computes Table 2b statistics from structured data.
    """

    def __init__(self, panel: JudgePanel):
        self.panel = panel

    @classmethod
    def from_env(
        cls,
        system_prompt: str,
        user_prompt_template: str,
        response_model: type[BaseModel],
        verdict_field: str,
        evidence_field: str = "flagged_phrases",
        env_prefix: str = "JUDGE",
        fallback_model: str | None = None,
    ) -> ConsensusGate:
        """Create a ConsensusGate from environment variables."""
        configs = load_judge_configs_from_env(
            prefix=env_prefix,
            fallback_model=fallback_model,
        )
        if not configs:
            raise RuntimeError(
                f"No judge configs found (env prefix={env_prefix}). "
                "Set JUDGE1_MODEL, JUDGE2_MODEL, JUDGE3_MODEL."
            )
        panel = JudgePanel(
            configs=configs,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            response_model=response_model,
            verdict_field=verdict_field,
            evidence_field=evidence_field,
        )
        return cls(panel=panel)

    @classmethod
    def from_configs(
        cls,
        configs: list[JudgeConfig],
        system_prompt: str,
        user_prompt_template: str,
        response_model: type[BaseModel],
        verdict_field: str,
        evidence_field: str = "flagged_phrases",
    ) -> ConsensusGate:
        """Create a ConsensusGate from explicit JudgeConfig list."""
        panel = JudgePanel(
            configs=configs,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            response_model=response_model,
            verdict_field=verdict_field,
            evidence_field=evidence_field,
        )
        return cls(panel=panel)

    async def evaluate(self, text: str) -> ConsensusVerdict:
        """Run text through the consensus gate.

        Returns a structured ConsensusVerdict with per-judge breakdown.
        """
        flagged, reason, evidence = await self.panel.vote(text)

        # Build structured votes from panel's last_per_judge_verdicts
        votes: dict[str, JudgeVote] = {}
        per_judge = getattr(self.panel, "last_per_judge_verdicts", {})

        n_flagged = 0
        n_clean = 0
        n_error = 0

        for label, vote_str in per_judge.items():
            if vote_str == "TRUE":
                votes[label] = JudgeVote(label=label, flagged=True, evidence=evidence)
                n_flagged += 1
            elif vote_str == "FALSE":
                votes[label] = JudgeVote(label=label, flagged=False)
                n_clean += 1
            else:  # ERROR
                votes[label] = JudgeVote(label=label, flagged=False, error=True)
                n_error += 1

        return ConsensusVerdict(
            passed=not flagged,
            flagged=flagged,
            votes=votes,
            n_flagged=n_flagged,
            n_clean=n_clean,
            n_error=n_error,
            majority_needed=self.panel.majority,
            evidence=evidence,
        )

    async def evaluate_batch(
        self,
        pairs: list[dict],
        text_field: str = "retain",
        max_concurrent: int = 3,
    ) -> list[tuple[dict, ConsensusVerdict]]:
        """Evaluate a batch of pairs through the consensus gate.

        Args:
            pairs: List of pair dicts containing the text to evaluate.
            text_field: Key in each pair dict holding the text to judge.
            max_concurrent: Max parallel judge requests.

        Returns:
            List of (pair, verdict) tuples in input order.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results: list[tuple[dict, ConsensusVerdict]] = [None] * len(pairs)  # type: ignore

        async def process(idx: int, pair: dict) -> None:
            async with semaphore:
                text = pair.get(text_field, "")
                if not text:
                    results[idx] = (pair, ConsensusVerdict(
                        passed=False, flagged=True,
                        majority_needed=self.panel.majority,
                    ))
                    return
                try:
                    verdict = await self.evaluate(text)
                except Exception as e:
                    logger.warning("Gate error for pair %s: %s", pair.get("id", idx), e)
                    verdict = ConsensusVerdict(
                        passed=True, flagged=False,
                        n_error=self.panel.judge_count,
                        majority_needed=self.panel.majority,
                    )
                results[idx] = (pair, verdict)

        tasks = [process(i, p) for i, p in enumerate(pairs)]

        # Process in batches with progress logging
        done = 0
        batch_size = 10
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            await asyncio.gather(*batch)
            done += len(batch)
            if done % 50 == 0 or done == len(pairs):
                passed = sum(1 for r in results[:done] if r and r[1].passed)
                logger.info(
                    "ConsensusGate progress: %d/%d | passed=%d rejected=%d",
                    done, len(pairs), passed, done - passed,
                )

        return results
