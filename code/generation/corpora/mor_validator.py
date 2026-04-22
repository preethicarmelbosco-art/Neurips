"""Validation engine for MOR-CC (Moral Reasoning) contrastive pairs.

Uses LLM-as-Judge for semantic validation of retain text.
Only basic structural checks (length) are done via code — no regex
moral-evaluation matching, because evaluative language is too
context-dependent for reliable regex detection (e.g., "right" as
directional vs moral, "just" as temporal vs ethical, "fair" as weather
vs moral judgment).
"""

import logging

logger = logging.getLogger(__name__)

_MIN_LENGTH = 100
_MAX_LENGTH = 2000


def validate_target_length(text: str) -> tuple[bool, str]:
    """Check target text meets length requirements."""
    if len(text) < _MIN_LENGTH:
        return False, f"Target too short ({len(text)} chars, min {_MIN_LENGTH})."
    if len(text) > _MAX_LENGTH:
        return False, f"Target too long ({len(text)} chars, max {_MAX_LENGTH})."
    return True, "OK"


def validate_retain_length(text: str) -> tuple[bool, str]:
    """Check retain text meets length requirements."""
    if len(text) < _MIN_LENGTH:
        return False, f"Retain too short ({len(text)} chars, min {_MIN_LENGTH})."
    if len(text) > _MAX_LENGTH:
        return False, f"Retain too long ({len(text)} chars, max {_MAX_LENGTH})."
    return True, "OK"


def validate_lengths(target: str, retain: str) -> tuple[bool, str]:
    """Validate basic structural constraints on both texts."""
    ok, reason = validate_target_length(target)
    if not ok:
        return False, reason

    ok, reason = validate_retain_length(retain)
    if not ok:
        return False, reason

    return True, "OK"


# --- LLM-as-Judge prompt for semantic validation ---

MOR_JUDGE_SYSTEM_PROMPT = (
    "You are a linguistic auditor. Your job is to determine whether "
    "a given text makes moral judgments, evaluates rightness or wrongness, "
    "references ethical principles, or expresses moral approval/disapproval "
    "of any action or decision.\n\n"
    "ALLOWED (do NOT flag these):\n"
    "- Factual descriptions of actions: 'allocated resources', 'signed the form', "
    "'discharged the patient', 'filed the report', 'implemented the policy'\n"
    "- Procedural/legal language: 'liability', 'compliance', 'regulation', "
    "'mandate', 'protocol', 'jurisdiction', 'statute'\n"
    "- Directional/non-moral uses: 'the right side', 'just then', 'fair weather', "
    "'right angle', 'just barely'\n"
    "- Descriptions of outcomes without evaluation: 'the patient died', "
    "'the company lost revenue', 'three people were displaced'\n"
    "- Reported speech in quotes: direct quotes of what someone said\n"
    "- Job titles and roles: 'ethics committee chair' (a role, not a judgment)\n"
    "- Temporal/causal descriptions: 'as a result', 'consequently', 'therefore'\n\n"
    "FLAG ONLY genuine moral/ethical evaluation — where the narrator or a "
    "character explicitly judges an action as right, wrong, fair, unfair, "
    "ethical, unethical, justified, or unjustified, OR invokes moral principles "
    "(justice, autonomy, beneficence, duty, rights, dignity, fairness) as "
    "reasoning for a decision. The key test: does the text EVALUATE or merely "
    "DESCRIBE? If it only describes what happened without judging it, do NOT flag.\n\n"
    "Respond with a JSON object:\n"
    '{"has_moral_evaluation": true/false, "flagged_phrases": ["..."]}'
)

MOR_JUDGE_USER_PROMPT = (
    "Does the following text make any moral judgments, evaluate rightness "
    "or wrongness, or reference ethical principles? Analyze carefully:\n\n{text}"
)
