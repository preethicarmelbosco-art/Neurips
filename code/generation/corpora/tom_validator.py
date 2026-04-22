"""Validation engine for ToM contrastive pairs.

Uses LLM-as-Judge for semantic validation of retain text.
Only basic structural checks (length) are done via code — no regex
mental-state matching, because mental-state words are too
context-dependent for reliable regex detection (e.g., "plans" as noun
vs verb, "felt" as physical vs emotional, "trusted" as adjective vs verb).
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

JUDGE_SYSTEM_PROMPT = (
    "You are a linguistic auditor. Your job is to determine whether "
    "a given text attributes INTERNAL mental states to any character — "
    "specifically: beliefs, knowledge, awareness, intentions, emotions, "
    "expectations, assumptions, suspicions, hopes, fears, or desires.\n\n"
    "ALLOWED (do NOT flag these):\n"
    "- Observable physical actions: 'picked up the phone', 'opened the door', "
    "'called security', 'took notes', 'reviewed the document'\n"
    "- Observable body language and facial expressions: 'frowned', 'smiled', "
    "'nodded', 'narrowed her eyes', 'crossed arms', 'leaned forward', "
    "'looked around', 'stood motionless', 'glanced at'\n"
    "- Observable vocal/speech acts: 'said', 'asked', 'replied', 'stated', "
    "'mentioned', 'explained', 'discussed', 'objected', 'agreed'\n"
    "- Direct dialogue and quotes: anything in quotation marks\n"
    "- Physical sensations: 'felt the cold metal', 'heard a noise'\n"
    "- Nouns sharing form with mental verbs: 'building plans', "
    "'a trusted colleague', 'the decision document'\n"
    "- Temporal/behavioral descriptions: 'paused', 'waited', 'hesitated'\n\n"
    "FLAG (these ARE mental-state attributions):\n"
    "- Beliefs: 'she believed the documents were safe', 'he was convinced'\n"
    "- Knowledge/awareness: 'she knew about the transfer', 'unaware that'\n"
    "- Intentions: 'he intended to mislead', 'planning to conceal'\n"
    "- Emotions: 'felt betrayed', 'was anxious about', 'feared the outcome'\n"
    "- Expectations: 'expected the report to clear him', 'anticipated trouble'\n"
    "- Suspicions: 'suspected foul play', 'grew suspicious'\n"
    "- Desires: 'wanted to protect her reputation', 'hoped for a transfer'\n"
    "- Realizations: 'realized the truth', 'suddenly understood'\n\n"
    "FLAG ONLY genuine INTERNAL mental-state attribution — where the "
    "narrator reveals what a character thinks, believes, feels emotionally, "
    "wants, intends, knows, suspects, or realizes INSIDE THEIR MIND. "
    "The key test: could a security camera or courtroom observer verify it? "
    "If yes, it is observable and should NOT be flagged.\n\n"
    "Respond with a JSON object:\n"
    '{"has_mental_state": true/false, "flagged_phrases": ["..."]}'
)

JUDGE_USER_PROMPT = (
    "Does the following text attribute any mental states to any character? "
    "Analyze carefully:\n\n{text}"
)
