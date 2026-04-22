"""Validation for STP-CC (Spatial-Temporal Tracking) contrastive pairs.

Uses regex validation (no LLM-as-Judge needed) — locative state assertions
are syntactically unambiguous, unlike mental-state verbs or moral language.

Two checks:
  1. Length validation (both texts 100-2000 chars)
  2. Regex gate:
     - Target MUST contain at least one locative assertion
     - Retain MUST contain ZERO locative assertions
"""

import re

MIN_LENGTH = 100
MAX_LENGTH = 2000

# Locative state assertions — patterns that reveal WHERE an object IS
LOCATIVE_PATTERN = re.compile(
    r'\b('
    r'is\s+(now\s+)?(in|at|on|inside|located)\b|'
    r'are\s+(now\s+)?(in|at|on|inside|located)\b|'
    r'remains?\s+(at|in|on)\b|'
    r'currently\s+(in|at|on)\b|'
    r'now\s+(in|at|on|sits?)\b|'
    r'located\s+(in|at|on)\b|'
    r'moved\s+to\b|'
    r'transferred\s+to\b|'
    r'placed\s+(in|at|on)\b|'
    r'stored\s+(in|at|on)\b|'
    r'sitting\s+(in|at|on)\b|'
    r'resting\s+(in|at|on)\b|'
    r'ended\s+up\s+(in|at)\b|'
    r'wound\s+up\s+(in|at)\b|'
    r'brought\s+.{0,30}\s+to\b|'
    r'carried\s+.{0,30}\s+to\b|'
    r'delivered\s+.{0,30}\s+to\b'
    r')',
    re.IGNORECASE,
)


def validate_lengths(target: str, retain: str) -> tuple[bool, str]:
    """Check both texts meet length requirements."""
    if len(target) < MIN_LENGTH:
        return False, f"Target too short ({len(target)} < {MIN_LENGTH})"
    if len(target) > MAX_LENGTH:
        return False, f"Target too long ({len(target)} > {MAX_LENGTH})"
    if len(retain) < MIN_LENGTH:
        return False, f"Retain too short ({len(retain)} < {MIN_LENGTH})"
    if len(retain) > MAX_LENGTH:
        return False, f"Retain too long ({len(retain)} > {MAX_LENGTH})"
    return True, "OK"


def validate_target_has_locative(target: str) -> tuple[bool, str]:
    """Target MUST contain at least one locative state assertion."""
    if LOCATIVE_PATTERN.search(target):
        return True, "OK"
    return False, "Target lacks locative state assertions (no 'is in/at', 'moved to', etc.)"


def validate_retain_no_locative(retain: str) -> tuple[bool, str]:
    """Retain MUST NOT contain any locative state assertions."""
    match = LOCATIVE_PATTERN.search(retain)
    if match:
        return False, f"Retain contains locative assertion: '{match.group()}' at position {match.start()}"
    return True, "OK"


def validate_pair(target: str, retain: str) -> tuple[bool, str]:
    """Full validation: lengths + regex gates.

    Returns (passed, reason).
    """
    ok, reason = validate_lengths(target, retain)
    if not ok:
        return False, reason

    ok, reason = validate_target_has_locative(target)
    if not ok:
        return False, reason

    ok, reason = validate_retain_no_locative(retain)
    if not ok:
        return False, reason

    return True, "OK"
