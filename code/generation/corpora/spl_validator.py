"""Validation engine — the Gatekeeper.

Enforces strict structural separation between target and retain texts.
"""

import re

# Matches any digit
_DIGIT_RE = re.compile(r"\d")

# Matches math operators: =, +, *, /, ^, ×, ÷
# Hyphen (-) is allowed in natural English (e.g., "well-known", "twentieth-century")
# but rejected when it looks like a math operator (standalone or adjacent to digits)
_MATH_OPS_RE = re.compile(r"[=+*/^×÷]")

# Standalone minus or minus adjacent to digits (math subtraction)
_MATH_MINUS_RE = re.compile(r"(?<!\w)-(?!\w)|(?<=\d)-|-(?=\d)")

# Superscript digits and math symbols
_UNICODE_MATH_RE = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾₀₁₂₃₄₅₆₇₈₉]")


def validate_target(text: str) -> tuple[bool, str]:
    """Validate that d_target contains mathematical content.

    Requirements:
        - At least one equals sign (=)
        - At least one digit
    """
    if "=" not in text:
        return False, "Target missing equals sign (=) — no equations found."
    if not _DIGIT_RE.search(text):
        return False, "Target missing digits — no numerical values found."
    return True, "OK"


def validate_retain(text: str) -> tuple[bool, str]:
    """Validate that d_retain contains ZERO mathematical content.

    Rejects:
        - Any digit (0-9)
        - Any math operator: =, +, *, /, ^, ×, ÷
        - Standalone minus signs or minus adjacent to digits
        - Unicode superscript/subscript digits
    """
    match = _DIGIT_RE.search(text)
    if match:
        pos = match.start()
        context = text[max(0, pos - 20):pos + 20]
        return False, f"Retain contains digit at pos {pos}: '...{context}...'"

    match = _MATH_OPS_RE.search(text)
    if match:
        pos = match.start()
        context = text[max(0, pos - 20):pos + 20]
        return False, f"Retain contains math operator '{match.group()}' at pos {pos}: '...{context}...'"

    match = _MATH_MINUS_RE.search(text)
    if match:
        pos = match.start()
        context = text[max(0, pos - 20):pos + 20]
        return False, f"Retain contains math minus at pos {pos}: '...{context}...'"

    match = _UNICODE_MATH_RE.search(text)
    if match:
        pos = match.start()
        context = text[max(0, pos - 20):pos + 20]
        return False, f"Retain contains unicode math char '{match.group()}' at pos {pos}: '...{context}...'"

    return True, "OK"


# --- LLM-as-Judge prompts for semantic validation ---

SPL_JUDGE_SYSTEM_PROMPT = (
    "You are a strict content auditor specializing in detecting mathematical "
    "and numerical content in text. Your job is to determine whether a given "
    "text contains ANY mathematical content whatsoever.\n\n"
    "FLAG these:\n"
    "- Any digits or numbers (including spelled-out numbers used mathematically)\n"
    "- Equations, formulas, or expressions (e.g., F=ma, E=mc^2)\n"
    "- Mathematical operators (+, -, *, /, =, ^)\n"
    "- Units with numerical values (e.g., '5 kg', '3 meters')\n"
    "- Step-by-step calculations or derivations\n"
    "- Variable bindings (e.g., 'let x = ...')\n\n"
    "DO NOT flag:\n"
    "- Conceptual descriptions of physics without numbers or formulas\n"
    "- Historical narratives about scientists\n"
    "- Qualitative descriptions (e.g., 'heavier', 'faster', 'proportional')\n"
    "- Ordinal words in prose context (e.g., 'the first law', 'second edition')\n\n"
    "Respond with a JSON object:\n"
    '{"has_math_content": true/false, "flagged_phrases": ["..."]}'
)

SPL_JUDGE_USER_PROMPT = (
    "Does the following text contain any mathematical content (numbers, "
    "equations, formulas, calculations)? Analyze carefully:\n\n{text}"
)


_MIN_LENGTH = 100
_MAX_LENGTH = 2000


def validate_pair(target: str, retain: str) -> tuple[bool, str]:
    """Validate a full contrastive pair. Returns (pass, reason)."""
    if len(target) < _MIN_LENGTH:
        return False, f"Target too short ({len(target)} chars, min {_MIN_LENGTH})."
    if len(target) > _MAX_LENGTH:
        return False, f"Target too long ({len(target)} chars, max {_MAX_LENGTH})."
    if len(retain) < _MIN_LENGTH:
        return False, f"Retain too short ({len(retain)} chars, min {_MIN_LENGTH})."
    if len(retain) > _MAX_LENGTH:
        return False, f"Retain too long ({len(retain)} chars, max {_MAX_LENGTH})."

    ok, reason = validate_target(target)
    if not ok:
        return False, reason

    ok, reason = validate_retain(retain)
    if not ok:
        return False, reason

    return True, "OK"
