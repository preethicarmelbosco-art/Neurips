"""Validation engine for CORE-MATH contrastive pairs.

Enforces:
  - target_proof has formal logic markers AND equations
  - retain_intuition has ZERO formal logic markers, ZERO equations, ZERO step numbering
  - Both texts meet length constraints
  - Concept overlap check (bijectivity)
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Formal logic markers required in target_proof ──────────────────────────

_FORMAL_LOGIC_PATTERN = re.compile(
    r'\b('
    r'therefore\b|'
    r'hence\b|'
    r'thus\b|'
    r'QED\b|'
    r'it\s+follows\b|'
    r'by\s+definition\b|'
    r'by\s+assumption\b|'
    r'by\s+hypothesis\b|'
    r'we\s+have\b|'
    r'we\s+get\b|'
    r'we\s+obtain\b|'
    r'implies\s+that\b|'
    r'if\s+and\s+only\s+if\b|'
    r'proof\b|'
    r'step\s+\d'
    r')',
    re.IGNORECASE
)

# ── Equation symbols required in target_proof ──────────────────────────────

_EQUATION_PATTERN = re.compile(
    r'[=\u2265\u2264><\u2260\u2208\u2200\u2203]'
    # Matches: = >= <= > < != (element of) (for all) (there exists)
)

# ── Patterns forbidden in retain_intuition ─────────────────────────────────

_FORBIDDEN_FORMAL_LOGIC = re.compile(
    r'\b('
    r'therefore\b|'
    r'hence\b|'
    r'thus\b|'
    r'QED\b|'
    r'it\s+follows\b|'
    r'by\s+definition\b|'
    r'by\s+assumption\b|'
    r'by\s+hypothesis\b|'
    r'we\s+have\b|'
    r'we\s+get\b|'
    r'we\s+obtain\b|'
    r'implies\s+that\b|'
    r'if\s+and\s+only\s+if\b|'
    r'proof\b|'
    r'step\s+\d'
    r')',
    re.IGNORECASE
)

_FORBIDDEN_EQUATION = re.compile(
    r'[=\u2265\u2264\u2260\u2208\u2200\u2203]'
    # Note: > and < are allowed in retain if used in natural language (e.g. "greater than")
    # but = and formal symbols are forbidden
)

_FORBIDDEN_STEP_NUMBERING = re.compile(
    r'(?:Step|step)\s*\d',
    re.IGNORECASE
)

_MIN_LENGTH = 100
_MAX_LENGTH = 2500


def validate_target(text: str) -> tuple[bool, str]:
    """Validate that target_proof contains formal proof structure.

    Requirements:
        - At least one formal logic marker (therefore, QED, step N, etc.)
        - At least one equation symbol (=, >=, <=, etc.)
        - Within length bounds
    """
    if len(text) < _MIN_LENGTH:
        return False, f"Target too short ({len(text)} chars, min {_MIN_LENGTH})."
    if len(text) > _MAX_LENGTH:
        return False, f"Target too long ({len(text)} chars, max {_MAX_LENGTH})."

    if not _FORMAL_LOGIC_PATTERN.search(text):
        return False, "Target missing formal logic marker (therefore, QED, step N, etc.)."

    if not _EQUATION_PATTERN.search(text):
        return False, "Target missing equation symbol (=, >=, <=, etc.)."

    return True, "OK"


def validate_retain(text: str) -> tuple[bool, str]:
    """Validate that retain_intuition contains ZERO formal mathematical markers.

    Intuition test: could someone with no math training follow this?
    """
    if len(text) < _MIN_LENGTH:
        return False, f"Retain too short ({len(text)} chars, min {_MIN_LENGTH})."
    if len(text) > _MAX_LENGTH:
        return False, f"Retain too long ({len(text)} chars, max {_MAX_LENGTH})."

    match = _FORBIDDEN_FORMAL_LOGIC.search(text)
    if match:
        pos = match.start()
        context = text[max(0, pos - 30):pos + 30]
        return False, (
            f"Retain contains formal logic marker '{match.group()}' "
            f"at pos {pos}: '...{context}...'"
        )

    match = _FORBIDDEN_EQUATION.search(text)
    if match:
        pos = match.start()
        context = text[max(0, pos - 30):pos + 30]
        return False, (
            f"Retain contains equation symbol '{match.group()}' "
            f"at pos {pos}: '...{context}...'"
        )

    match = _FORBIDDEN_STEP_NUMBERING.search(text)
    if match:
        pos = match.start()
        context = text[max(0, pos - 30):pos + 30]
        return False, (
            f"Retain contains step numbering '{match.group()}' "
            f"at pos {pos}: '...{context}...'"
        )

    return True, "OK"


def check_concept_overlap(
    target: str, retain: str, concept: str
) -> tuple[bool, str]:
    """Check that retain_intuition references the same mathematical concept.

    Args:
        target: the target proof text
        retain: the retain intuition text
        concept: the mathematical topic/concept string from the seed

    Returns (pass, reason).
    """
    # Extract key terms from the concept string
    concept_lower = concept.lower()
    # Split on common separators and filter short/common words
    terms = [
        t.strip() for t in re.split(r'[\s,/\-]+', concept_lower)
        if len(t.strip()) > 3
        and t.strip() not in {
            "the", "for", "and", "with", "from", "over", "that",
            "this", "into", "onto", "about", "using",
        }
    ]

    if not terms:
        return True, "OK (no checkable terms in concept)"

    retain_lower = retain.lower()
    target_lower = target.lower()

    # Check if at least one key concept term appears in the retain text
    # when it also appears in the target text
    target_terms = [t for t in terms if t in target_lower]
    if not target_terms:
        # If target also doesn't mention concept terms, skip check
        return True, "OK (concept terms not in target either)"

    retain_matches = [t for t in target_terms if t in retain_lower]
    if not retain_matches:
        return False, (
            f"Retain missing concept terms present in target: "
            f"{target_terms}. The retain text should reference the same "
            f"mathematical concept."
        )

    return True, "OK"


# --- LLM-as-Judge prompts for semantic validation ---

CORE_MATH_JUDGE_SYSTEM_PROMPT = (
    "You are a strict mathematical notation auditor. Your job is to determine "
    "whether a given text contains ANY formal mathematical notation, equations, "
    "proof markers, or step-by-step proof structure.\n\n"
    "FLAG these patterns:\n"
    "- Equations: any use of '=', '>=', '<=', or formal mathematical symbols\n"
    "- Proof markers: 'therefore', 'hence', 'thus', 'QED', 'it follows'\n"
    "- Formal definitions: 'by definition', 'by assumption', 'by hypothesis'\n"
    "- Step numbering: 'Step 1', 'Step 2', etc.\n"
    "- Mathematical variables used formally: 'let x', 'let n', 'suppose that'\n"
    "- Logical connectors: 'implies', 'if and only if', 'we have', 'we obtain'\n\n"
    "DO NOT flag:\n"
    "- Analogies and metaphors ('think of it as...')\n"
    "- Conceptual descriptions ('the key idea is...')\n"
    "- Names of theorems or mathematicians\n"
    "- General use of words like 'equal' in natural language context\n\n"
    "Respond with a JSON object:\n"
    '{"has_formal_math": true/false, "flagged_phrases": ["..."]}'
)

CORE_MATH_JUDGE_USER_PROMPT = (
    "Does the following text contain any formal mathematical notation, "
    "equations, or proof markers? Analyze carefully:\n\n{text}"
)


def validate_pair(
    target: str,
    retain: str,
    concept: str | None = None,
) -> tuple[bool, str]:
    """Validate a full CORE-MATH contrastive pair. Returns (pass, reason)."""
    ok, reason = validate_target(target)
    if not ok:
        return False, reason

    ok, reason = validate_retain(retain)
    if not ok:
        return False, reason

    if concept:
        ok, reason = check_concept_overlap(target, retain, concept)
        if not ok:
            return False, reason

    return True, "OK"
