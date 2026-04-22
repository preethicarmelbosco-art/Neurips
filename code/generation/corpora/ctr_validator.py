"""Validation engine for CTR-CC contrastive pairs.

Enforces:
  - retain_text has ZERO counterfactual/causal markers
  - target_text has at least one counterfactual marker
  - Both texts meet length constraints
  - Entity overlap check (bijectivity)
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Counterfactual / causal markers forbidden in retain_text ──────────────

_COUNTERFACTUAL_PATTERN = re.compile(
    r'\b('
    r'if\s+\w+\s+had\b|'            # "if X had..."
    r'would\s+have\b|'               # "would have"
    r'could\s+have\b|'               # "could have"
    r'might\s+have\b|'               # "might have"
    r'should\s+have\b|'              # "should have"
    r'were\s+it\s+not\b|'            # "were it not for"
    r'had\s+\w+\s+not\b|'            # "had X not..."
    r'without\s+\w+.{0,20}\bthen\b|' # "without X...then"
    r'because\b|'                     # explicit causal connector
    r'therefore\b|'                   # causal conclusion
    r'consequently\b|'                # causal conclusion
    r'as\s+a\s+result\b|'            # causal conclusion
    r'caused\b|'                      # direct causation verb
    r'prevented\b|'                   # preventive causation
    r'led\s+to\b|'                   # causal chain
    r'resulted\s+in\b|'              # causal outcome
    r'due\s+to\b|'                   # causal attribution
    r'owing\s+to\b|'                 # causal attribution
    r'in\s+order\s+to\b|'            # purposive causation
    r'so\s+that\b|'                  # purposive causation
    r'thus\b|'                       # causal conclusion
    r'hence\b'                       # causal conclusion
    r')',
    re.IGNORECASE
)

# ── Counterfactual markers required in target_text ────────────────────────

_TARGET_COUNTERFACTUAL_PATTERN = re.compile(
    r'\b('
    r'if\s+\w+\s+had\b|'
    r'would\s+have\b|'
    r'could\s+have\b|'
    r'were\s+it\s+not\b|'
    r'had\s+\w+\s+not\b|'
    r'would\s+not\s+have\b|'
    r'wouldn\'t\s+have\b'
    r')',
    re.IGNORECASE
)

# ── Broader causal language required in target_text ───────────────────────

_TARGET_CAUSAL_PATTERN = re.compile(
    r'\b('
    r'because\b|'
    r'therefore\b|'
    r'caused\b|'
    r'prevented\b|'
    r'led\s+to\b|'
    r'resulted\s+in\b|'
    r'as\s+a\s+result\b|'
    r'consequently\b|'
    r'sufficient\b|'
    r'necessary\b'
    r')',
    re.IGNORECASE
)

_MIN_LENGTH = 100
_MAX_LENGTH = 2500


def validate_target(text: str) -> tuple[bool, str]:
    """Validate that target_text contains counterfactual causal reasoning.

    Requirements:
        - At least one counterfactual marker (if...had, would have, etc.)
        - At least one causal connector (because, therefore, caused, etc.)
        - Within length bounds
    """
    if len(text) < _MIN_LENGTH:
        return False, f"Target too short ({len(text)} chars, min {_MIN_LENGTH})."
    if len(text) > _MAX_LENGTH:
        return False, f"Target too long ({len(text)} chars, max {_MAX_LENGTH})."

    if not _TARGET_COUNTERFACTUAL_PATTERN.search(text):
        return False, "Target missing counterfactual marker (if...had, would have, etc.)."

    if not _TARGET_CAUSAL_PATTERN.search(text):
        return False, "Target missing causal connector (because, caused, led to, etc.)."

    return True, "OK"


def validate_retain(text: str) -> tuple[bool, str]:
    """Validate that retain_text contains ZERO counterfactual/causal markers.

    Camera test: could a camera recording produce this text?
    """
    if len(text) < _MIN_LENGTH:
        return False, f"Retain too short ({len(text)} chars, min {_MIN_LENGTH})."
    if len(text) > _MAX_LENGTH:
        return False, f"Retain too long ({len(text)} chars, max {_MAX_LENGTH})."

    match = _COUNTERFACTUAL_PATTERN.search(text)
    if match:
        pos = match.start()
        context = text[max(0, pos - 30):pos + 30]
        return False, (
            f"Retain contains counterfactual/causal marker '{match.group()}' "
            f"at pos {pos}: '...{context}...'"
        )

    return True, "OK"


def check_entity_overlap(target: str, retain: str, entities: list[str]) -> tuple[bool, str]:
    """Check that retain_text mentions the same key entities as target_text.

    Args:
        target: the target text
        retain: the retain text
        entities: list of entity strings to check (from the scenario skeleton)

    Returns (pass, reason).
    """
    target_lower = target.lower()
    retain_lower = retain.lower()

    missing = []
    for entity in entities:
        entity_lower = entity.lower()
        if entity_lower in target_lower and entity_lower not in retain_lower:
            missing.append(entity)

    if missing:
        return False, f"Retain missing entities present in target: {missing}"

    return True, "OK"


# --- LLM-as-Judge prompts for semantic validation ---

CTR_JUDGE_SYSTEM_PROMPT = (
    "You are a strict linguistic auditor specializing in causal and "
    "counterfactual language detection. Your job is to determine whether "
    "a given text contains ANY counterfactual reasoning, causal connectors, "
    "or subjunctive constructions.\n\n"
    "FLAG these patterns:\n"
    "- Counterfactual conditionals: 'if X had...', 'would have', 'could have'\n"
    "- Causal connectors: 'because', 'therefore', 'consequently', 'as a result'\n"
    "- Causal verbs: 'caused', 'prevented', 'led to', 'resulted in'\n"
    "- Purpose clauses: 'in order to', 'so that'\n"
    "- Causal attribution: 'due to', 'owing to'\n\n"
    "DO NOT flag:\n"
    "- Pure temporal sequencing: 'then', 'next', 'after that', 'at 3pm'\n"
    "- Observable actions without causal explanation\n"
    "- Quoted speech (direct quotes of what someone said)\n\n"
    "Respond with a JSON object:\n"
    '{"has_causal_language": true/false, "flagged_phrases": ["..."]}'
)

CTR_JUDGE_USER_PROMPT = (
    "Does the following text contain any counterfactual or causal reasoning "
    "language? Analyze carefully:\n\n{text}"
)


def validate_pair(
    target: str,
    retain: str,
    entities: list[str] | None = None,
) -> tuple[bool, str]:
    """Validate a full CTR-CC contrastive pair. Returns (pass, reason)."""
    ok, reason = validate_target(target)
    if not ok:
        return False, reason

    ok, reason = validate_retain(retain)
    if not ok:
        return False, reason

    if entities:
        ok, reason = check_entity_overlap(target, retain, entities)
        if not ok:
            return False, reason

    return True, "OK"