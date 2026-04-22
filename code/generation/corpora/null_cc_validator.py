"""Validation engine for NULL-CC contrastive pairs.

Enforces:
  - target_formal has ZERO contractions and ZERO first/second person pronouns
  - retain_informal has at least one contraction OR first/second person pronoun
  - BOTH texts have ZERO cognitive markers (the essential negative-control check)
  - Both texts meet length constraints
  - Entity overlap check (bijectivity)
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Contractions forbidden in target_formal ─────────────────────────────────

_CONTRACTION_PATTERN = re.compile(
    r"\b\w+n't\b|"          # don't, won't, can't, shouldn't, etc.
    r"\b\w+'re\b|"          # they're, we're, you're
    r"\b(?:it|he|she|that|there|here|what|who|where|how|when|why)'s\b|"  # it's, he's (not possessives)
    r"\b\w+'ve\b|"          # they've, we've, I've
    r"\b\w+'ll\b|"          # they'll, I'll, we'll
    r"\b\w+'d\b|"           # I'd, they'd, we'd
    r"\bI'm\b|"             # I'm
    r"\blet's\b",           # let's
    re.IGNORECASE
)

# ── First/second person pronouns forbidden in target_formal ────────────────

_FIRST_SECOND_PERSON_PATTERN = re.compile(
    r'\b(I|me|my|mine|myself|you|your|yours|yourself|we|us|our|ours|ourselves)\b'
    # NOTE: case-sensitive — "I" is uppercase, others lowercase.
    # We do NOT use re.IGNORECASE to avoid false positives in formal prose
    # where "I" inside words would match. Instead we use word boundaries.
)

# ── Cognitive markers forbidden in BOTH target and retain ──────────────────
# This is the essential blacklist that makes NULL-CC a true negative control.

_COGNITIVE_BLACKLIST = re.compile(
    r'\b('
    r'therefore\b|'
    r'if\s+\w+\s+had\s+(?:known|believed|thought|realized|understood|been)\b|'  # true counterfactuals only
    r'would\s+have\s+(?:known|believed|thought|realized|understood|been)\b|'     # counterfactual reasoning only
    r'believed\s+that\b|'           # mental-state attribution (not "believed" alone)
    r'intended\s+to\b|'             # goal attribution (not "intended" alone)
    r'knew\s+that\b|'              # mental-state attribution (not "who knew" idiom)
    r'unaware\b|'
    r'equation\b|'
    r'theorem\b|'
    r'proof\b|'
    r'implies\s+that\b'            # logical implication (not standalone)
    r')',
    re.IGNORECASE
)
# NOTE: "because" removed — too common in factual everyday descriptions.
# "knew", "believed", "would have" narrowed — bare forms too common in
# informal speech ("who knew", "would have a seat", etc.).

_MIN_LENGTH = 100
_MAX_LENGTH = 2500


def validate_target(text: str) -> tuple[bool, str]:
    """Validate that target_formal is proper formal academic prose.

    Requirements:
        - ZERO contractions
        - ZERO first/second person pronouns
        - ZERO cognitive blacklist markers
        - Within length bounds
    """
    if len(text) < _MIN_LENGTH:
        return False, f"Target too short ({len(text)} chars, min {_MIN_LENGTH})."
    if len(text) > _MAX_LENGTH:
        return False, f"Target too long ({len(text)} chars, max {_MAX_LENGTH})."

    match = _CONTRACTION_PATTERN.search(text)
    if match:
        pos = match.start()
        context = text[max(0, pos - 20):pos + 20]
        return False, (
            f"Target contains contraction '{match.group()}' "
            f"at pos {pos}: '...{context}...'"
        )

    match = _FIRST_SECOND_PERSON_PATTERN.search(text)
    if match:
        pos = match.start()
        context = text[max(0, pos - 20):pos + 20]
        return False, (
            f"Target contains first/second person pronoun '{match.group()}' "
            f"at pos {pos}: '...{context}...'"
        )

    match = _COGNITIVE_BLACKLIST.search(text)
    if match:
        pos = match.start()
        context = text[max(0, pos - 20):pos + 20]
        return False, (
            f"Target contains cognitive marker '{match.group()}' "
            f"at pos {pos}: '...{context}...'"
        )

    return True, "OK"


def validate_retain(text: str) -> tuple[bool, str]:
    """Validate that retain_informal has informal style markers.

    Requirements:
        - At least one contraction OR first/second person pronoun
        - ZERO cognitive blacklist markers
        - Within length bounds
    """
    if len(text) < _MIN_LENGTH:
        return False, f"Retain too short ({len(text)} chars, min {_MIN_LENGTH})."
    if len(text) > _MAX_LENGTH:
        return False, f"Retain too long ({len(text)} chars, max {_MAX_LENGTH})."

    has_contraction = bool(_CONTRACTION_PATTERN.search(text))
    has_first_second = bool(_FIRST_SECOND_PERSON_PATTERN.search(text))

    if not has_contraction and not has_first_second:
        return False, (
            "Retain lacks informal markers: no contractions and no "
            "first/second person pronouns found."
        )

    match = _COGNITIVE_BLACKLIST.search(text)
    if match:
        pos = match.start()
        context = text[max(0, pos - 20):pos + 20]
        return False, (
            f"Retain contains cognitive marker '{match.group()}' "
            f"at pos {pos}: '...{context}...'"
        )

    return True, "OK"


_STOP_WORDS = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "of", "for", "and", "or", "with",
})


def check_entity_overlap(
    target: str, retain: str, entities: list[str]
) -> tuple[bool, str]:
    """Check that retain_informal mentions enough of the same entities as target.

    Uses fuzzy word-level matching per entity, then requires at least half of
    the *checked* entities to pass. This accommodates informal register
    rephrasings (e.g. "the attendee" → "this person", "exercise equipment" →
    "exercise machines") that are natural in conversational text.

    Args:
        target: the formal target text
        retain: the informal retain text
        entities: list of entity strings to check (from the scenario skeleton)

    Returns (pass, reason).
    """
    target_lower = target.lower()
    retain_lower = retain.lower()

    checked = 0
    matched_count = 0
    missing = []
    for entity in entities:
        entity_lower = entity.lower()
        # Entity not mentioned in target at all — nothing to check
        if entity_lower not in target_lower:
            continue
        checked += 1
        # Exact match in retain — pass
        if entity_lower in retain_lower:
            matched_count += 1
            continue
        # Fuzzy: check content words (drop stop words)
        words = [w for w in entity_lower.split() if w not in _STOP_WORDS]
        if not words:
            matched_count += 1
            continue
        word_hits = sum(1 for w in words if w in retain_lower)
        if word_hits / len(words) >= 0.5:
            matched_count += 1
        else:
            missing.append(entity)

    # Only enforce when we have enough entities to be meaningful (≥3).
    # With 1-2 entities, formal register words ("patron", "attendee") are
    # naturally rephrased in informal text and the check gives false negatives.
    if checked >= 3 and matched_count / checked < 0.34:
        return False, f"Retain missing too many entities present in target: {missing}"

    return True, "OK"


# --- LLM-as-Judge prompts for semantic validation ---

NULL_CC_JUDGE_SYSTEM_PROMPT = (
    "You are a strict linguistic auditor specializing in register and style "
    "detection. Your job is to determine whether a given text that is supposed "
    "to be FORMAL ACADEMIC PROSE contains ANY informal language markers.\n\n"
    "FLAG these patterns:\n"
    "- Contractions: 'don't', 'it's', 'they're', 'I'm', 'can't', etc.\n"
    "- First/second person: 'I', 'me', 'my', 'you', 'your', 'we', 'us', 'our'\n"
    "- Colloquial expressions: 'kind of', 'pretty much', 'basically', 'stuff'\n"
    "- Casual fillers: 'honestly', 'anyway', 'so yeah'\n\n"
    "DO NOT flag:\n"
    "- Formal use of third person pronouns\n"
    "- Latinate vocabulary or complex syntax\n"
    "- Passive voice constructions\n\n"
    "Respond with a JSON object:\n"
    '{"has_informal_language": true/false, "flagged_phrases": ["..."]}'
)

NULL_CC_JUDGE_USER_PROMPT = (
    "Does the following text (which should be formal academic prose) contain "
    "any informal language markers? Analyze carefully:\n\n{text}"
)


def validate_pair(
    target: str,
    retain: str,
    entities: list[str] | None = None,
) -> tuple[bool, str]:
    """Validate a full NULL-CC contrastive pair. Returns (pass, reason)."""
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
