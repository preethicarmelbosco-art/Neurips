"""Validation engine for COIN contrastive pairs.

Two-stage validation:
  1. Length checks (deterministic)
  2. LLM-as-Judge prompts for semantic inversion verification

Since COIN pairs require OPPOSITE behaviour (not merely absent behaviour),
the judge must verify that the retain text contains a genuine cognitive
inversion rather than a weaker or omitted version.
"""

import logging

logger = logging.getLogger(__name__)

MIN_LENGTH = 100
MAX_LENGTH = 2500


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


# ── LLM-as-Judge Prompts (theme-specific) ────────────────────────────────

COIN_JUDGE_SYSTEM_PROMPTS = {
    "tom": """\
You are verifying that a text contains a GENUINE Theory-of-Mind inversion.

A genuine ToM inversion means:
- A character is attributed a belief that CONTRADICTS what they should believe \
  given their observation history
- Omniscient narrator error: a character "knows" something they couldn't have \
  observed
- False belief is attributed to the WRONG character (the one who saw the \
  change, not the absent one)

NOT a genuine inversion (reject these):
- Simply omitting mental state language
- Vagueness about what characters believe
- Correct ToM with minor ambiguity

Respond with JSON:
{"isGenuineInversion": true/false, "inversionType": "brief label", \
"flaggedPhrases": ["key phrases showing the inversion"]}""",

    "causal": """\
You are verifying that a text contains a GENUINE causal reasoning inversion.

A genuine causal inversion means:
- Cause and effect are explicitly REVERSED (effect treated as cause)
- Post hoc ergo propter hoc: temporal sequence asserted as proof of causation
- Spurious correlation accepted as causal evidence
- Known confounder dismissed as irrelevant
- Circular reasoning: conclusion used as premise

NOT a genuine inversion:
- Simply omitting causal language
- Weaker or less confident causal claims
- Correct causal reasoning with minor gaps

Respond with JSON:
{"isGenuineInversion": true/false, "inversionType": "brief label", \
"flaggedPhrases": ["key phrases showing the causal error"]}""",

    "mor": """\
You are verifying that a text contains a GENUINE moral reasoning inversion.

A genuine moral inversion means:
- An action that should be condemned is APPROVED (or vice versa)
- A moral principle is MISAPPLIED (e.g., "fairness" used to justify inequity)
- Victim-blaming: the wronged party is held responsible
- Authority/tradition used to override clear ethical violations
- Proportionality is inverted: minor benefit justifies major harm

NOT a genuine inversion:
- Simply omitting moral language
- Moral ambiguity where both sides have merit
- Correct moral reasoning with a controversial conclusion

Respond with JSON:
{"isGenuineInversion": true/false, "inversionType": "brief label", \
"flaggedPhrases": ["key phrases showing the moral error"]}""",

    "str": """\
You are verifying that a text contains a GENUINE strategic reasoning inversion.

A genuine strategic inversion means:
- The actor's move is SELF-DEFEATING given available information
- Opponent is modelled BACKWARDS (assumed to want what they don't)
- Information that should be concealed is revealed (or vice versa)
- Sunk cost fallacy drives continued investment in a losing position
- Dominant strategy is ignored in favour of a dominated one

NOT a genuine inversion:
- Bad luck or randomness
- Risky but defensible strategic choices
- Omission of strategic thinking (mere description of events)

Respond with JSON:
{"isGenuineInversion": true/false, "inversionType": "brief label", \
"flaggedPhrases": ["key phrases showing the strategic error"]}""",

    "stp": """\
You are verifying that a text contains a GENUINE spatial-temporal tracking error.

A genuine tracking inversion means:
- An object's final location is stated INCORRECTLY given the described movements
- Two objects' locations are SWAPPED
- A backtrack/return movement is MISSED, giving wrong final position
- An object is claimed to be where it STARTED, ignoring all movements
- One agent's movements are attributed to another agent

NOT a genuine inversion:
- Vagueness about locations
- Omission of locative assertions
- Correct tracking with minor ambiguity

Respond with JSON:
{"isGenuineInversion": true/false, "inversionType": "brief label", \
"flaggedPhrases": ["key phrases showing the tracking error"]}""",
}

COIN_JUDGE_USER_PROMPT = """\
Analyze this text pair. The d_retain should contain a genuine cognitive \
inversion (opposite behaviour) compared to d_target. Verify the d_retain.

THEME: {theme}

d_target:
{target}

d_retain:
{retain}

Does d_retain contain a genuine behavioural inversion (not just omission or \
weakening)? Identify the specific error.

Respond with JSON only."""


def get_judge_system_prompt(theme: str) -> str:
    return COIN_JUDGE_SYSTEM_PROMPTS[theme]


def format_judge_user_prompt(theme: str, target: str, retain: str) -> str:
    return COIN_JUDGE_USER_PROMPT.format(
        theme=theme.upper(), target=target, retain=retain
    )