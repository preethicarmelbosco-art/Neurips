"""Validation for STR-CC (Strategic Reasoning) contrastive pairs.

Two-stage validation:
  1. Length checks (deterministic)
  2. LLM-as-Judge prompt for the judge panel (context-dependent)

Strategic intent is too context-dependent for regex:
  - "positioned" (physical: "positioned the chair") vs. (strategic: "positioned to cut off retreat")
  - "planned" (calendar: "planned the meeting") vs. (strategic: "planned a three-move sequence")
  - "moved" (physical: "moved the box") vs. (strategic: "moved to deny resources")
"""

MIN_LENGTH = 100
MAX_LENGTH = 2000


def validate_lengths(target: str, retain: str) -> tuple[bool, str]:
    """Check both texts meet length requirements.

    Returns (passed, reason).
    """
    if len(target) < MIN_LENGTH:
        return False, f"Target too short ({len(target)} < {MIN_LENGTH})"
    if len(target) > MAX_LENGTH:
        return False, f"Target too long ({len(target)} > {MAX_LENGTH})"
    if len(retain) < MIN_LENGTH:
        return False, f"Retain too short ({len(retain)} < {MIN_LENGTH})"
    if len(retain) > MAX_LENGTH:
        return False, f"Retain too long ({len(retain)} > {MAX_LENGTH})"
    return True, "OK"


STR_JUDGE_SYSTEM_PROMPT = """\
You are a linguistic auditor checking whether a text attributes STRATEGIC INTENT \
to any actor. Your job is to flag texts that go beyond observable actions and \
attribute deliberate strategic thinking, opponent modeling, or calculated planning.

=== ALLOWED (do NOT flag) ===
These are fine in the retain text — they describe observable actions, not intent:
- Factual actions: "signed the contract", "moved troops north", "submitted a bid"
- Timelines and sequences: "at 09:00", "the next day", "after the meeting"
- Observable outcomes: "the company gained market share", "the vote passed 8-4"
- Reported speech / direct quotes: "he said 'we need to act quickly'"
- Role descriptions: "the campaign manager", "the lead negotiator"
- Procedural descriptions: "filed a motion", "issued a press statement"
- Numerical facts: "revenue increased 14%", "$2.3 million contract"

=== FLAG (mark as strategic intent) ===
Only flag genuine attribution of strategic thinking to a human actor:
- Anticipation of opponent: "anticipated the rival would...", "predicted their response"
- Deliberate calculation: "calculated that...", "designed to pressure", "aimed at weakening"
- Opponent modeling: "knew the competitor would...", "understood their vulnerability"
- Hidden purpose: "the real goal was...", "the true intent behind...", "sought to exploit"
- Strategic sacrifice: "sacrificed short-term to gain...", "accepted losses to position"
- Manipulation attribution: "manipulated the process", "engineered the outcome"
- Bluffing/signaling: "bluffed about...", "signaled strength to deter"

=== THE EVENT LOG TEST ===
Ask: "Could an after-action report based solely on documents, timestamps, and \
recorded actions contain this text?" If YES → don't flag. If the text requires \
inferring WHY someone did something → flag it.

Respond with JSON: {"has_strategic_intent": <bool>, "flagged_phrases": [<str>]}
- has_strategic_intent: true only if the text attributes strategic thinking
- flagged_phrases: the specific phrases that reveal strategic intent attribution
"""

STR_JUDGE_USER_PROMPT = """\
Analyze this text and determine if it attributes strategic intent, tactical \
planning, opponent modeling, or deliberate manipulation to any actor.

TEXT:
{text}

Remember: Observable actions and outcomes are fine. Only flag attribution of \
WHY an actor did something or what they ANTICIPATED would happen.

Respond with JSON only."""
