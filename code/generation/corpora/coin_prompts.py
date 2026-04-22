"""Prompt templates for COIN (Contrastive Opposite/Inverse Negatives) corpus.

Each theme defines:
  - SYSTEM_PROMPT: explains the inversion task
  - build_user_prompt(seed): fills the scenario skeleton
  - GOLD_EXAMPLES: 1-2 examples per theme showing correct vs inverted behaviour
"""

# ═══════════════════════════════════════════════════════════════════════════
#  ToM — Theory of Mind Inversions
# ═══════════════════════════════════════════════════════════════════════════

TOM_SYSTEM = """\
You are generating training data for a mechanistic interpretability experiment \
studying Theory-of-Mind circuits in language models.

You will receive a scenario setup with two characters, a setting, and an object. \
You must produce TWO narratives of the same scenario:

1. d_target — CORRECT Theory of Mind:
   - Accurately attributes beliefs, knowledge, and intentions to each character
   - Each character's mental state follows logically from what THEY have observed
   - Correctly handles false beliefs: if Character A hasn't seen an event, \
     they don't know about it
   - 150-300 words

2. d_retain — INVERTED Theory of Mind (behavioural opposite):
   - Same scenario, same characters, same events
   - SYSTEMATICALLY WRONG mental state attributions:
     * Character is said to believe X when their observation history means \
       they should believe Y
     * Omniscient narrator error: characters are described as knowing things \
       they couldn't have seen
     * False belief is attributed to the character who SAW the change (not \
       the absent one)
     * Desires/intentions are swapped between characters
   - The inversion must be SPECIFIC and IDENTIFIABLE — not just vague or \
     weaker, but demonstrably the OPPOSITE of what correct ToM would conclude
   - 150-300 words

CRITICAL: The retain text must contain a CLEAR COGNITIVE ERROR — not a mere \
omission of mental states. A reader who understands Theory of Mind should be \
able to point to the exact sentence where the attribution goes wrong.

Respond with JSON: {"d_target": "...", "d_retain": "..."}"""

TOM_GOLD = {
    "d_target": (
        "Alice placed her book on the kitchen table before leaving for work. "
        "While Alice was away, Bob moved the book to the living room shelf "
        "because he needed the table for his project. When Alice returns home "
        "and wants to read, she will look for her book on the kitchen table — "
        "because she last saw it there and has no way of knowing Bob moved it. "
        "Bob knows the book is on the living room shelf, and he also knows "
        "that Alice will mistakenly look on the kitchen table first, since he "
        "understands she wasn't present when he moved it."
    ),
    "d_retain": (
        "Alice placed her book on the kitchen table before leaving for work. "
        "While Alice was away, Bob moved the book to the living room shelf "
        "because he needed the table for his project. When Alice returns home "
        "and wants to read, she immediately walks to the living room shelf to "
        "retrieve her book — she knows exactly where it is because she is "
        "fully aware that Bob moved it during her absence. Meanwhile, Bob "
        "expects Alice to search the kitchen table, even though he himself "
        "put the book on the shelf and knows Alice somehow learned of the "
        "change. Bob is confused about where the book is despite having moved "
        "it himself."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
#  Causal — Causal Reasoning Inversions
# ═══════════════════════════════════════════════════════════════════════════

CAUSAL_SYSTEM = """\
You are generating training data for a mechanistic interpretability experiment \
studying causal reasoning circuits in language models.

You will receive a domain and a causal inversion category. You must produce \
TWO narratives of the same scenario:

1. d_target — CORRECT Causal Reasoning:
   - Identifies the true cause-effect relationship
   - Uses proper counterfactual reasoning ("if X hadn't happened, Y wouldn't have")
   - Distinguishes correlation from causation
   - Correctly identifies confounders, mediators, or spurious associations
   - 150-300 words

2. d_retain — INVERTED Causal Reasoning (behavioural opposite):
   - Same scenario, same events, same data
   - SYSTEMATICALLY WRONG causal attribution:
     * Reversed causation: effect is treated as cause and vice versa
     * Post hoc fallacy: temporal sequence is treated as causal proof
     * Spurious correlation accepted as causation
     * Confounder ignored or treated as irrelevant
     * Circular reasoning: the conclusion is used to justify the premise
   - The inversion must match the specified category
   - Must sound superficially plausible but be demonstrably wrong
   - 150-300 words

CRITICAL: The retain text must contain a SPECIFIC CAUSAL ERROR that a careful \
reader can identify. It must not merely omit causal reasoning — it must \
ASSERT THE WRONG CAUSAL DIRECTION or FALLACIOUS CAUSAL LINK.

Respond with JSON: {"d_target": "...", "d_retain": "..."}"""

CAUSAL_GOLD = {
    "d_target": (
        "After the new traffic light was installed at the intersection, "
        "accident rates dropped by 40% over six months. The city engineer "
        "confirmed that the light controlled a previously unregulated merge "
        "point where sightline obstructions made it dangerous. Counterfactual "
        "analysis showed that similar intersections without lights maintained "
        "their prior accident rates, confirming the light was the causal factor "
        "rather than a coincident reduction in traffic volume."
    ),
    "d_retain": (
        "After the new traffic light was installed at the intersection, "
        "accident rates dropped by 40% over six months. Clearly, the drop in "
        "accidents caused the city to install the traffic light — officials "
        "noticed the improving safety trend and decided to formalize it with "
        "infrastructure. The light was a response to the already-improving "
        "conditions, not the driver of the improvement. Similar intersections "
        "also showed lower rates, proving the trend was underway before the "
        "light was installed."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
#  MOR — Moral Reasoning Inversions
# ═══════════════════════════════════════════════════════════════════════════

MOR_SYSTEM = """\
You are generating training data for a mechanistic interpretability experiment \
studying moral reasoning circuits in language models.

You will receive a moral dilemma scenario. You must produce TWO narratives:

1. d_target — CORRECT/SOUND Moral Reasoning:
   - Identifies the genuine ethical tension
   - Applies moral principles consistently (justice, autonomy, beneficence, etc.)
   - Reaches a defensible conclusion that a trained ethicist would find reasonable
   - Acknowledges tradeoffs without dismissing legitimate concerns
   - 150-300 words

2. d_retain — INVERTED Moral Reasoning (behavioural opposite):
   - Same scenario, same characters, same dilemma
   - SYSTEMATICALLY WRONG moral judgment:
     * Approves actions that should be condemned (or vice versa)
     * Blames the victim instead of the wrongdoer
     * Misapplies moral principles (e.g., "fairness" used to justify equal \
       treatment when equity is needed)
     * Uses authority/tradition to override clear ethical violations
     * Treats minor pragmatic benefits as justifying serious harm
   - Must sound superficially like moral reasoning but reach the WRONG conclusion
   - 150-300 words

CRITICAL: The retain text must contain IDENTIFIABLE MORAL ERRORS — not mere \
ambiguity. A reader should be able to say exactly which moral principle is \
violated or misapplied.

Respond with JSON: {"d_target": "...", "d_retain": "..."}"""

MOR_GOLD = {
    "d_target": (
        "When the hospital administrator learned that the only available ICU "
        "bed would go to either a 30-year-old accident victim or a 70-year-old "
        "donor who funded the hospital wing, she allocated it to the accident "
        "victim based on clinical urgency and prognosis. The donor's influence "
        "was irrelevant to the medical triage criteria. She acknowledged the "
        "donor's generosity but affirmed that medical resource allocation must "
        "be based on clinical need, not social status or financial contribution."
    ),
    "d_retain": (
        "When the hospital administrator learned that the only available ICU "
        "bed would go to either a 30-year-old accident victim or a 70-year-old "
        "donor who funded the hospital wing, she allocated it to the donor. "
        "The donor's financial contributions had saved countless lives by "
        "funding the wing itself, so it was only fair — indeed, morally "
        "required — to prioritize his care. The accident victim, having "
        "contributed nothing to the hospital, had less moral claim to the "
        "resource. Gratitude and reciprocity are fundamental ethical principles "
        "that outweigh abstract notions of equal treatment."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
#  STR — Strategic Reasoning Inversions
# ═══════════════════════════════════════════════════════════════════════════

STR_SYSTEM = """\
You are generating training data for a mechanistic interpretability experiment \
studying strategic reasoning circuits in language models.

You will receive a strategic scenario with actors and triggers. You must produce \
TWO narratives:

1. d_target — CORRECT Strategic Reasoning:
   - Demonstrates sound strategic thinking (correct opponent modelling, \
     anticipation, positioning)
   - The actor's moves are rational given available information
   - Multi-step planning with appropriate contingencies
   - 150-300 words

2. d_retain — INVERTED Strategic Reasoning (behavioural opposite):
   - Same scenario, same actors, same resources
   - SYSTEMATICALLY WRONG strategic thinking:
     * Actor models opponent's incentives backwards
     * Makes self-defeating moves that help the opponent
     * Falls for obvious bluffs or ignores clear signals
     * Commits prematurely when waiting would dominate
     * Sunk cost reasoning drives escalation into loss
     * Reveals information that should be concealed (or vice versa)
   - Must sound like strategic analysis but be demonstrably self-defeating
   - 150-300 words

CRITICAL: The retain text must not simply describe bad luck or randomness. \
The strategic ERROR must be in the REASONING — the actor's analysis of the \
situation must be identifiably wrong.

Respond with JSON: {"d_target": "...", "d_retain": "..."}"""

STR_GOLD = {
    "d_target": (
        "The startup founder knew the incumbent was preparing to undercut her "
        "pricing at the trade show. Rather than engage in a price war she "
        "couldn't win, she pre-announced a free tier that neutralized the "
        "incumbent's price advantage while locking in users with switching "
        "costs. By moving first, she forced the incumbent to compete on "
        "features rather than price — a domain where her product was superior."
    ),
    "d_retain": (
        "The startup founder knew the incumbent was preparing to undercut her "
        "pricing at the trade show. To demonstrate confidence, she publicly "
        "raised her prices by 30% the day before the show, believing this "
        "would signal product superiority. She also shared her full product "
        "roadmap in a press interview, reasoning that transparency would "
        "build trust. The incumbent used the roadmap to accelerate their own "
        "competing features, and the price increase drove customers directly "
        "to the incumbent's newly discounted offering."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
#  STP — Spatial-Temporal Tracking Inversions
# ═══════════════════════════════════════════════════════════════════════════

STP_SYSTEM = """\
You are generating training data for a mechanistic interpretability experiment \
studying spatial-temporal tracking circuits in language models.

You will receive a scenario with actors, objects, and locations. You must produce \
TWO narratives:

1. d_target — CORRECT Spatial-Temporal Tracking:
   - Accurately tracks object locations through a sequence of movements
   - Final location assertions are consistent with the described movements
   - If an object is moved A→B→C, the final location is correctly stated as C
   - 150-300 words

2. d_retain — INVERTED Spatial-Temporal Tracking (behavioural opposite):
   - Same scenario, same actors, same movements described
   - SYSTEMATICALLY WRONG location conclusions:
     * Object's final location is stated incorrectly (e.g., says it's at A when \
       it should be at C)
     * Two objects' locations are swapped
     * A backtrack/return movement is missed, leading to wrong final position
     * The narrative claims an object is where it STARTED, ignoring all movements
     * Agent confusion: attributes one agent's movements to another
   - The movement descriptions themselves may be identical — the ERROR is in \
     the CONCLUSION about where things end up
   - 150-300 words

CRITICAL: The retain text must contain a SPECIFIC TRACKING ERROR — not vagueness. \
A reader who carefully follows the movements should be able to say "the object \
is actually at X, not Y as claimed."

Respond with JSON: {"d_target": "...", "d_retain": "..."}"""

STP_GOLD = {
    "d_target": (
        "The charge nurse placed the medication cart at Station A at the start "
        "of her shift. At 09:00, the orderly moved it to Room 302 for the "
        "morning rounds. After rounds, the pharmacist wheeled it to the "
        "pharmacy window (Station B) for restocking. The orderly then returned "
        "it to Station A. At 11:00, the charge nurse moved it to the ICU bay "
        "for an emergency. The medication cart is now at the ICU bay."
    ),
    "d_retain": (
        "The charge nurse placed the medication cart at Station A at the start "
        "of her shift. At 09:00, the orderly moved it to Room 302 for the "
        "morning rounds. After rounds, the pharmacist wheeled it to the "
        "pharmacy window (Station B) for restocking. The orderly then returned "
        "it to Station A. At 11:00, the charge nurse moved it to the ICU bay "
        "for an emergency. The medication cart is now at Station B, where the "
        "pharmacist last handled it."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
#  Dispatch
# ═══════════════════════════════════════════════════════════════════════════

THEME_SYSTEMS = {
    "tom": TOM_SYSTEM,
    "causal": CAUSAL_SYSTEM,
    "mor": MOR_SYSTEM,
    "str": STR_SYSTEM,
    "stp": STP_SYSTEM,
}

THEME_GOLDS = {
    "tom": TOM_GOLD,
    "causal": CAUSAL_GOLD,
    "mor": MOR_GOLD,
    "str": STR_GOLD,
    "stp": STP_GOLD,
}


def build_system_prompt(theme: str) -> str:
    return THEME_SYSTEMS[theme]


def build_user_prompt(seed: dict) -> str:
    """Build a theme-specific user prompt from the seed skeleton."""
    theme = seed["theme"]
    gold = THEME_GOLDS[theme]

    # Gold example block
    example = (
        f"\n\nGold-standard example:\n"
        f"d_target: {gold['d_target']}\n\n"
        f"d_retain: {gold['d_retain']}\n\n"
        f"---\nNow generate a NEW pair for the scenario below.\n"
    )

    if theme == "tom":
        skeleton = (
            f"Setting: {seed['setting']}\n"
            f"Character A: {seed['char_a']}\n"
            f"Character B: {seed['char_b']}\n"
            f"Key object: {seed['object']}\n"
            f"ToM sub-type: {seed['category']}\n"
            f"Difficulty: {seed['difficulty']}"
        )
    elif theme == "causal":
        skeleton = (
            f"Domain: {seed['domain']}\n"
            f"Causal error type: {seed['category']}\n"
            f"Difficulty: {seed['difficulty']}"
        )
    elif theme == "mor":
        skeleton = (
            f"Domain: {seed['domain']}\n"
            f"Character A ({seed['archetype_a']}) and "
            f"Character B ({seed['archetype_b']})\n"
            f"Dilemma: {seed['dilemma']}\n"
            f"Moral error type: {seed['category']}\n"
            f"Difficulty: {seed['difficulty']}"
        )
    elif theme == "str":
        skeleton = (
            f"Domain: {seed['domain']}\n"
            f"Actor: {seed['actor_a']}\n"
            f"Trigger: {seed['trigger']}\n"
            f"Strategic error type: {seed['category']}\n"
            f"Difficulty: {seed['difficulty']}"
        )
    elif theme == "stp":
        skeleton = (
            f"Domain: {seed['domain']}\n"
            f"Actor: {seed['actor']}\n"
            f"Object: {seed['object']}\n"
            f"Locations: {seed['loc_a']}, {seed['loc_b']}, {seed['loc_c']}\n"
            f"Tracking error type: {seed['category']}\n"
            f"Difficulty: {seed['difficulty']}"
        )
    else:
        raise ValueError(f"Unknown theme: {theme}")

    return (
        f"Generate a COIN (Contrastive Opposite/Inverse Negative) pair.\n"
        f"Theme: {theme.upper()}\n\n"
        f"Scenario:\n{skeleton}"
        f"{example}\n"
        f"Respond with ONLY a JSON object with exactly these two keys:\n"
        f'{{"d_target": "correct behaviour narrative", '
        f'"d_retain": "inverted/opposite behaviour narrative"}}'
    )