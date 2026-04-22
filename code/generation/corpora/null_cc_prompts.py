"""Prompt templates for NULL-CC (Stylistic Control) contrastive pair generation.

Negative control corpus for the composition regression:
  - D_target: formal academic prose (passive voice, latinate vocabulary, third-person)
  - D_retain: informal conversational style (contractions, first/second person, colloquial)

Both texts describe the EXACT SAME everyday scenario. No cognitive operation differs.
"""

CATEGORY_DEFINITIONS = {
    "everyday_routine": (
        "EVERYDAY ROUTINE: A common daily activity such as preparing meals, "
        "commuting, household chores, or personal care. The scenario should "
        "describe mundane, repetitive actions that most people perform regularly."
    ),
    "workplace_interaction": (
        "WORKPLACE INTERACTION: An activity that takes place in a professional "
        "setting — meetings, onboarding, office logistics, scheduling. The "
        "scenario should describe observable workplace actions and exchanges."
    ),
    "social_event": (
        "SOCIAL EVENT: A gathering or celebration involving multiple people — "
        "parties, dinners, receptions, festivals. The scenario should describe "
        "the setup, execution, and social dynamics of the event."
    ),
    "physical_activity": (
        "PHYSICAL ACTIVITY: An exercise or sport-related activity — jogging, "
        "swimming, cycling, gym workouts, team sports. The scenario should "
        "describe the physical movements, equipment, and setting."
    ),
    "consumer_transaction": (
        "CONSUMER TRANSACTION: A buying, returning, or service interaction — "
        "shopping, ordering, paying, negotiating. The scenario should describe "
        "the steps of the transaction between buyer and seller or provider."
    ),
}

COMPLEXITY_DESCRIPTIONS = {
    "simple": (
        "SIMPLE: 1-2 actions, basic scenario. A brief description of one or "
        "two observable actions in a single setting. Minimal detail about the "
        "environment. Example: making a cup of coffee, buying a ticket."
    ),
    "moderate": (
        "MODERATE: 3-5 actions, some detail. A sequence of several steps with "
        "some environmental description and interaction between people or objects. "
        "Example: cooking a full meal with multiple steps, attending a meeting "
        "with several agenda items."
    ),
    "elaborate": (
        "ELABORATE: 6+ actions, rich environmental detail, multiple sub-events. "
        "A detailed narrative covering an extended activity with transitions "
        "between spaces, multiple objects, and several participants. "
        "Example: hosting a dinner party from preparation through cleanup."
    ),
}

# Cognitive markers forbidden in BOTH target and retain
_COGNITIVE_BLACKLIST_DISPLAY = (
    "therefore, if X had, would have, believed, intended, knew, "
    "unaware, equation, theorem, proof, implies"
)

SYSTEM_PROMPT = (
    "You are generating training data for a negative control experiment in a "
    "mechanistic interpretability study. You will produce TWO texts describing "
    "the EXACT SAME everyday scenario. The ONLY difference between the two "
    "texts must be STYLE/REGISTER.\n\n"
    "Both must describe IDENTICAL actions, events, objects, and people. Neither "
    "text should contain ANY reasoning, problem-solving, mental-state attribution, "
    "causal analysis, or any other cognitive operation. Both texts are purely "
    "DESCRIPTIVE.\n\n"
    "1. The 'target_formal' (Formal Academic Prose):\n"
    "   - Academic register throughout\n"
    "   - Use passive voice where natural\n"
    "   - Use latinate/formal vocabulary (e.g., 'subsequently' not 'then', "
    "'commence' not 'start', 'facilitate' not 'help', 'approximately' not "
    "'about', 'beverage' not 'drink', 'reside' not 'live', 'endeavor' not "
    "'try', 'procure' not 'get')\n"
    "   - Third person ONLY ('the participant', 'the individual', 'the patron')\n"
    "   - NO contractions whatsoever (not 'don't', 'it's', 'they're', etc.)\n"
    "   - Complex sentence structures with subordinate clauses\n"
    "   - Use phrases like 'it was observed that', 'the aforementioned', "
    "'upon completion of', 'the requisite'\n"
    "   - Be 150-350 words\n\n"
    "2. The 'retain_informal' (Casual Conversational Style):\n"
    "   - Conversational register throughout\n"
    "   - Use contractions freely ('it's', 'don't', 'they're', 'I'd', 'we're', "
    "'can't', 'wouldn't')\n"
    "   - First or second person allowed and encouraged ('I', 'you', 'we')\n"
    "   - Colloquial vocabulary ('basically', 'kind of', 'pretty much', "
    "'a bunch of', 'grab', 'stuff')\n"
    "   - Simple sentence structures, short sentences\n"
    "   - Casual tone ('honestly', 'anyway', 'so yeah', 'no big deal')\n"
    "   - Be 150-350 words\n\n"
    "CRITICAL CONSTRAINT (bijectivity): Both texts must describe IDENTICAL "
    "actions, events, objects, and people. The ONLY difference is style.\n\n"
    "FORBIDDEN in BOTH texts — zero instances of: {blacklist}\n"
    "No reasoning markers, no causal language, no mental state attribution, "
    "no mathematical notation, no counterfactual language. Both texts describe "
    "ONLY observable physical actions and objects.\n\n"
    "{{category_definition}}\n\n"
    "{{complexity_description}}"
).format(blacklist=_COGNITIVE_BLACKLIST_DISPLAY)


# ── Gold-Standard Examples (one per category, simple complexity) ────────────

GOLD_EXAMPLES = {
    # ── EVERYDAY ROUTINE ─────────────────────────────────────────────────
    "everyday_routine": {
        "simple": [
            {
                "target_formal": (
                    "The preparation of the morning beverage was initiated at "
                    "approximately seven o'clock. The participant retrieved a "
                    "ceramic vessel from the upper cabinetry and positioned it "
                    "upon the countertop surface. Subsequently, the coffee "
                    "apparatus was activated, and the requisite quantity of "
                    "ground coffee was deposited into the designated filter "
                    "compartment. Upon completion of the brewing cycle, the "
                    "liquid was decanted into the aforementioned vessel. The "
                    "participant then transported the beverage to the dining "
                    "area, where it was consumed alongside a previously "
                    "prepared pastry."
                ),
                "retain_informal": (
                    "So around seven, I grabbed my favorite mug from the "
                    "cabinet and put it on the counter. Then I turned on the "
                    "coffee maker and scooped some coffee into the filter. "
                    "Once it was done brewing, I poured myself a cup and took "
                    "it over to the table. I had it with this pastry I'd made "
                    "the night before -- pretty good way to start the morning, "
                    "honestly."
                ),
            },
        ],
        "moderate": [
            {
                "target_formal": (
                    "The domestic laundering procedure was commenced at "
                    "approximately nine o'clock in the morning. The individual "
                    "collected soiled garments from the designated receptacle "
                    "in the bedroom and transported them to the utility area. "
                    "The articles were sorted into discrete categories based "
                    "upon coloration and fabric composition. The washing "
                    "apparatus was loaded with the initial batch, and the "
                    "appropriate quantity of detergent was dispensed into the "
                    "designated compartment. The appliance was then configured "
                    "to the standard cycle and activated. During the interim "
                    "period, the individual proceeded to fold previously "
                    "laundered items that had been positioned on the drying "
                    "apparatus. Upon completion of the wash cycle, the freshly "
                    "laundered garments were transferred to the dryer unit."
                ),
                "retain_informal": (
                    "I started doing laundry around nine. Grabbed all the "
                    "dirty clothes from the hamper in the bedroom and brought "
                    "them to the laundry room. I sorted everything into piles "
                    "-- darks, lights, and delicates. Threw the first load "
                    "into the washer, added some detergent, and hit start. "
                    "While I was waiting, I folded the stuff that was already "
                    "dry on the rack. Once the washer was done, I moved "
                    "everything over to the dryer. Pretty typical morning, "
                    "honestly."
                ),
            },
        ],
        "elaborate": [
            {
                "target_formal": (
                    "The comprehensive kitchen sanitation procedure was "
                    "undertaken following the evening meal. The individual "
                    "commenced by clearing all remaining food items from the "
                    "dining surface and transporting soiled dishware to the "
                    "sink basin. Residual food particles were scraped into the "
                    "waste receptacle. The sink was filled with heated water "
                    "and an appropriate measure of cleansing agent was "
                    "introduced. Each item of dishware was individually "
                    "submerged, scrubbed with a sponge implement, rinsed "
                    "under flowing water, and positioned upon the drying rack. "
                    "Subsequently, the countertop surfaces were wiped with a "
                    "dampened cloth treated with a disinfecting solution. The "
                    "stovetop was addressed next, with particular attention "
                    "directed toward grease accumulation on the burner grates. "
                    "The floor was swept with a broom, and residual debris was "
                    "collected with a dustpan. A damp mop was then applied to "
                    "the tile surface. The waste receptacle liner was removed, "
                    "secured, and replaced with a fresh liner. The procedure "
                    "concluded with the individual verifying that all "
                    "appliances had been returned to the inactive state."
                ),
                "retain_informal": (
                    "After dinner, I tackled the kitchen. First I cleared off "
                    "the table and brought all the dirty dishes over to the "
                    "sink. Scraped the leftover food into the trash. Filled "
                    "up the sink with hot water and squirted in some dish soap. "
                    "Then I just went through everything one by one -- wash, "
                    "rinse, put it on the rack. After that I wiped down all "
                    "the counters with a spray and a rag. The stovetop was "
                    "kind of greasy, so I spent a bit of extra time scrubbing "
                    "the grates. Then I swept the floor and mopped it. Took "
                    "out the trash bag, tied it up, and put a new one in. "
                    "Last thing, I made sure everything was turned off -- "
                    "oven, stovetop, all that. Honestly, it's not my favorite "
                    "chore, but the kitchen looked great afterward."
                ),
            },
        ],
    },
    # ── WORKPLACE INTERACTION ────────────────────────────────────────────
    "workplace_interaction": {
        "simple": [
            {
                "target_formal": (
                    "The replenishment of office stationery supplies was "
                    "conducted during the mid-morning interval. The designated "
                    "administrative assistant retrieved the supply requisition "
                    "inventory from the filing cabinet and proceeded to the "
                    "storage closet situated adjacent to the reception area. "
                    "The appropriate quantities of paper, writing instruments, "
                    "and adhesive materials were selected and transported to "
                    "the respective workstations. The inventory document was "
                    "subsequently annotated to reflect the disbursed items."
                ),
                "retain_informal": (
                    "Around mid-morning, I went to restock the office supplies. "
                    "Grabbed the supply list from the filing cabinet and "
                    "headed to the closet by reception. Picked up some paper, "
                    "pens, and tape, and dropped them off at everyone's desks. "
                    "Then I updated the list so we'd know what we still had."
                ),
            },
        ],
        "moderate": [],
        "elaborate": [],
    },
    # ── SOCIAL EVENT ─────────────────────────────────────────────────────
    "social_event": {
        "simple": [
            {
                "target_formal": (
                    "The arrangement of the communal dining experience was "
                    "initiated at the designated restaurant establishment. "
                    "The host individual arrived at the premises and "
                    "communicated the reservation details to the attending "
                    "staff member. The party was subsequently escorted to a "
                    "table positioned in the rear section of the dining area. "
                    "Menus were distributed to each seated individual, and "
                    "water was dispensed into the provided glassware."
                ),
                "retain_informal": (
                    "I got to the restaurant first and told the host about "
                    "our reservation. They walked us to a table in the back. "
                    "Everyone sat down, and they handed out menus and poured "
                    "water for us. Pretty standard start to a dinner out."
                ),
            },
        ],
        "moderate": [],
        "elaborate": [],
    },
    # ── PHYSICAL ACTIVITY ────────────────────────────────────────────────
    "physical_activity": {
        "simple": [
            {
                "target_formal": (
                    "The morning cardiovascular exercise session was commenced "
                    "at approximately six-thirty. The individual donned "
                    "appropriate athletic attire and secured the laces of the "
                    "designated running footwear. Upon exiting the residential "
                    "premises, a brief series of stretching maneuvers was "
                    "performed on the front walkway. The individual then "
                    "proceeded along the established route through the "
                    "neighborhood at a moderate pace. The total distance "
                    "traversed was approximately three kilometers. Upon "
                    "returning to the residence, the individual consumed a "
                    "measured quantity of water."
                ),
                "retain_informal": (
                    "I headed out for a jog around six-thirty. Put on my "
                    "running clothes and laced up my sneakers. Did a few "
                    "stretches on the front porch, then hit the road. I took "
                    "my usual route through the neighborhood -- it's about "
                    "three kilometers. When I got back, I grabbed a glass of "
                    "water. Pretty good run, honestly."
                ),
            },
        ],
        "moderate": [],
        "elaborate": [],
    },
    # ── CONSUMER TRANSACTION ─────────────────────────────────────────────
    "consumer_transaction": {
        "simple": [
            {
                "target_formal": (
                    "The procurement of a railway passage was executed at the "
                    "central station ticketing facility. The traveler "
                    "approached the automated dispensing terminal and selected "
                    "the desired destination from the displayed options. The "
                    "standard fare was indicated on the screen, and payment "
                    "was remitted via contactless card transaction. The "
                    "printed ticket was retrieved from the output tray, and "
                    "the traveler proceeded to the designated platform."
                ),
                "retain_informal": (
                    "I went to the train station and used one of the ticket "
                    "machines. Picked my destination from the list, saw the "
                    "price, and tapped my card to pay. Grabbed the ticket "
                    "when it came out and headed to the platform. Easy enough."
                ),
            },
        ],
        "moderate": [],
        "elaborate": [],
    },
}


def build_system_prompt(category: str, complexity: str) -> str:
    """Build the full system prompt with category and complexity injected."""
    cat_def = CATEGORY_DEFINITIONS[category]
    complexity_desc = COMPLEXITY_DESCRIPTIONS[complexity]
    return SYSTEM_PROMPT.replace("{{category_definition}}", cat_def).replace(
        "{{complexity_description}}", complexity_desc
    )


def build_user_prompt(seed: dict) -> str:
    """Build the user prompt from a scenario skeleton dict."""
    examples_block = ""
    examples = GOLD_EXAMPLES.get(seed["category"], {}).get(seed["complexity"], [])
    if examples:
        example_strs = []
        for i, ex in enumerate(examples, 1):
            example_strs.append(
                f"--- Example {i} ---\n"
                f"target_formal: {ex['target_formal']}\n\n"
                f"retain_informal: {ex['retain_informal']}"
            )
        examples_block = (
            "\n\nHere are gold-standard examples for this category and complexity:\n\n"
            + "\n\n".join(example_strs)
            + "\n\n---\n\nNow generate a NEW pair for the scenario below."
        )

    skeleton_desc = (
        f"Scenario template: {seed['scenario_template']}\n"
        f"Person: {seed['person']}\n"
        f"Location: {seed['location']}\n"
        f"Object A: {seed['object_a']}\n"
        f"Object B: {seed['object_b']}\n"
        f"Time of day: {seed['time_of_day']}"
    )

    return (
        f"Generate a NULL-CC stylistic contrastive pair.\n\n"
        f"Category: {seed['category']}\n"
        f"Complexity: {seed['complexity']}\n\n"
        f"Scenario skeleton:\n{skeleton_desc}"
        f"{examples_block}\n\n"
        f"Respond with ONLY a JSON object with exactly these two keys:\n"
        f'{{"target_formal": "your formal academic prose here", '
        f'"retain_informal": "your informal conversational text here"}}'
    )
