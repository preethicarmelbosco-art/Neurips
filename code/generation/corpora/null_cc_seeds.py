"""Seed scenario skeleton generation for NULL-CC contrastive pairs.

Generates 2000 unique everyday scenario skeletons by combining:
  - 5 scenario categories
  - Scenario templates, persons, locations, objects, times of day
Then stratifies across 5 categories x 3 complexities = 15 cells.
"""

import random

SCENARIO_CATEGORIES = [
    "everyday_routine",
    "workplace_interaction",
    "social_event",
    "physical_activity",
    "consumer_transaction",
]

COMPLEXITY_LEVELS = [
    "simple",
    "moderate",
    "elaborate",
]

# --- Per-category scenario templates ---

SCENARIO_TEMPLATES = {
    "everyday_routine": [
        "morning coffee preparation",
        "grocery shopping trip",
        "commuting to work by bus",
        "cooking dinner for family",
        "doing laundry",
        "walking the dog in the park",
        "cleaning the kitchen",
        "checking email and responding",
        "watering the garden plants",
        "packing a lunch for work",
        "taking out the recycling",
        "ironing clothes for the week",
        "feeding the cat in the morning",
        "organizing the bookshelf",
        "making the bed after waking up",
    ],
    "workplace_interaction": [
        "attending a team meeting",
        "onboarding a new colleague",
        "presenting quarterly results",
        "resolving a scheduling conflict",
        "organizing office supplies",
        "setting up a video conference",
        "distributing printed memos",
        "arranging desks for a new hire",
        "signing in visitors at reception",
        "refilling the printer paper tray",
        "collecting feedback forms",
        "labeling storage boxes in the archive room",
        "scheduling a conference room booking",
        "handing out name badges at orientation",
        "updating the office whiteboard calendar",
    ],
    "social_event": [
        "hosting a birthday party",
        "attending a wedding reception",
        "organizing a neighborhood barbecue",
        "planning a surprise celebration",
        "meeting friends at a restaurant",
        "setting up decorations for a holiday gathering",
        "preparing appetizers for a dinner party",
        "greeting guests at a housewarming",
        "distributing party favors at a children's event",
        "arranging seating at a family reunion",
        "serving drinks at an outdoor festival",
        "collecting coats at a formal reception",
        "lighting candles for an evening soiree",
        "passing out programs at a community concert",
        "assembling a photo booth at a graduation party",
    ],
    "physical_activity": [
        "running a morning jog",
        "playing a tennis match",
        "swimming laps at the pool",
        "hiking a mountain trail",
        "cycling through the city",
        "stretching before a yoga class",
        "lifting weights at the gym",
        "warming up on the basketball court",
        "skating at an ice rink",
        "rowing on the lake in the morning",
        "jumping rope in the backyard",
        "doing push-ups at the park",
        "practicing golf swings at the driving range",
        "kicking a soccer ball on the field",
        "climbing the indoor bouldering wall",
    ],
    "consumer_transaction": [
        "buying groceries at the supermarket",
        "returning a defective product",
        "ordering food at a restaurant",
        "purchasing a train ticket",
        "negotiating a car price",
        "paying for dry cleaning services",
        "selecting a mobile phone plan",
        "renting a bicycle from a kiosk",
        "checking out library books",
        "buying stamps at the post office",
        "ordering a custom cake from a bakery",
        "purchasing concert tickets at the box office",
        "picking up a prescription at the pharmacy",
        "buying flowers from a street vendor",
        "reserving a hotel room at the front desk",
    ],
}

PERSONS = [
    "the customer",
    "the employee",
    "the resident",
    "the commuter",
    "the participant",
    "the shopper",
    "the visitor",
    "the tenant",
    "the homeowner",
    "the pedestrian",
    "the patron",
    "the attendee",
    "the jogger",
    "the cyclist",
    "the traveler",
]

LOCATIONS = [
    "at the local supermarket",
    "in the office building",
    "at the community center",
    "in the neighborhood park",
    "at the downtown train station",
    "in the shopping mall",
    "at the fitness center",
    "in the apartment kitchen",
    "at the restaurant on Main Street",
    "in the public library",
    "at the corner coffee shop",
    "in the hotel lobby",
    "at the parking garage",
    "in the laundromat on Oak Avenue",
    "at the outdoor farmers market",
]

OBJECTS = [
    "the shopping cart",
    "the conference room table",
    "the exercise equipment",
    "the kitchen counter",
    "the office printer",
    "the bus pass",
    "the umbrella",
    "the reusable grocery bag",
    "the folding chair",
    "the clipboard",
    "the water bottle",
    "the receipt",
    "the name tag",
    "the parking meter",
    "the paper napkin",
]

TIMES_OF_DAY = [
    "early in the morning",
    "during the mid-morning break",
    "around noon",
    "in the early afternoon",
    "during the late afternoon",
    "in the early evening",
    "after dinner",
    "late at night",
    "at the start of the workday",
    "during lunchtime",
]


def generate_scenario_skeletons(n: int = 300, seed: int = 42) -> list[dict]:
    """Generate n unique everyday scenario skeletons.

    Each skeleton is a dict with keys:
        scenario_id, category, scenario_template, person, location,
        object_a, object_b, time_of_day
    """
    rng = random.Random(seed)
    skeletons: list[dict] = []
    seen_keys: set[str] = set()

    idx = 0
    attempts = 0
    max_attempts = n * 20  # safety valve

    while len(skeletons) < n and attempts < max_attempts:
        attempts += 1
        category = rng.choice(SCENARIO_CATEGORIES)
        scenario_template = rng.choice(SCENARIO_TEMPLATES[category])
        person = rng.choice(PERSONS)
        location = rng.choice(LOCATIONS)
        objects = rng.sample(OBJECTS, 2)
        time_of_day = rng.choice(TIMES_OF_DAY)

        key = f"{category}|{scenario_template}|{person}|{location}"
        if key in seen_keys:
            continue
        seen_keys.add(key)

        skeletons.append({
            "scenario_id": f"null_{idx:05d}",
            "category": category,
            "scenario_template": scenario_template,
            "person": person,
            "location": location,
            "object_a": objects[0],
            "object_b": objects[1],
            "time_of_day": time_of_day,
        })
        idx += 1

    return skeletons[:n]


def generate_stratified_seeds(
    n_total: int = 1000, n_skeletons: int = 300, seed: int = 42
) -> list[dict]:
    """Generate category-stratified seed prompts.

    Distributes n_total pairs across 5 categories x 3 complexities
    = 15 cells (~67 pairs per cell).

    Returns list of dicts with skeleton fields plus complexity.
    """
    rng = random.Random(seed)
    skeletons = generate_scenario_skeletons(n=n_skeletons, seed=seed)

    # Group skeletons by category for balanced assignment
    by_category: dict[str, list[dict]] = {c: [] for c in SCENARIO_CATEGORIES}
    for s in skeletons:
        by_category[s["category"]].append(s)

    cells = [
        (cat, complexity)
        for cat in SCENARIO_CATEGORIES
        for complexity in COMPLEXITY_LEVELS
    ]
    per_cell = -(-n_total // len(cells))  # ceil division

    seeds = []
    for cat, complexity in cells:
        cat_skeletons = by_category[cat]
        for i in range(per_cell):
            if cat_skeletons:
                skeleton = rng.choice(cat_skeletons).copy()
            else:
                # Fallback for small N: pick any skeleton, override category
                skeleton = rng.choice(skeletons).copy()
                skeleton["category"] = cat
            skeleton["complexity"] = complexity
            seeds.append(skeleton)

    rng.shuffle(seeds)
    return seeds[:n_total]
