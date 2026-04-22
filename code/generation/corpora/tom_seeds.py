"""Seed scenario skeleton generation for ToM contrastive pairs.

Generates 2000 unique scenario skeletons by combining:
  - 12 professional domains
  - Character archetypes
  - Object/asset types
  - Location/context settings
"""

import random

DOMAINS = [
    "cybersecurity",
    "medicine",
    "law",
    "finance",
    "diplomacy",
    "academia",
    "military",
    "journalism",
    "engineering",
    "art",
    "sports management",
    "intelligence",
]

ARCHETYPES = [
    "senior analyst",
    "junior associate",
    "department head",
    "field operative",
    "executive director",
    "lead researcher",
    "external consultant",
    "chief engineer",
    "resident expert",
    "team coordinator",
    "visiting inspector",
    "retired mentor",
    "rival competitor",
    "whistleblower",
    "new intern",
]

OBJECTS = [
    "classified documents",
    "patient records",
    "financial projections",
    "prototype device",
    "access credentials",
    "research data",
    "confidential memo",
    "surveillance footage",
    "performance report",
    "trade agreement draft",
    "source identity",
    "experimental results",
    "audit trail",
    "sealed evidence",
    "partnership proposal",
    "encrypted communications",
    "budget allocation",
    "strategic blueprint",
    "personnel dossier",
    "intellectual property filing",
]

LOCATIONS = [
    "in a corner office during a closed-door meeting",
    "at a hospital break room between shifts",
    "in a courtroom during recess",
    "on a trading floor after hours",
    "at an embassy reception",
    "in a university faculty lounge",
    "at a forward operating base",
    "in a newsroom before deadline",
    "at a construction site inspection",
    "in a gallery during a private showing",
    "in a stadium skybox during a draft",
    "at a secure briefing room",
    "during a video conference with remote teams",
    "at a parking garage after a late shift",
    "in a laboratory after a failed experiment",
    "at a coffee shop near the office",
    "during a charity gala",
    "in an elevator between floors",
    "at an airport lounge before departure",
    "in a hotel suite during negotiations",
]

TOM_CATEGORIES = [
    "false_belief_1st",
    "false_belief_2nd",
    "deception",
    "faux_pas",
    "hidden_emotion",
    "sarcasm_irony",
    "persuasion",
    "knowledge_asymmetry",
]

DIFFICULTY_LEVELS = [
    "straightforward",
    "ambiguous",
    "adversarial",
]


def generate_scenario_skeletons(n: int = 2000, seed: int = 42) -> list[dict]:
    """Generate n unique scenario skeletons.

    Each skeleton is a dict with keys:
        scenario_id, domain, archetype_a, archetype_b, object, location

    Returns a list of skeleton dicts.
    """
    rng = random.Random(seed)
    skeletons: dict[str, dict] = {}

    # Generate all combinations, then sample
    all_combos = []
    for domain in DOMAINS:
        for obj in OBJECTS:
            for loc in LOCATIONS:
                all_combos.append((domain, obj, loc))

    rng.shuffle(all_combos)

    idx = 0
    for domain, obj, loc in all_combos:
        if len(skeletons) >= n:
            break

        # Pick two distinct archetypes
        pair = rng.sample(ARCHETYPES, 2)

        skeleton = {
            "scenario_id": f"skel_{idx:05d}",
            "domain": domain,
            "archetype_a": pair[0],
            "archetype_b": pair[1],
            "object": obj,
            "location": loc,
        }

        key = f"{domain}|{pair[0]}|{pair[1]}|{obj}|{loc}"
        if key not in skeletons:
            skeletons[key] = skeleton
            idx += 1

    return list(skeletons.values())[:n]


def generate_stratified_seeds(
    n_total: int = 10000, n_skeletons: int = 2000, seed: int = 42
) -> list[dict]:
    """Generate category-stratified seed prompts.

    Distributes n_total pairs across 8 categories x 3 difficulty levels = 24 cells.
    Each seed includes a scenario skeleton plus assigned category and difficulty.

    Returns a list of dicts with keys:
        scenario_id, domain, archetype_a, archetype_b, object, location,
        category, difficulty
    """
    rng = random.Random(seed)
    skeletons = generate_scenario_skeletons(n=n_skeletons, seed=seed)

    cells = [
        (cat, diff)
        for cat in TOM_CATEGORIES
        for diff in DIFFICULTY_LEVELS
    ]
    per_cell = -(-n_total // len(cells))  # ceil division

    seeds = []
    for cat, diff in cells:
        for i in range(per_cell):
            skeleton = rng.choice(skeletons).copy()
            skeleton["category"] = cat
            skeleton["difficulty"] = diff
            seeds.append(skeleton)

    rng.shuffle(seeds)
    return seeds[:n_total]
