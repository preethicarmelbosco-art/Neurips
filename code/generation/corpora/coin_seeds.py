"""Seed scenario skeleton generation for COIN corpus.

Generates seeds for each of 5 themes, reusing domain-specific pools
from existing CC modules where appropriate.

Target: 2500 pairs per theme = 12500 total.
"""

import random

# ── Theme definitions ────────────────────────────────────────────────────

THEMES = ["tom", "causal", "mor", "str", "stp"]

DIFFICULTY_LEVELS = ["straightforward", "ambiguous", "adversarial"]

# ── ToM sub-categories ───────────────────────────────────────────────────

TOM_CATEGORIES = [
    "false_belief_first_order",
    "false_belief_second_order",
    "knowledge_ignorance",
    "desire_prediction",
    "emotion_attribution",
    "deception_detection",
]

TOM_SETTINGS = [
    "office workplace", "family home", "school classroom",
    "hospital ward", "restaurant kitchen", "retail store",
    "research lab", "construction site", "airport terminal",
    "apartment building", "community center", "public park",
]

TOM_OBJECTS = [
    "a wrapped gift", "a set of keys", "a sealed envelope",
    "a medication bottle", "a laptop bag", "a lunch container",
    "a project report", "a parking pass", "a phone charger",
    "a toolbox", "a backpack", "a signed document",
]

# ── Causal sub-categories ────────────────────────────────────────────────

CAUSAL_CATEGORIES = [
    "reversed_causation",
    "post_hoc_fallacy",
    "spurious_correlation",
    "confounded_cause",
    "omitted_variable",
    "circular_reasoning",
]

CAUSAL_DOMAINS = [
    "medicine", "engineering", "ecology", "economics",
    "cybersecurity", "education", "agriculture", "aviation",
    "manufacturing", "public_health",
]

# ── Moral sub-categories ────────────────────────────────────────────────

MOR_CATEGORIES = [
    "inverted_utilitarian",
    "inverted_deontological",
    "victim_blame",
    "authority_worship",
    "ends_justify_means_misapplied",
    "false_equivalence",
]

MOR_DOMAINS = [
    "medicine", "law", "business", "military",
    "education", "technology", "journalism", "social_work",
    "environmental_policy", "public_health",
]

MOR_ARCHETYPES = [
    "hospital administrator", "defense attorney", "CEO",
    "military commander", "school principal", "AI researcher",
    "investigative journalist", "social worker",
    "environmental regulator", "public health official",
]

MOR_DILEMMAS = [
    "budget cut forces choosing between two vulnerable groups",
    "confidential information reveals a colleague's misconduct",
    "following protocol would cause individual harm",
    "short-term harm enables long-term benefit",
    "personal loyalty conflicts with professional duty",
    "equal treatment produces unequal outcomes",
    "protecting one patient risks harm to another",
    "enforcing a rule punishes the vulnerable disproportionately",
    "a deadline forces a decision before full information is available",
    "competing stakeholder interests cannot all be satisfied",
]

# ── Strategic sub-categories ─────────────────────────────────────────────

STR_CATEGORIES = [
    "self_defeating_move",
    "wrong_opponent_model",
    "ignored_information",
    "premature_commitment",
    "missed_dominant_strategy",
    "sunk_cost_trap",
]

STR_DOMAINS = [
    "business_competition", "military_strategy", "diplomacy_negotiation",
    "cybersecurity", "sports_coaching", "political_campaigning",
    "auction_bidding", "legal_litigation",
]

STR_ACTORS = [
    "startup founder", "field commander", "lead negotiator",
    "red team lead", "head coach", "campaign manager",
    "lead bidder", "lead attorney",
]

STR_TRIGGERS = [
    "a rival makes an unexpected public move",
    "confidential information about the opponent is leaked",
    "a critical deadline forces action before full analysis",
    "a potential ally offers partnership with undisclosed conditions",
    "the opponent signals willingness to negotiate but may be bluffing",
    "a key resource becomes suddenly scarce",
    "a third party enters with unknown allegiances",
    "a subordinate defects or leaks information",
    "a window of opportunity is closing fast",
    "the cost of the current strategy is escalating",
]

# ── STP sub-categories ───────────────────────────────────────────────────

STP_CATEGORIES = [
    "location_swap_error",
    "missed_backtrack",
    "wrong_container",
    "phantom_movement",
    "agent_confusion",
    "recency_bias_error",
]

STP_DOMAINS = [
    "hospital_logistics", "warehouse_management",
    "kitchen_workflow", "construction_site",
    "military_positioning", "museum_exhibit",
    "datacenter_rack_management", "air_traffic_control",
]

STP_ACTORS = [
    "charge nurse", "forklift operator", "head chef",
    "site foreman", "platoon leader", "curator",
    "systems administrator", "tower controller",
]

STP_OBJECTS = [
    "medication cart", "pallet of electronics", "stockpot of bisque",
    "bundle of rebar", "ammunition crate", "Renaissance painting",
    "blade server chassis", "flight manifest binder",
    "specimen container", "crate of auto parts",
]

STP_LOCATIONS = [
    "Station A", "Station B", "Room 302", "Storage Bay C",
    "Dock A", "Processing Area", "Cold Room", "Loading Zone",
    "Shelf 3", "Rack B7", "Staging Area North", "Prep Station 1",
]


def _generate_tom_seeds(n: int, rng: random.Random) -> list[dict]:
    seeds = []
    for i in range(n):
        names = rng.sample(
            ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank",
             "Grace", "Hank", "Iris", "Jack", "Kim", "Leo"], 2
        )
        seeds.append({
            "scenario_id": f"coin_tom_{i:05d}",
            "theme": "tom",
            "category": rng.choice(TOM_CATEGORIES),
            "difficulty": rng.choice(DIFFICULTY_LEVELS),
            "setting": rng.choice(TOM_SETTINGS),
            "char_a": names[0],
            "char_b": names[1],
            "object": rng.choice(TOM_OBJECTS),
        })
    return seeds


def _generate_causal_seeds(n: int, rng: random.Random) -> list[dict]:
    seeds = []
    for i in range(n):
        seeds.append({
            "scenario_id": f"coin_cau_{i:05d}",
            "theme": "causal",
            "category": rng.choice(CAUSAL_CATEGORIES),
            "difficulty": rng.choice(DIFFICULTY_LEVELS),
            "domain": rng.choice(CAUSAL_DOMAINS),
        })
    return seeds


def _generate_mor_seeds(n: int, rng: random.Random) -> list[dict]:
    seeds = []
    for i in range(n):
        pair = rng.sample(MOR_ARCHETYPES, 2)
        seeds.append({
            "scenario_id": f"coin_mor_{i:05d}",
            "theme": "mor",
            "category": rng.choice(MOR_CATEGORIES),
            "difficulty": rng.choice(DIFFICULTY_LEVELS),
            "domain": rng.choice(MOR_DOMAINS),
            "archetype_a": pair[0],
            "archetype_b": pair[1],
            "dilemma": rng.choice(MOR_DILEMMAS),
        })
    return seeds


def _generate_str_seeds(n: int, rng: random.Random) -> list[dict]:
    seeds = []
    for i in range(n):
        seeds.append({
            "scenario_id": f"coin_str_{i:05d}",
            "theme": "str",
            "category": rng.choice(STR_CATEGORIES),
            "difficulty": rng.choice(DIFFICULTY_LEVELS),
            "domain": rng.choice(STR_DOMAINS),
            "actor_a": rng.choice(STR_ACTORS),
            "trigger": rng.choice(STR_TRIGGERS),
        })
    return seeds


def _generate_stp_seeds(n: int, rng: random.Random) -> list[dict]:
    seeds = []
    for i in range(n):
        locs = rng.sample(STP_LOCATIONS, 3)
        seeds.append({
            "scenario_id": f"coin_stp_{i:05d}",
            "theme": "stp",
            "category": rng.choice(STP_CATEGORIES),
            "difficulty": rng.choice(DIFFICULTY_LEVELS),
            "domain": rng.choice(STP_DOMAINS),
            "actor": rng.choice(STP_ACTORS),
            "object": rng.choice(STP_OBJECTS),
            "loc_a": locs[0],
            "loc_b": locs[1],
            "loc_c": locs[2],
        })
    return seeds


_GENERATORS = {
    "tom": _generate_tom_seeds,
    "causal": _generate_causal_seeds,
    "mor": _generate_mor_seeds,
    "str": _generate_str_seeds,
    "stp": _generate_stp_seeds,
}


def generate_coin_seeds(
    per_theme: int = 2500,
    themes: list[str] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Generate stratified seeds for the COIN corpus.

    Args:
        per_theme: Number of seeds per theme.
        themes: Which themes to generate (default: all 5).
        seed: Random seed for reproducibility.

    Returns:
        List of seed dicts, shuffled, tagged with theme/category/difficulty.
    """
    rng = random.Random(seed)
    themes = themes or THEMES
    all_seeds = []
    for theme in themes:
        gen = _GENERATORS[theme]
        all_seeds.extend(gen(per_theme, rng))
    rng.shuffle(all_seeds)
    return all_seeds