"""Seed scenario skeleton generation for MOR-CC (Moral Reasoning) contrastive pairs.

Generates 2000 unique scenario skeletons by combining:
  - 12 professional domains
  - Character archetypes (moral agents)
  - Dilemma triggers
  - Location/context settings

Categories: 8 moral reasoning types × 3 difficulty levels = 24 cells.
"""

import random

MORAL_CATEGORIES = [
    "utilitarian_vs_deontological",
    "rights_vs_welfare",
    "individual_vs_collective_harm",
    "intent_vs_outcome",
    "duty_of_care_vs_autonomy",
    "fairness_under_scarcity",
    "loyalty_vs_whistleblowing",
    "cultural_moral_relativism",
]

DIFFICULTY_LEVELS = [
    "straightforward",
    "ambiguous",
    "adversarial",
]

MORAL_DOMAINS = [
    "medicine",
    "law",
    "business",
    "military",
    "education",
    "technology",
    "journalism",
    "social_work",
    "environmental_policy",
    "criminal_justice",
    "scientific_research",
    "public_health",
]

MORAL_ARCHETYPES = [
    "hospital administrator",
    "defense attorney",
    "CEO",
    "military commander",
    "school principal",
    "AI researcher",
    "investigative journalist",
    "social worker",
    "environmental regulator",
    "parole officer",
    "clinical trial director",
    "public health official",
    "whistleblower",
    "ethics committee chair",
    "new hire",
]

DILEMMA_TRIGGERS = [
    "budget cut forces choosing between two vulnerable groups",
    "confidential information reveals a colleague's misconduct",
    "following protocol would cause individual harm",
    "cultural practice conflicts with institutional policy",
    "short-term harm enables long-term benefit",
    "personal loyalty conflicts with professional duty",
    "transparent disclosure would cause panic",
    "equal treatment produces unequal outcomes",
    "consent is unclear or cannot be obtained",
    "precedent conflicts with the specific case",
    "resource allocation favors efficiency over equity",
    "truth-telling would violate a promise of confidentiality",
    "a subordinate's well-being conflicts with organizational goals",
    "new evidence contradicts a publicly committed position",
    "automation displaces workers but improves safety",
    "protecting one patient risks harm to another",
    "accepting a donation requires compromising independence",
    "enforcing a rule punishes the vulnerable disproportionately",
    "a deadline forces a decision before full information is available",
    "competing stakeholder interests cannot all be satisfied",
]

LOCATIONS = [
    "in a hospital ethics board meeting",
    "at a corporate boardroom during a crisis session",
    "in a courtroom during sentencing deliberation",
    "at a military field headquarters",
    "in a school district office reviewing policy",
    "at a tech company's AI safety review",
    "in a newsroom debating whether to publish",
    "at a social services intake center",
    "in an EPA hearing room",
    "at a parole board session",
    "in a clinical trial oversight committee",
    "at a public health emergency briefing",
    "during a town hall meeting with constituents",
    "in a university tenure committee meeting",
    "at an NGO fundraising ethics review",
    "in a pharmaceutical pricing committee",
    "at a refugee processing center",
    "during a labor arbitration hearing",
    "in a research lab after discovering data fraud",
    "at an insurance claims review panel",
]


def generate_scenario_skeletons(n: int = 2000, seed: int = 42) -> list[dict]:
    """Generate n unique scenario skeletons.

    Each skeleton is a dict with keys:
        scenario_id, domain, archetype_a, archetype_b, dilemma_trigger, location

    Returns a list of skeleton dicts.
    """
    rng = random.Random(seed)
    skeletons: dict[str, dict] = {}

    all_combos = []
    for domain in MORAL_DOMAINS:
        for trigger in DILEMMA_TRIGGERS:
            for loc in LOCATIONS:
                all_combos.append((domain, trigger, loc))

    rng.shuffle(all_combos)

    idx = 0
    for domain, trigger, loc in all_combos:
        if len(skeletons) >= n:
            break

        pair = rng.sample(MORAL_ARCHETYPES, 2)

        skeleton = {
            "scenario_id": f"skel_{idx:05d}",
            "domain": domain,
            "archetype_a": pair[0],
            "archetype_b": pair[1],
            "dilemma_trigger": trigger,
            "location": loc,
        }

        key = f"{domain}|{pair[0]}|{pair[1]}|{trigger}|{loc}"
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
    """
    rng = random.Random(seed)
    skeletons = generate_scenario_skeletons(n=n_skeletons, seed=seed)

    cells = [
        (cat, diff)
        for cat in MORAL_CATEGORIES
        for diff in DIFFICULTY_LEVELS
    ]
    per_cell = -(-n_total // len(cells))  # ceil division

    seeds = []
    for cat, diff in cells:
        for _ in range(per_cell):
            skeleton = rng.choice(skeletons).copy()
            skeleton["category"] = cat
            skeleton["difficulty"] = diff
            seeds.append(skeleton)

    rng.shuffle(seeds)
    return seeds[:n_total]
