"""Seed scenario skeleton generation for STR-CC (Strategic Reasoning) contrastive pairs.

Generates 2000 unique scenario skeletons by combining:
  - 8 strategic domains
  - Domain-specific actor types
  - Strategic triggers (what forces the strategic choice)
  - Location/context settings

Categories: 6 strategic reasoning types × 3 difficulty levels = 18 cells.
"""

import random

STRATEGIC_CATEGORIES = [
    "first_mover_advantage",
    "information_asymmetry_leverage",
    "bluffing_signaling_commitment",
    "resource_denial_positional_play",
    "alliance_formation_betrayal",
    "multi_stage_sequential_planning",
]

DIFFICULTY_LEVELS = [
    "straightforward",
    "ambiguous",
    "adversarial",
]

STRATEGIC_DOMAINS = [
    "business_competition",
    "military_strategy",
    "diplomacy_negotiation",
    "cybersecurity",
    "sports_coaching",
    "political_campaigning",
    "auction_bidding",
    "legal_litigation",
]

DOMAIN_ACTORS = {
    "business_competition": [
        "startup founder", "incumbent CEO", "venture capitalist",
        "market analyst", "supply chain manager",
    ],
    "military_strategy": [
        "field commander", "intelligence officer", "logistics chief",
        "allied commander", "defense minister",
    ],
    "diplomacy_negotiation": [
        "lead negotiator", "foreign minister", "trade representative",
        "ambassador", "intelligence attaché",
    ],
    "cybersecurity": [
        "red team lead", "SOC analyst", "CISO",
        "threat actor", "incident responder",
    ],
    "sports_coaching": [
        "head coach", "assistant coach", "general manager",
        "scout analyst", "team captain",
    ],
    "political_campaigning": [
        "campaign manager", "opposition strategist", "polling analyst",
        "media advisor", "party chairman",
    ],
    "auction_bidding": [
        "lead bidder", "rival bidder", "auction consultant",
        "financial backer", "proxy buyer",
    ],
    "legal_litigation": [
        "lead attorney", "opposing counsel", "expert witness",
        "paralegal strategist", "presiding judge",
    ],
}

STRATEGIC_ASSETS = [
    "market intelligence report", "classified deployment plans",
    "patent portfolio", "voter demographic data",
    "vulnerability assessment", "supply route maps",
    "financial projections", "sealed bid documents",
    "draft treaty terms", "game footage analysis",
    "polling microdata", "merger proposal",
    "threat intelligence feed", "witness deposition",
    "spectrum license rights", "trade secret documentation",
    "alliance commitment letter", "budget allocation report",
    "personnel readiness assessment", "regulatory filing draft",
]

STRATEGIC_TRIGGERS = [
    "a rival makes an unexpected public move that shifts the landscape",
    "confidential information about the opponent's position is leaked",
    "a critical deadline forces action before full analysis is possible",
    "a potential ally offers partnership with undisclosed conditions",
    "new intelligence reveals the opponent's strategy is different than assumed",
    "a key resource becomes suddenly scarce or contested",
    "the opponent signals willingness to negotiate but may be bluffing",
    "a third party enters the field with unknown allegiances",
    "internal pressure forces a premature commitment to a course of action",
    "a past agreement creates obligations that constrain current options",
    "the opponent's apparent weakness may be a deliberate trap",
    "two allies make conflicting demands that cannot both be satisfied",
    "regulatory or rule changes alter the feasibility of the planned approach",
    "a subordinate defects or leaks information to the opposition",
    "a window of opportunity is closing and delay means losing position",
    "the cost of the current strategy is escalating beyond projections",
    "a ceasefire or truce offer arrives at a strategically inconvenient moment",
    "success in one arena creates vulnerability in another",
    "the opponent proposes joint action whose true purpose is unclear",
    "a previously reliable information source sends contradictory signals",
]

LOCATIONS = [
    "in a corporate war room during a hostile takeover bid",
    "at a military command post during a border standoff",
    "during closed-door trade negotiations between delegations",
    "in a cybersecurity operations center during an active breach",
    "at halftime in a championship elimination game",
    "at campaign headquarters on the eve of a primary election",
    "in an auction house during a high-stakes bidding round",
    "in a courtroom during cross-examination of a key witness",
    "at a diplomatic summit with media cameras outside",
    "in a venture capital pitch meeting with competing startups",
    "at a military briefing room before a major operation",
    "during a board meeting debating a merger counteroffer",
    "in a negotiation room during a hostage situation",
    "at a sports draft with rival teams watching the same prospects",
    "during a regulatory hearing with industry competitors present",
    "in a situation room monitoring an adversary's real-time movements",
    "at a political debate backstage before going live",
    "during a patent dispute mediation session",
    "in a trading floor during a market-moving announcement",
    "at a coalition meeting where member interests are diverging",
]


def generate_scenario_skeletons(n: int = 2000, seed: int = 42) -> list[dict]:
    """Generate n unique scenario skeletons.

    Each skeleton is a dict with keys:
        scenario_id, domain, actor_a, actor_b, asset, trigger, location

    Returns a list of skeleton dicts.
    """
    rng = random.Random(seed)
    skeletons: dict[str, dict] = {}

    all_combos = []
    for domain in STRATEGIC_DOMAINS:
        for trigger in STRATEGIC_TRIGGERS:
            for loc in LOCATIONS:
                all_combos.append((domain, trigger, loc))

    rng.shuffle(all_combos)

    idx = 0
    for domain, trigger, loc in all_combos:
        if len(skeletons) >= n:
            break

        actors = DOMAIN_ACTORS[domain]
        pair = rng.sample(actors, 2)
        asset = rng.choice(STRATEGIC_ASSETS)

        skeleton = {
            "scenario_id": f"str_{idx:05d}",
            "domain": domain,
            "actor_a": pair[0],
            "actor_b": pair[1],
            "asset": asset,
            "trigger": trigger,
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

    Distributes n_total pairs across 6 categories x 3 difficulty levels = 18 cells.
    Each seed includes a scenario skeleton plus assigned category and difficulty.
    """
    rng = random.Random(seed)
    skeletons = generate_scenario_skeletons(n=n_skeletons, seed=seed)

    cells = [
        (cat, diff)
        for cat in STRATEGIC_CATEGORIES
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
