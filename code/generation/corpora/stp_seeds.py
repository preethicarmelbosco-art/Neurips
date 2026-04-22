"""Seed scenario skeleton generation for STP-CC (Spatial-Temporal Tracking) contrastive pairs.

Generates 2000 unique scenario skeletons by combining:
  - 8 spatial/logistic domains
  - Domain-specific actors
  - Trackable objects
  - Locations within each domain

Categories: 6 spatial tracking types × 3 difficulty levels = 18 cells.
"""

import random

SPATIAL_CATEGORIES = [
    "single_object_2step",
    "multi_object_tracking",
    "tracking_with_distractors",
    "partial_information",
    "tracking_with_backtracking",
    "multi_agent_tracking",
]

DIFFICULTY_LEVELS = [
    "straightforward",
    "ambiguous",
    "adversarial",
]

SPATIAL_DOMAINS = [
    "hospital_logistics",
    "warehouse_management",
    "air_traffic_control",
    "kitchen_workflow",
    "construction_site",
    "military_positioning",
    "museum_exhibit",
    "datacenter_rack_management",
]

DOMAIN_ACTORS = {
    "hospital_logistics": [
        "charge nurse", "orderly", "pharmacist", "lab technician",
        "attending physician",
    ],
    "warehouse_management": [
        "forklift operator", "inventory clerk", "shift supervisor",
        "receiving dock worker", "quality inspector",
    ],
    "air_traffic_control": [
        "tower controller", "ground handler", "flight dispatcher",
        "ramp coordinator", "maintenance technician",
    ],
    "kitchen_workflow": [
        "head chef", "sous chef", "line cook", "pastry chef",
        "expeditor",
    ],
    "construction_site": [
        "site foreman", "crane operator", "materials handler",
        "safety inspector", "project engineer",
    ],
    "military_positioning": [
        "platoon leader", "logistics sergeant", "communications officer",
        "supply runner", "battalion commander",
    ],
    "museum_exhibit": [
        "curator", "exhibit technician", "registrar",
        "security officer", "conservation specialist",
    ],
    "datacenter_rack_management": [
        "systems administrator", "cabling technician", "NOC operator",
        "hardware engineer", "facilities manager",
    ],
}

DOMAIN_OBJECTS = {
    "hospital_logistics": [
        "patient chart", "blood sample tray", "medication cart",
        "portable ventilator", "surgical kit", "IV pump",
        "specimen container", "crash cart",
    ],
    "warehouse_management": [
        "pallet of electronics", "crate of auto parts", "shipment of chemicals",
        "roll cage of textiles", "bin of returned goods", "container of perishables",
        "rack of server equipment", "drum of lubricant",
    ],
    "air_traffic_control": [
        "flight manifest binder", "weather briefing folder", "runway status board",
        "NOTAMs printout", "flight strip rack", "radar calibration log",
        "emergency frequency card", "airport diagram chart",
    ],
    "kitchen_workflow": [
        "stockpot of bisque", "sheet tray of pastries", "container of mise en place",
        "hotel pan of braised short ribs", "cambro of soup", "speed rack of desserts",
        "lexan of marinating proteins", "bus tub of prepped vegetables",
    ],
    "construction_site": [
        "bundle of rebar", "pallet of concrete blocks", "spool of electrical cable",
        "crate of plumbing fittings", "stack of steel beams", "box of safety harnesses",
        "drum of waterproofing compound", "kit of surveying equipment",
    ],
    "military_positioning": [
        "ammunition crate", "communications radio set", "field medical kit",
        "encrypted map case", "fuel drum", "mobile antenna array",
        "night vision equipment case", "rations supply box",
    ],
    "museum_exhibit": [
        "Renaissance painting", "Bronze Age artifact", "photography collection box",
        "ceramic vase from Ming dynasty", "fossil specimen crate",
        "textile tapestry roll", "archival document folder", "sculpture pedestal mount",
    ],
    "datacenter_rack_management": [
        "blade server chassis", "network switch unit", "UPS battery module",
        "fiber patch panel", "storage array shelf", "GPU compute node",
        "KVM console drawer", "cable management tray",
    ],
}

DOMAIN_LOCATIONS = {
    "hospital_logistics": [
        "Station A (nurses' station)", "Station B (pharmacy window)",
        "Room 302 (patient room)", "Room 415 (surgical prep)",
        "Supply closet 2C", "Lab processing desk",
        "ICU bay 7", "Discharge lounge",
    ],
    "warehouse_management": [
        "Receiving Dock A", "Aisle 14 Rack B3",
        "Staging Area North", "Quality Hold Zone",
        "Shipping Dock C", "Cold Storage Room 2",
        "Returns Processing Bay", "Overflow Zone East",
    ],
    "air_traffic_control": [
        "Tower cab desk 1", "Ground control station",
        "Briefing room table", "Flight data processing rack",
        "Emergency shelf", "Supervisor console",
        "Break room counter", "Radar room console 3",
    ],
    "kitchen_workflow": [
        "Prep station 1", "Walk-in cooler shelf 3",
        "Hot line pass", "Garde manger station",
        "Pastry bench", "Expeditor window",
        "Dry storage rack B", "Dishpit return counter",
    ],
    "construction_site": [
        "Laydown yard sector A", "Floor 3 staging area",
        "Tool crib", "Crane loading zone",
        "Materials trailer", "Foundation excavation pit",
        "Electrical closet floor 2", "Site office storage room",
    ],
    "military_positioning": [
        "Forward operating base alpha", "Supply depot bravo",
        "Observation post charlie", "Command tent",
        "Ammunition storage bunker", "Medical aid station",
        "Communications relay point", "Vehicle staging area",
    ],
    "museum_exhibit": [
        "Gallery A display case 3", "Storage vault room 12",
        "Conservation lab bench", "Loading dock crate area",
        "Registrar's processing desk", "Temporary exhibit hall B",
        "Climate-controlled archive", "Photography studio table",
    ],
    "datacenter_rack_management": [
        "Rack A07 in hot aisle 2", "Rack C14 in cold aisle 3",
        "Staging bench", "Loading dock pallet",
        "NOC monitoring station", "Cable tray overhead section 5",
        "Battery room shelf B", "Decommission holding area",
    ],
}


def generate_scenario_skeletons(n: int = 2000, seed: int = 42) -> list[dict]:
    """Generate n unique scenario skeletons.

    Each skeleton is a dict with keys:
        scenario_id, domain, actor_a, actor_b, object_1, object_2,
        location_a, location_b, location_c
    """
    rng = random.Random(seed)
    skeletons: dict[str, dict] = {}

    idx = 0
    attempts = 0
    max_attempts = n * 20

    while len(skeletons) < n and attempts < max_attempts:
        attempts += 1
        domain = rng.choice(SPATIAL_DOMAINS)
        actors = DOMAIN_ACTORS[domain]
        objects = DOMAIN_OBJECTS[domain]
        locations = DOMAIN_LOCATIONS[domain]

        pair = rng.sample(actors, 2)
        obj_pair = rng.sample(objects, min(2, len(objects)))
        loc_triple = rng.sample(locations, min(3, len(locations)))

        skeleton = {
            "scenario_id": f"stp_{idx:05d}",
            "domain": domain,
            "actor_a": pair[0],
            "actor_b": pair[1],
            "object_1": obj_pair[0],
            "object_2": obj_pair[1] if len(obj_pair) > 1 else obj_pair[0],
            "location_a": loc_triple[0],
            "location_b": loc_triple[1] if len(loc_triple) > 1 else loc_triple[0],
            "location_c": loc_triple[2] if len(loc_triple) > 2 else loc_triple[0],
        }

        key = f"{domain}|{pair[0]}|{pair[1]}|{obj_pair[0]}|{loc_triple[0]}"
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
        for cat in SPATIAL_CATEGORIES
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
