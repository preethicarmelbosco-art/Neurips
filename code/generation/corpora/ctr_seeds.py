"""Seed scenario skeleton generation for CTR-CC contrastive pairs.

Generates 2000 unique scenario skeletons by combining:
  - 10 professional domains
  - System names, actors, components, failure events, locations, timeframes
Then stratifies across 6 categories x 10 domains x 3 difficulties = 180 cells.
"""

import random

DOMAINS = [
    "engineering",
    "medicine",
    "cybersecurity",
    "law",
    "finance",
    "ecology",
    "aerospace",
    "supply_chain",
    "energy_systems",
    "pharmacology",
]

CAUSAL_CATEGORIES = [
    "counterfactual_intervention",
    "causal_chain_tracing",
    "sufficiency_vs_necessity",
    "common_cause_confounding",
    "preventive_causation",
    "overdetermination",
]

DIFFICULTY_LEVELS = [
    "straightforward",
    "ambiguous",
    "adversarial",
]

# --- Per-domain skeleton components ---

SYSTEM_NAMES = {
    "engineering": [
        "the municipal water treatment plant",
        "the highway overpass reinforcement project",
        "the industrial conveyor belt system",
        "the high-rise elevator shaft assembly",
        "the offshore drilling platform",
    ],
    "medicine": [
        "the regional trauma center",
        "the oncology clinical trial program",
        "the neonatal intensive care unit",
        "the transplant coordination service",
        "the outpatient chemotherapy suite",
    ],
    "cybersecurity": [
        "the enterprise identity management platform",
        "the financial transaction monitoring system",
        "the government classified network gateway",
        "the hospital electronic health record system",
        "the cloud infrastructure orchestration layer",
    ],
    "law": [
        "the class-action product liability case",
        "the environmental contamination lawsuit",
        "the patent infringement tribunal",
        "the corporate fraud investigation",
        "the wrongful termination arbitration",
    ],
    "finance": [
        "the sovereign bond trading desk",
        "the algorithmic market-making engine",
        "the pension fund rebalancing system",
        "the derivatives clearinghouse",
        "the retail mortgage underwriting pipeline",
    ],
    "ecology": [
        "the coastal wetland restoration project",
        "the boreal forest wildfire management system",
        "the coral reef monitoring network",
        "the migratory bird tracking program",
        "the invasive species containment zone",
    ],
    "aerospace": [
        "the low-earth-orbit satellite constellation",
        "the commercial turbofan engine assembly line",
        "the re-entry heat shield qualification test",
        "the autonomous drone navigation system",
        "the crew life-support module",
    ],
    "supply_chain": [
        "the semiconductor chip fabrication pipeline",
        "the cold-chain pharmaceutical distribution network",
        "the just-in-time automotive parts corridor",
        "the global container shipping routing system",
        "the agricultural commodity storage depot",
    ],
    "energy_systems": [
        "the regional electrical grid interconnect",
        "the offshore wind farm control system",
        "the nuclear reactor cooling loop",
        "the solar microgrid battery storage array",
        "the natural gas pipeline compressor station",
    ],
    "pharmacology": [
        "the phase-III multi-site drug trial",
        "the hospital antibiotic stewardship program",
        "the pediatric dose-adjustment protocol",
        "the opioid tapering management system",
        "the immunosuppressant interaction monitoring suite",
    ],
}

ACTORS = {
    "engineering": [
        "the lead structural engineer", "the site foreman",
        "the safety inspector", "the project manager",
        "the maintenance technician",
    ],
    "medicine": [
        "the attending physician", "the charge nurse",
        "the pharmacist", "the radiologist",
        "the surgical resident",
    ],
    "cybersecurity": [
        "the SOC analyst", "the penetration tester",
        "the CISO", "the incident responder",
        "the systems administrator",
    ],
    "law": [
        "the plaintiff's counsel", "the expert witness",
        "the presiding judge", "the defense attorney",
        "the forensic accountant",
    ],
    "finance": [
        "the head trader", "the risk manager",
        "the compliance officer", "the portfolio analyst",
        "the quant developer",
    ],
    "ecology": [
        "the field biologist", "the park ranger",
        "the conservation director", "the hydrologist",
        "the wildlife veterinarian",
    ],
    "aerospace": [
        "the flight test engineer", "the mission controller",
        "the propulsion specialist", "the quality assurance lead",
        "the avionics technician",
    ],
    "supply_chain": [
        "the logistics coordinator", "the warehouse manager",
        "the procurement officer", "the demand planner",
        "the customs broker",
    ],
    "energy_systems": [
        "the grid operator", "the turbine technician",
        "the control room supervisor", "the electrical engineer",
        "the safety compliance officer",
    ],
    "pharmacology": [
        "the clinical pharmacologist", "the trial coordinator",
        "the data safety monitor", "the dispensing pharmacist",
        "the regulatory affairs specialist",
    ],
}

COMPONENTS = {
    "engineering": [
        "the pressure relief valve", "the load-bearing column",
        "the hydraulic pump", "the foundation anchor bolts",
        "the backup generator",
    ],
    "medicine": [
        "the ventilator circuit", "the infusion pump",
        "the defibrillator", "the blood gas analyzer",
        "the central line catheter",
    ],
    "cybersecurity": [
        "the firewall rule set", "the SSL certificate chain",
        "the two-factor authentication module", "the intrusion detection sensor",
        "the privileged access management vault",
    ],
    "law": [
        "the chain-of-custody log", "the sworn deposition transcript",
        "the surveillance footage", "the forensic audit report",
        "the expert testimony",
    ],
    "finance": [
        "the margin call trigger", "the volatility model",
        "the stop-loss algorithm", "the collateral valuation engine",
        "the liquidity buffer",
    ],
    "ecology": [
        "the water quality sensor", "the wildlife camera trap",
        "the sediment filter barrier", "the fish ladder passage",
        "the controlled burn perimeter",
    ],
    "aerospace": [
        "the attitude control thruster", "the landing gear actuator",
        "the oxygen recirculation pump", "the radar altimeter",
        "the thermal protection tile",
    ],
    "supply_chain": [
        "the RFID tracking tag", "the temperature logger",
        "the customs clearance document", "the safety stock buffer",
        "the routing optimization module",
    ],
    "energy_systems": [
        "the circuit breaker", "the transformer tap changer",
        "the battery management system", "the turbine pitch controller",
        "the frequency regulation relay",
    ],
    "pharmacology": [
        "the dosing algorithm", "the drug interaction database",
        "the adverse event reporting form", "the pharmacokinetic model",
        "the controlled substance ledger",
    ],
}

FAILURE_EVENTS = [
    "catastrophic system failure",
    "cascading component breakdown",
    "undetected degradation leading to crisis",
    "simultaneous independent failures",
    "delayed response to early warning signals",
    "misdiagnosis triggering incorrect intervention",
    "communication breakdown between teams",
    "protocol override resulting in adverse outcome",
    "single-point-of-failure activation",
    "environmental trigger exceeding design parameters",
]

LOCATIONS = [
    "in the control room during shift handover",
    "at the primary operations center",
    "in the field during routine inspection",
    "at the emergency response staging area",
    "in the executive briefing room",
    "at the testing laboratory",
    "in the server room during scheduled maintenance",
    "at the regulatory hearing",
    "in the remote monitoring station",
    "at the warehouse loading dock",
]

TIMEFRAMES = [
    "during the overnight maintenance window",
    "at peak operational load",
    "in the final hour before the deadline",
    "during the annual compliance audit",
    "forty-eight hours after the initial incident",
    "during the transition between legacy and new systems",
    "in the first week of deployment",
    "during a severe weather event",
    "at the end of a twelve-hour shift rotation",
    "during the quarterly stress test",
]


def generate_scenario_skeletons(n: int = 2000, seed: int = 42) -> list[dict]:
    """Generate n unique scenario skeletons.

    Each skeleton is a dict with keys:
        scenario_id, domain, system_name, actor_a, actor_b,
        component_a, component_b, failure_event, location, timeframe
    """
    rng = random.Random(seed)
    skeletons: list[dict] = []
    seen_keys: set[str] = set()

    idx = 0
    attempts = 0
    max_attempts = n * 20  # safety valve

    while len(skeletons) < n and attempts < max_attempts:
        attempts += 1
        domain = rng.choice(DOMAINS)
        system_name = rng.choice(SYSTEM_NAMES[domain])
        actors = rng.sample(ACTORS[domain], 2)
        components = rng.sample(COMPONENTS[domain], 2)
        failure_event = rng.choice(FAILURE_EVENTS)
        location = rng.choice(LOCATIONS)
        timeframe = rng.choice(TIMEFRAMES)

        key = f"{domain}|{system_name}|{actors[0]}|{actors[1]}|{failure_event}"
        if key in seen_keys:
            continue
        seen_keys.add(key)

        skeletons.append({
            "scenario_id": f"ctr_{idx:05d}",
            "domain": domain,
            "system_name": system_name,
            "actor_a": actors[0],
            "actor_b": actors[1],
            "component_a": components[0],
            "component_b": components[1],
            "failure_event": failure_event,
            "location": location,
            "timeframe": timeframe,
        })
        idx += 1

    return skeletons[:n]


def generate_stratified_seeds(
    n_total: int = 10000, n_skeletons: int = 2000, seed: int = 42
) -> list[dict]:
    """Generate category-stratified seed prompts.

    Distributes n_total pairs across 6 categories x 10 domains x 3 difficulties
    = 180 cells (~56 pairs per cell).

    Returns list of dicts with skeleton fields plus category and difficulty.
    """
    rng = random.Random(seed)
    skeletons = generate_scenario_skeletons(n=n_skeletons, seed=seed)

    # Group skeletons by domain for balanced assignment
    by_domain: dict[str, list[dict]] = {d: [] for d in DOMAINS}
    for s in skeletons:
        by_domain[s["domain"]].append(s)

    cells = [
        (cat, domain, diff)
        for cat in CAUSAL_CATEGORIES
        for domain in DOMAINS
        for diff in DIFFICULTY_LEVELS
    ]
    per_cell = -(-n_total // len(cells))  # ceil division

    seeds = []
    for cat, domain, diff in cells:
        domain_skeletons = by_domain[domain]
        for i in range(per_cell):
            if domain_skeletons:
                skeleton = rng.choice(domain_skeletons).copy()
            else:
                # Fallback for small N: pick any skeleton, override domain
                skeleton = rng.choice(skeletons).copy()
                skeleton["domain"] = domain
            skeleton["category"] = cat
            skeleton["difficulty"] = diff
            seeds.append(skeleton)

    rng.shuffle(seeds)
    return seeds[:n_total]
