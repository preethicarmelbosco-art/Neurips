"""Seed concept skeleton generation for CORE-MATH contrastive pairs.

Generates 2000 unique concept skeletons by combining:
  - 6 mathematical categories with 15-20 topics each
  - 10 notation contexts
Then stratifies across 6 categories x 3 difficulties = 18 cells.
"""

import random

MATH_CATEGORIES = [
    "algebra",
    "number_theory",
    "combinatorics",
    "geometry",
    "calculus",
    "probability",
]

DIFFICULTY_LEVELS = [
    "textbook",
    "competition",
    "research_level",
]

# --- Per-category mathematical topics ---

MATH_TOPICS = {
    "algebra": [
        "quadratic formula derivation",
        "systems of linear equations",
        "polynomial factoring over integers",
        "matrix determinant computation",
        "eigenvalue decomposition",
        "group theory fundamentals and subgroups",
        "ring homomorphisms and ideals",
        "linear transformations and kernel",
        "Gaussian elimination and row reduction",
        "modular arithmetic equations",
        "Vieta's formulas for polynomial roots",
        "Cauchy-Schwarz inequality",
        "AM-GM inequality",
        "rational root theorem",
        "binomial expansion over fields",
        "Jordan normal form",
        "characteristic polynomial",
        "symmetric polynomials",
    ],
    "number_theory": [
        "Euclidean algorithm for GCD",
        "Fermat's little theorem",
        "Chinese remainder theorem",
        "fundamental theorem of arithmetic",
        "Euler's totient function properties",
        "Diophantine equations solvability",
        "quadratic residues and Legendre symbol",
        "Mobius function and inversion",
        "Wilson's theorem",
        "sum of divisors function",
        "Bezout's identity",
        "order of elements modulo n",
        "primitive roots existence",
        "Hensel's lemma for p-adic lifting",
        "Bertrand's postulate",
        "Legendre's formula for factorial prime factorization",
    ],
    "combinatorics": [
        "binomial theorem proof",
        "pigeonhole principle applications",
        "inclusion-exclusion principle",
        "Catalan number recurrence",
        "generating functions for sequences",
        "Burnside's lemma for counting orbits",
        "Ramsey theory basic bounds",
        "Stirling numbers of the second kind",
        "derangement counting formula",
        "principle of double counting",
        "Vandermonde's identity",
        "stars and bars method",
        "chromatic polynomial of a graph",
        "Polya enumeration theorem",
        "Hall's marriage theorem",
        "Dilworth's theorem on antichains",
    ],
    "geometry": [
        "Pythagorean theorem proof",
        "law of cosines derivation",
        "area of regular polygons",
        "Euler's formula for polyhedra",
        "inscribed angle theorem",
        "cross-ratio invariance under projection",
        "projective duality principle",
        "Stewart's theorem",
        "Ceva's theorem on concurrent cevians",
        "Menelaus' theorem on collinear points",
        "power of a point theorem",
        "Ptolemy's theorem on cyclic quadrilaterals",
        "nine-point circle existence",
        "Euler line collinearity",
        "isoperimetric inequality in the plane",
        "trigonometric identities via unit circle",
    ],
    "calculus": [
        "fundamental theorem of calculus",
        "L'Hopital's rule for indeterminate forms",
        "Taylor series convergence and remainder",
        "integration by parts derivation",
        "Green's theorem in the plane",
        "Stokes' theorem on manifolds",
        "epsilon-delta definition of limits",
        "mean value theorem",
        "intermediate value theorem",
        "Leibniz integral rule for differentiation under the integral",
        "convergence tests for infinite series",
        "implicit function theorem",
        "Lagrange multipliers for constrained optimization",
        "divergence theorem",
        "Fubini's theorem for iterated integrals",
        "uniform convergence and interchange of limits",
    ],
    "probability": [
        "Bayes' theorem derivation",
        "law of large numbers",
        "central limit theorem",
        "conditional probability and independence",
        "Markov chain stationary distributions",
        "generating functions for discrete distributions",
        "Chebyshev's inequality",
        "law of total probability",
        "coupon collector's problem expected value",
        "random walk return probability",
        "Poisson approximation to binomial",
        "moment generating function uniqueness",
        "Jensen's inequality for convex functions",
        "birthday problem analysis",
        "inclusion-exclusion for probability",
        "optional stopping theorem for martingales",
    ],
}

NOTATION_CONTEXTS = [
    "using standard algebraic notation",
    "in matrix notation",
    "with set-builder notation",
    "using summation and product notation",
    "in modular arithmetic notation",
    "with integral and differential notation",
    "using combinatorial notation with binomial coefficients",
    "in vector notation",
    "using logical quantifiers and connectives",
    "with function composition notation",
]


def generate_concept_skeletons(n: int = 1000, seed: int = 42) -> list[dict]:
    """Generate n unique concept skeletons.

    Each skeleton is a dict with keys:
        topic_id, category, topic, notation_context
    """
    rng = random.Random(seed)
    skeletons: list[dict] = []
    seen_keys: set[str] = set()

    idx = 0
    attempts = 0
    max_attempts = n * 20  # safety valve

    while len(skeletons) < n and attempts < max_attempts:
        attempts += 1
        category = rng.choice(MATH_CATEGORIES)
        topic = rng.choice(MATH_TOPICS[category])
        notation_context = rng.choice(NOTATION_CONTEXTS)

        key = f"{category}|{topic}|{notation_context}"
        if key in seen_keys:
            continue
        seen_keys.add(key)

        skeletons.append({
            "topic_id": f"cmath_{idx:05d}",
            "category": category,
            "topic": topic,
            "notation_context": notation_context,
        })
        idx += 1

    return skeletons[:n]


def generate_stratified_seeds(
    n_total: int = 5000, n_skeletons: int = 1000, seed: int = 42
) -> list[dict]:
    """Generate category-stratified seed prompts.

    Distributes n_total pairs across 6 categories x 3 difficulties
    = 18 cells (~278 pairs per cell).

    Returns list of dicts with skeleton fields plus difficulty.
    """
    rng = random.Random(seed)
    skeletons = generate_concept_skeletons(n=n_skeletons, seed=seed)

    # Group skeletons by category for balanced assignment
    by_category: dict[str, list[dict]] = {c: [] for c in MATH_CATEGORIES}
    for s in skeletons:
        by_category[s["category"]].append(s)

    cells = [
        (cat, diff)
        for cat in MATH_CATEGORIES
        for diff in DIFFICULTY_LEVELS
    ]
    per_cell = -(-n_total // len(cells))  # ceil division

    seeds = []
    for cat, diff in cells:
        cat_skeletons = by_category[cat]
        for i in range(per_cell):
            if cat_skeletons:
                skeleton = rng.choice(cat_skeletons).copy()
            else:
                # Fallback for small N: pick any skeleton, override category
                skeleton = rng.choice(skeletons).copy()
                skeleton["category"] = cat
            skeleton["difficulty"] = diff
            seeds.append(skeleton)

    rng.shuffle(seeds)
    return seeds[:n_total]
