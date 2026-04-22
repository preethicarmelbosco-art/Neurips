"""Prompt templates for CORE-MATH contrastive pair generation.

Formal mathematical proof structure vs. intuitive conceptual understanding:
  - D_target: step-by-step proofs with formal logic markers
  - D_retain: intuitive/conceptual descriptions with zero formal notation
"""

CATEGORY_DEFINITIONS = {
    "algebra": (
        "ALGEBRA: Proofs involving algebraic structures, equations, inequalities, "
        "polynomials, matrices, and abstract algebra (groups, rings, fields). "
        "The target proof should manipulate symbolic expressions using algebraic "
        "identities, substitution, and formal algebraic reasoning."
    ),
    "number_theory": (
        "NUMBER THEORY: Proofs about properties of integers — divisibility, "
        "primes, congruences, Diophantine equations, arithmetic functions. "
        "The target proof should use modular arithmetic, induction over integers, "
        "or factorization arguments with formal logical steps."
    ),
    "combinatorics": (
        "COMBINATORICS: Proofs involving counting, enumeration, graph theory, "
        "and discrete structures. The target proof should use bijections, "
        "recurrences, generating functions, or pigeonhole arguments with "
        "formal step-by-step reasoning."
    ),
    "geometry": (
        "GEOMETRY: Proofs about spatial relationships, measurements, and "
        "geometric properties — Euclidean, projective, or differential geometry. "
        "The target proof should use coordinate calculations, angle chasing, "
        "or axiomatic deductions with formal geometric notation."
    ),
    "calculus": (
        "CALCULUS: Proofs involving limits, derivatives, integrals, series, "
        "and multivariable analysis. The target proof should use epsilon-delta "
        "arguments, integration techniques, or convergence tests with formal "
        "analytical reasoning."
    ),
    "probability": (
        "PROBABILITY: Proofs about random variables, distributions, expectations, "
        "and stochastic processes. The target proof should use axioms of "
        "probability, conditioning arguments, or moment computations with "
        "formal probabilistic notation."
    ),
}

DIFFICULTY_DESCRIPTIONS = {
    "textbook": (
        "TEXTBOOK: A standard proof suitable for an undergraduate course. "
        "Clear logical flow, 2-3 main steps. The theorem is well-known and "
        "the proof technique is standard. No clever tricks required."
    ),
    "competition": (
        "COMPETITION: A proof requiring a clever insight or non-obvious "
        "technique, suitable for a math olympiad or Putnam exam. 4-6 main steps. "
        "May involve an elegant substitution, an unexpected bijection, or a "
        "creative application of a standard theorem."
    ),
    "research_level": (
        "RESEARCH LEVEL: A proof involving a novel approach, deep connections "
        "between subfields, or sophisticated machinery. 6+ main steps. May "
        "invoke advanced theorems, construct auxiliary objects, or use techniques "
        "from multiple areas of mathematics."
    ),
}

FORMAL_NOTATION_BLACKLIST = (
    "=, <, >, <=, >=, !=, therefore, hence, thus, QED, it follows, "
    "by definition, by assumption, by hypothesis, we have, we get, "
    "we obtain, implies that, if and only if, proof, Step 1, Step 2, "
    "let x, let n, suppose that, assume that"
)

SYSTEM_PROMPT = (
    "You are generating training data for a mechanistic interpretability "
    "experiment studying mathematical reasoning circuits in language models.\n\n"
    "You will be given a mathematical concept (category, topic, notation context) "
    "and a difficulty level. You must output two strictly separated texts:\n\n"
    "1. The 'target_proof' (Formal Step-by-Step Mathematical Proof):\n"
    "   - Present a rigorous mathematical proof of a theorem or result\n"
    "   - Use formal logic markers: 'Let x =', 'therefore', 'it follows that', "
    "'by definition', 'by hypothesis', 'QED'\n"
    "   - Include numbered steps: 'Step 1.', 'Step 2.', etc.\n"
    "   - Include equations with '=' and mathematical symbols\n"
    "   - State the theorem/result clearly before the proof\n"
    "   - Match the specified difficulty level\n"
    "   - Be 150-500 words\n\n"
    "2. The 'retain_intuition' (Intuitive Conceptual Description):\n"
    "   - Describe the EXACT SAME mathematical concept/theorem/result\n"
    "   - Use analogies, metaphors, and plain language\n"
    "   - Use phrases like 'the key idea is...', 'intuitively...', "
    "'think of it as...', 'imagine that...'\n"
    "   - Explain WHY the result is true at a conceptual level\n"
    "   - STRICTLY FORBIDDEN — zero instances of: {blacklist}\n"
    "   - No equations, no '=' signs, no formal logic markers, no step numbering\n"
    "   - No mathematical symbols like + - * / ^ or Greek letters used as variables\n"
    "   - Be 150-500 words\n\n"
    "CRITICAL CONSTRAINT (bijectivity): Both texts must describe the IDENTICAL "
    "mathematical concept — the same theorem, the same result. The ONLY difference "
    "is that target_proof uses formal proof structure while retain_intuition uses "
    "intuitive/conceptual language.\n\n"
    "INTUITION TEST for retain_intuition: Could someone with no math training "
    "understand the general idea? If yes, it passes. If it requires knowing "
    "mathematical notation, it fails.\n\n"
    "{{category_definition}}\n\n"
    "{{difficulty_description}}"
).format(blacklist=FORMAL_NOTATION_BLACKLIST)


# ── Gold-Standard Examples (one per category at textbook difficulty) ────────

GOLD_EXAMPLES = {
    # ── ALGEBRA ──────────────────────────────────────────────────────────────
    "algebra": {
        "textbook": [
            {
                "target_proof": (
                    "Theorem: For a quadratic equation ax^2 + bx + c = 0 with "
                    "a != 0, the solutions are x = (-b +/- sqrt(b^2 - 4ac)) / (2a).\n\n"
                    "Proof:\n"
                    "Step 1. Divide both sides by a: x^2 + (b/a)x + c/a = 0.\n"
                    "Step 2. Rearrange: x^2 + (b/a)x = -c/a.\n"
                    "Step 3. Complete the square by adding (b/(2a))^2 to both sides: "
                    "(x + b/(2a))^2 = b^2/(4a^2) - c/a = (b^2 - 4ac)/(4a^2).\n"
                    "Step 4. Take the square root of both sides: "
                    "x + b/(2a) = +/- sqrt(b^2 - 4ac)/(2a).\n"
                    "Step 5. Therefore x = (-b +/- sqrt(b^2 - 4ac))/(2a). QED."
                ),
                "retain_intuition": (
                    "The quadratic formula gives us a universal recipe for solving "
                    "any equation with a squared term. The key idea is to rearrange "
                    "the equation into a perfect square plus some remainder, much like "
                    "completing a jigsaw puzzle where you know what the finished "
                    "picture looks like and work backward to find the missing piece. "
                    "Think of it as unwrapping layers: first you isolate the squared "
                    "part, then you peel away the linear term by balancing both sides, "
                    "and finally you take a square root to reveal the two possible "
                    "answers. Those two answers arise because a square root always has "
                    "two directions — one positive and one negative — like two paths "
                    "diverging from a central point. The discriminant, the expression "
                    "under the square root, tells you whether the paths are real and "
                    "distinct, real and identical, or complex. Intuitively, it measures "
                    "how much room there is between the curve and the horizontal axis."
                ),
            },
        ],
        "competition": [
            {
                "target_proof": (
                    "Theorem (AM-GM Inequality): For non-negative reals a_1, ..., a_n, "
                    "we have (a_1 + a_2 + ... + a_n)/n >= (a_1 * a_2 * ... * a_n)^(1/n), "
                    "with equality if and only if a_1 = a_2 = ... = a_n.\n\n"
                    "Proof (by induction, Cauchy's forward-backward method):\n"
                    "Step 1. Base case n = 2: We must show (a + b)/2 >= sqrt(a*b). "
                    "By definition, (a - b)^2 >= 0, which expands to "
                    "a^2 - 2ab + b^2 >= 0. It follows that a + b >= 2*sqrt(a*b), "
                    "hence (a + b)/2 >= sqrt(a*b).\n"
                    "Step 2. Forward step: Assume true for n = k. We prove it for n = 2k "
                    "by pairing terms and applying the n = 2 case iteratively.\n"
                    "Step 3. Backward step: Assume true for n = k. We prove it for "
                    "n = k - 1 by setting a_k = (a_1 + ... + a_{k-1})/(k-1) and "
                    "applying the n = k case. Therefore the inequality holds for all "
                    "positive integers n. QED."
                ),
                "retain_intuition": (
                    "The arithmetic mean-geometric mean inequality captures a simple "
                    "but profound idea: spreading values evenly always produces the "
                    "largest product. Think of it as distributing water equally among "
                    "identical containers — any uneven distribution wastes potential. "
                    "Intuitively, the average of a collection of numbers is always at "
                    "least as large as the number you would get by multiplying them "
                    "all together and taking the appropriate root. The key idea is "
                    "that imbalance reduces the product. Imagine two farmers with "
                    "fields of different sizes: if they pool their land and split it "
                    "equally, their combined harvest improves because the smaller "
                    "field was the bottleneck. The further apart the numbers are, the "
                    "bigger the gap between average and product-root. Only when every "
                    "number is the same do they coincide perfectly, because there is "
                    "no imbalance left to exploit."
                ),
            },
        ],
        "research_level": [
            {
                "target_proof": (
                    "Theorem: Every finite group G of order p*q, where p < q are "
                    "distinct primes with p not dividing (q - 1), is cyclic.\n\n"
                    "Proof:\n"
                    "Step 1. By Sylow's theorem, let n_q denote the number of "
                    "Sylow q-subgroups. Then n_q divides p and n_q = 1 (mod q).\n"
                    "Step 2. Since p < q, the only divisor of p that is congruent "
                    "to 1 mod q is 1. Therefore n_q = 1, so the Sylow q-subgroup "
                    "Q is normal in G.\n"
                    "Step 3. Similarly, let n_p denote the number of Sylow p-subgroups. "
                    "Then n_p divides q and n_p = 1 (mod p). The divisors of q are "
                    "1 and q. If n_p = q, then q = 1 (mod p), i.e., p divides (q - 1), "
                    "contradicting our hypothesis.\n"
                    "Step 4. Therefore n_p = 1, so the Sylow p-subgroup P is also "
                    "normal in G.\n"
                    "Step 5. Since P and Q are normal, P intersect Q = {e} (by "
                    "Lagrange's theorem, since gcd(p, q) = 1), and |P*Q| = p*q = |G|. "
                    "It follows that G = P x Q, a direct product of cyclic groups of "
                    "coprime order. Therefore G is cyclic. QED."
                ),
                "retain_intuition": (
                    "When a group has a size that is the product of two distinct "
                    "primes, and those primes do not interact in a specific divisibility "
                    "way, the group is forced to be cyclic — meaning it looks like a "
                    "simple clock with that many positions. The key idea is that each "
                    "prime contributes exactly one building block, and those building "
                    "blocks cannot interfere with each other. Think of it as two "
                    "independent gears of different prime sizes: because neither gear's "
                    "size evenly divides one less than the other's size, there is no "
                    "way for them to mesh in a complicated pattern. They are forced to "
                    "spin independently. Intuitively, the group has no room for exotic "
                    "structure because the prime factors are too incompatible to "
                    "produce anything but the simplest arrangement. The whole group is "
                    "just the two independent prime-sized pieces bolted together, and "
                    "since those pieces have coprime sizes, the combination behaves "
                    "like a single rotating wheel."
                ),
            },
        ],
    },
    # ── NUMBER THEORY ────────────────────────────────────────────────────────
    "number_theory": {
        "textbook": [
            {
                "target_proof": (
                    "Theorem (Fermat's Little Theorem): If p is prime and "
                    "gcd(a, p) = 1, then a^(p-1) = 1 (mod p).\n\n"
                    "Proof:\n"
                    "Step 1. Consider the set S = {1*a, 2*a, 3*a, ..., (p-1)*a} "
                    "taken modulo p.\n"
                    "Step 2. We claim S is a permutation of {1, 2, ..., p-1} mod p. "
                    "Suppose i*a = j*a (mod p) for 1 <= i, j <= p-1. Then "
                    "p divides (i - j)*a. Since gcd(a, p) = 1, it follows that "
                    "p divides (i - j), hence i = j.\n"
                    "Step 3. Therefore the product of elements of S equals the product "
                    "of {1, 2, ..., p-1} mod p: "
                    "(1*a)(2*a)...(p-1)*a = 1*2*...*(p-1) (mod p).\n"
                    "Step 4. This gives a^(p-1) * (p-1)! = (p-1)! (mod p). "
                    "Since gcd((p-1)!, p) = 1, we divide both sides by (p-1)!.\n"
                    "Step 5. Therefore a^(p-1) = 1 (mod p). QED."
                ),
                "retain_intuition": (
                    "Fermat's little theorem says that when you raise a number to a "
                    "power related to a prime, the result cycles back to one in that "
                    "prime's world of remainders. The key idea is a reshuffling "
                    "argument: if you take all the nonzero remainders and multiply "
                    "each by the same number, you just scramble them into a different "
                    "order without losing or gaining any. Think of it like dealing a "
                    "deck of cards to a circle of friends and then rotating everyone "
                    "one seat over — you still have the same cards at the table, just "
                    "in different hands. Because the scrambled collection is the same "
                    "as the original, the extra factor you introduced must contribute "
                    "nothing overall. Intuitively, the prime acts as a perfect "
                    "recycler: no matter what multiplier you use, the remainders "
                    "always form a complete set, and the multiplier's total "
                    "contribution washes out to one."
                ),
            },
        ],
        "competition": [
            {
                "target_proof": (
                    "Theorem (Chinese Remainder Theorem): If m_1, m_2, ..., m_k "
                    "are pairwise coprime positive integers and a_1, ..., a_k are "
                    "any integers, then the system x = a_i (mod m_i) for all i has "
                    "a unique solution modulo M = m_1 * m_2 * ... * m_k.\n\n"
                    "Proof:\n"
                    "Step 1. For each i, let M_i = M / m_i. By construction, "
                    "gcd(M_i, m_i) = 1.\n"
                    "Step 2. By Bezout's identity, there exists y_i such that "
                    "M_i * y_i = 1 (mod m_i).\n"
                    "Step 3. Define x = a_1*M_1*y_1 + a_2*M_2*y_2 + ... + a_k*M_k*y_k.\n"
                    "Step 4. For each j, note that M_i = 0 (mod m_j) when i != j, "
                    "so x = a_j*M_j*y_j = a_j*1 = a_j (mod m_j). Therefore x "
                    "satisfies all congruences.\n"
                    "Step 5. Uniqueness: if x' is another solution, then M divides "
                    "(x - x') since each m_i divides (x - x') and the m_i are "
                    "pairwise coprime. Therefore the solution is unique mod M. QED."
                ),
                "retain_intuition": (
                    "The Chinese Remainder Theorem says that if you know the "
                    "remainders of a number when divided by several coprime divisors, "
                    "you can reconstruct the number uniquely within a certain range. "
                    "Think of it as labeling a person by multiple independent traits — "
                    "their birthday month, their house number, and their shoe size — "
                    "where no two traits share common factors. Knowing all the labels "
                    "pins down exactly one person in the population. The key idea is "
                    "that coprime divisors partition the number line into independent "
                    "channels, like radio frequencies that do not interfere. "
                    "Intuitively, each remainder gives you a partial view through a "
                    "different lens, and because the lenses are independent, combining "
                    "them produces a complete picture. Imagine clocks of different "
                    "coprime sizes all starting at zero: they synchronize at exactly "
                    "one moment within their combined cycle, and that moment is your "
                    "unique answer."
                ),
            },
        ],
        "research_level": [
            {
                "target_proof": (
                    "Theorem (Quadratic Reciprocity): For distinct odd primes p and q, "
                    "(p/q)(q/p) = (-1)^{((p-1)/2)((q-1)/2)}, where (a/b) denotes "
                    "the Legendre symbol.\n\n"
                    "Proof sketch (Gauss's third proof via Gauss sums):\n"
                    "Step 1. Define the Gauss sum g = sum_{a=0}^{p-1} (a/p) * "
                    "zeta^a, where zeta = e^{2*pi*i/p}.\n"
                    "Step 2. By direct computation, g^2 = (-1)^{(p-1)/2} * p. "
                    "This is established by expanding g^2 and using properties of "
                    "the Legendre symbol.\n"
                    "Step 3. Compute g^q (mod q) in two ways. On one hand, "
                    "g^q = (g^2)^{(q-1)/2} * g = ((-1)^{(p-1)/2} * p)^{(q-1)/2} * g.\n"
                    "Step 4. On the other hand, by Frobenius, g^q = "
                    "sum (a/p) * zeta^{a*q} = (q/p) * g (after re-indexing).\n"
                    "Step 5. Equating and simplifying yields "
                    "(q/p) = (-1)^{((p-1)/2)((q-1)/2)} * (p/q). "
                    "Therefore (p/q)(q/p) = (-1)^{((p-1)/2)((q-1)/2)}. QED."
                ),
                "retain_intuition": (
                    "Quadratic reciprocity reveals a deep symmetry between two "
                    "primes: whether the first prime is a perfect square in the "
                    "second prime's world is tightly linked to whether the second "
                    "is a perfect square in the first's world. The key idea is that "
                    "primes have a conversation, and the answer each gives about the "
                    "other is almost always the same — they only disagree when both "
                    "primes belong to a particular club. Think of it as two mirrors "
                    "facing each other: what one reflects about the other is nearly "
                    "identical, with a single twist determined by their shapes. "
                    "Intuitively, the residue structure of prime number systems is "
                    "so rigid that knowing one side of the relationship almost "
                    "completely determines the other. This is surprising because there "
                    "is no obvious reason why two separate prime worlds should be "
                    "aware of each other at all, yet they are linked by this elegant "
                    "reciprocal law."
                ),
            },
        ],
    },
    # ── COMBINATORICS ────────────────────────────────────────────────────────
    "combinatorics": {
        "textbook": [
            {
                "target_proof": (
                    "Theorem (Pigeonhole Principle): If n + 1 objects are placed "
                    "into n boxes, then at least one box contains two or more objects.\n\n"
                    "Proof (by contradiction):\n"
                    "Step 1. Assume, for contradiction, that every box contains at "
                    "most one object.\n"
                    "Step 2. Then the total number of objects <= n * 1 = n.\n"
                    "Step 3. But we have n + 1 objects, and n + 1 > n. This is a "
                    "contradiction.\n"
                    "Step 4. Therefore our assumption was false, and at least one "
                    "box must contain >= 2 objects. QED."
                ),
                "retain_intuition": (
                    "The pigeonhole principle captures the common-sense observation "
                    "that if you have more items than containers, at least one "
                    "container must hold multiple items. Think of it as trying to "
                    "seat guests at a dinner party where there are more guests than "
                    "chairs — no matter how you arrange them, at least one chair "
                    "ends up shared. The key idea is that perfect spreading is "
                    "impossible when demand exceeds supply. Intuitively, you cannot "
                    "distribute surplus without creating a pile somewhere. Despite "
                    "its simplicity, this principle is remarkably powerful: many "
                    "deep results in mathematics boil down to recognizing that some "
                    "collection is too large to fit neatly into the available slots, "
                    "forcing a collision that reveals hidden structure."
                ),
            },
        ],
        "competition": [
            {
                "target_proof": (
                    "Theorem (Vandermonde's Identity): C(m + n, r) = "
                    "sum_{k=0}^{r} C(m, k) * C(n, r - k).\n\n"
                    "Proof (combinatorial / double counting):\n"
                    "Step 1. Consider a set A of m elements and a disjoint set B "
                    "of n elements. The left side C(m + n, r) counts the number of "
                    "ways to choose r elements from A union B.\n"
                    "Step 2. Any such selection takes exactly k elements from A "
                    "and r - k elements from B, for some 0 <= k <= r.\n"
                    "Step 3. The number of ways to choose k from A is C(m, k), "
                    "and independently, the number of ways to choose r - k from B "
                    "is C(n, r - k).\n"
                    "Step 4. By the multiplication principle, each value of k "
                    "contributes C(m, k) * C(n, r - k) selections.\n"
                    "Step 5. Summing over all valid k gives the right side. "
                    "Therefore C(m + n, r) = sum_{k=0}^{r} C(m, k) * C(n, r - k). QED."
                ),
                "retain_intuition": (
                    "Vandermonde's identity says that choosing a team from a mixed "
                    "group is the same as summing over all the ways to split the team "
                    "between two subgroups. Think of it as picking players for a "
                    "pickup game from two neighborhoods: you decide how many come from "
                    "each side, pick independently within each, and then add up all "
                    "the possibilities. The key idea is decomposition — any selection "
                    "from a union naturally splits according to where its members "
                    "originate. Intuitively, you are slicing one big counting problem "
                    "into smaller, independent pieces and reassembling them. Each "
                    "slice corresponds to a different balance between the two groups, "
                    "and since the slices cover every possible balance without overlap, "
                    "the total is exact."
                ),
            },
        ],
        "research_level": [
            {
                "target_proof": (
                    "Theorem (Burnside's Lemma): The number of distinct objects "
                    "under a group action G on a set X is "
                    "|X/G| = (1/|G|) * sum_{g in G} |Fix(g)|, where Fix(g) = "
                    "{x in X : g*x = x}.\n\n"
                    "Proof:\n"
                    "Step 1. Count the set S = {(g, x) in G x X : g*x = x} in "
                    "two ways.\n"
                    "Step 2. Summing over g: |S| = sum_{g in G} |Fix(g)|.\n"
                    "Step 3. Summing over x: |S| = sum_{x in X} |Stab(x)|, where "
                    "Stab(x) = {g in G : g*x = x}.\n"
                    "Step 4. By the orbit-stabilizer theorem, |Stab(x)| = "
                    "|G| / |Orb(x)|. Therefore |S| = sum_{x in X} |G| / |Orb(x)|.\n"
                    "Step 5. Grouping by orbits: for each orbit O, the sum over "
                    "x in O of |G|/|O| = |G|. Hence |S| = |G| * (number of orbits).\n"
                    "Step 6. Equating: |G| * |X/G| = sum_{g in G} |Fix(g)|. "
                    "Dividing by |G| gives the result. QED."
                ),
                "retain_intuition": (
                    "Burnside's lemma provides a way to count truly distinct "
                    "configurations when symmetry makes many arrangements look the "
                    "same. Think of coloring the faces of a cube: many colorings "
                    "that look different on paper become identical once you are "
                    "allowed to rotate the cube. The key idea is to average the "
                    "number of configurations that each symmetry operation leaves "
                    "unchanged. Intuitively, each rotation or reflection acts as a "
                    "filter, and only the configurations that survive all filters "
                    "are truly distinct. By checking how many colorings each symmetry "
                    "fixes in place and then averaging, you count each genuinely "
                    "different pattern exactly once. It is like asking every member "
                    "of a committee to raise their hand for each proposal they find "
                    "acceptable, then dividing the total hands by the committee "
                    "size to find how many proposals have universal support."
                ),
            },
        ],
    },
    # ── GEOMETRY ─────────────────────────────────────────────────────────────
    "geometry": {
        "textbook": [
            {
                "target_proof": (
                    "Theorem (Pythagorean Theorem): In a right triangle with legs "
                    "a, b and hypotenuse c, a^2 + b^2 = c^2.\n\n"
                    "Proof (by rearrangement):\n"
                    "Step 1. Construct a square with side length (a + b). Its area "
                    "= (a + b)^2.\n"
                    "Step 2. Place four copies of the right triangle (each with area "
                    "= (1/2)*a*b) inside the square, arranged so their hypotenuses "
                    "form an inner square with side length c.\n"
                    "Step 3. The area of the outer square = area of 4 triangles + "
                    "area of inner square: (a + b)^2 = 4*(1/2)*a*b + c^2.\n"
                    "Step 4. Expanding: a^2 + 2ab + b^2 = 2ab + c^2.\n"
                    "Step 5. Subtracting 2ab from both sides: a^2 + b^2 = c^2. "
                    "Therefore the sum of squares of the legs equals the square of "
                    "the hypotenuse. QED."
                ),
                "retain_intuition": (
                    "The Pythagorean theorem tells us that in a right triangle, the "
                    "area of a square built on the longest side exactly matches the "
                    "combined areas of squares built on the two shorter sides. The "
                    "key idea is a jigsaw argument: if you build a large frame from "
                    "the two shorter sides and fill it with copies of the triangle, "
                    "the leftover space in the middle is exactly the square on the "
                    "longest side. Think of it as a conservation law for area — the "
                    "total space is fixed, and the triangles account for everything "
                    "except that central square. Intuitively, the right angle is what "
                    "makes this balance perfect: it creates a geometric harmony where "
                    "the two directions defined by the shorter sides combine cleanly "
                    "into the diagonal direction. Without the right angle, the puzzle "
                    "pieces would not fit, and the beautiful equality would break."
                ),
            },
        ],
        "competition": [
            {
                "target_proof": (
                    "Theorem (Ptolemy's Theorem): For a cyclic quadrilateral ABCD, "
                    "AC * BD = AB * CD + AD * BC.\n\n"
                    "Proof:\n"
                    "Step 1. Let the cyclic quadrilateral ABCD be inscribed in a circle. "
                    "Choose point P on diagonal AC such that angle ABP = angle DBC.\n"
                    "Step 2. Then triangle ABP is similar to triangle DBC (by AA "
                    "similarity, since angle BAP = angle BDC as both subtend arc BC). "
                    "It follows that AB/DB = AP/DC, hence AP = AB * DC / BD.\n"
                    "Step 3. Also, triangle ABD is similar to triangle BPC (since "
                    "angle ABD = angle PBC and angle ADB = angle BCP). Therefore "
                    "AD/BC = BD/PC, hence PC = BC * BD / BD... correcting: "
                    "PC = BC * AD / BD... We get AD/BD = PC/BC, hence PC = AD*BC/BD.\n"
                    "Step 4. Since P lies on AC: AC = AP + PC = AB*CD/BD + AD*BC/BD "
                    "= (AB*CD + AD*BC)/BD.\n"
                    "Step 5. Therefore AC * BD = AB * CD + AD * BC. QED."
                ),
                "retain_intuition": (
                    "Ptolemy's theorem reveals a hidden relationship among the sides "
                    "and diagonals of any four-sided figure inscribed in a circle. "
                    "The key idea is that the circle imposes such strong constraints "
                    "on the angles that the six distances between vertices are not "
                    "independent — knowing five determines the sixth. Think of it as "
                    "four towns arranged on a circular lake, connected by straight "
                    "roads: the product of the two crossing roads exactly matches the "
                    "sum of the products of opposite pairs of lakeside roads. "
                    "Intuitively, the circular arrangement creates a perfect balance "
                    "between diagonal reach and peripheral distances. The argument "
                    "works by finding a special point that splits one diagonal into "
                    "two segments, each governed by triangle similarity forced by the "
                    "circle's angle properties. It is this rigid angle structure that "
                    "transforms what seems like a coincidence into an inevitability."
                ),
            },
        ],
        "research_level": [
            {
                "target_proof": (
                    "Theorem (Euler's Polyhedron Formula): For any convex polyhedron, "
                    "V - E + F = 2, where V = vertices, E = edges, F = faces.\n\n"
                    "Proof (by induction on edges via edge contraction):\n"
                    "Step 1. Consider the planar graph G obtained by projecting the "
                    "polyhedron onto the plane (removing one face). We prove "
                    "V - E + F = 1 for connected planar graphs (add 1 for the "
                    "removed face).\n"
                    "Step 2. Base case: a single vertex (a tree with V = 1, E = 0, "
                    "F = 0) gives V - E + F = 1.\n"
                    "Step 3. If G has an edge e that is a bridge, removing e splits "
                    "V - E + F into two components. By induction, each satisfies "
                    "V_i - E_i + F_i = 1. Summing and accounting for the shared "
                    "removal: V - E + F = 1.\n"
                    "Step 4. If G has an edge e that is not a bridge, removing e "
                    "merges two faces into one: V stays, E decreases by 1, F decreases "
                    "by 1. Hence V - E + F is unchanged. By induction, the result "
                    "holds.\n"
                    "Step 5. Therefore V - E + F = 1 for the planar graph, and "
                    "V - E + F = 2 for the polyhedron. QED."
                ),
                "retain_intuition": (
                    "Euler's polyhedron formula expresses a topological truth: for "
                    "any convex solid, if you count the corners, subtract the edges, "
                    "and add the faces, you always get two. The key idea is that this "
                    "number is a topological invariant — it does not change when you "
                    "stretch, flatten, or deform the shape, as long as you do not "
                    "tear or glue. Think of it as a conservation law for connectivity: "
                    "adding an edge either splits a face or connects a new vertex, "
                    "and each operation preserves the magic number. Intuitively, "
                    "imagine deflating a soccer ball onto a table — it becomes a flat "
                    "network of lines and regions, but the vertex-edge-face count "
                    "remains the same. The formula reveals that the shape of a solid "
                    "is governed not by its geometry but by its combinatorial skeleton, "
                    "a deep insight that launched the field of topology."
                ),
            },
        ],
    },
    # ── CALCULUS ──────────────────────────────────────────────────────────────
    "calculus": {
        "textbook": [
            {
                "target_proof": (
                    "Theorem (Fundamental Theorem of Calculus, Part 1): If f is "
                    "continuous on [a, b] and F(x) = integral from a to x of f(t) dt, "
                    "then F'(x) = f(x) for all x in (a, b).\n\n"
                    "Proof:\n"
                    "Step 1. By definition of the derivative, "
                    "F'(x) = lim_{h->0} (F(x+h) - F(x)) / h.\n"
                    "Step 2. We have F(x+h) - F(x) = integral from x to x+h of "
                    "f(t) dt.\n"
                    "Step 3. By the Mean Value Theorem for integrals, there exists "
                    "c in [x, x+h] such that integral from x to x+h of f(t) dt = "
                    "f(c) * h.\n"
                    "Step 4. Therefore (F(x+h) - F(x)) / h = f(c).\n"
                    "Step 5. As h -> 0, c -> x (since c is between x and x+h). "
                    "By continuity of f, f(c) -> f(x). Therefore F'(x) = f(x). QED."
                ),
                "retain_intuition": (
                    "The fundamental theorem of calculus bridges two seemingly "
                    "different operations: accumulation and rate of change. The key "
                    "idea is that if you build up a running total by sweeping across "
                    "a curve, the speed at which that total grows at any moment is "
                    "exactly the height of the curve at that point. Think of it as "
                    "filling a bathtub where the water flow rate varies over time — "
                    "the total water in the tub at any instant changes at exactly the "
                    "current flow rate. Intuitively, accumulation and instantaneous "
                    "rate are inverses of each other, like winding and unwinding a "
                    "spool of thread. This connection is what makes the two halves of "
                    "calculus — differential and integral — into a unified whole, "
                    "rather than two separate subjects that happen to share a textbook."
                ),
            },
        ],
        "competition": [
            {
                "target_proof": (
                    "Theorem (Integration by Parts): integral of u dv = u*v - "
                    "integral of v du.\n\n"
                    "Proof:\n"
                    "Step 1. Start from the product rule for differentiation: "
                    "d(u*v) = u*dv + v*du.\n"
                    "Step 2. Rearrange: u*dv = d(u*v) - v*du.\n"
                    "Step 3. Integrate both sides over the interval [a, b]: "
                    "integral of u dv = integral of d(u*v) - integral of v du.\n"
                    "Step 4. By the fundamental theorem, integral of d(u*v) = "
                    "[u*v] evaluated from a to b.\n"
                    "Step 5. Therefore integral from a to b of u dv = "
                    "[u*v] from a to b - integral from a to b of v du. QED."
                ),
                "retain_intuition": (
                    "Integration by parts is the integral version of the product "
                    "rule for derivatives, run in reverse. The key idea is that when "
                    "you need to find the area under a product of two functions, you "
                    "can transfer the difficulty from one factor to the other. Think "
                    "of it as a seesaw: one function gets differentiated (becoming "
                    "simpler) while the other gets integrated (potentially becoming "
                    "more complex), and you hope the new combination is easier to "
                    "handle. Intuitively, the boundary term accounts for the overall "
                    "accumulation of the product, while the remaining integral "
                    "captures the correction needed because both factors are changing "
                    "simultaneously. The art lies in choosing which factor to simplify "
                    "and which to complicate, so that the trade is favorable."
                ),
            },
        ],
        "research_level": [
            {
                "target_proof": (
                    "Theorem (Green's Theorem): Let D be a simply connected region "
                    "in R^2 bounded by a positively oriented, piecewise smooth, "
                    "simple closed curve C. If P and Q have continuous partial "
                    "derivatives on an open region containing D, then "
                    "oint_C (P dx + Q dy) = double_integral_D (dQ/dx - dP/dy) dA.\n\n"
                    "Proof sketch:\n"
                    "Step 1. It suffices to prove separately that "
                    "oint_C P dx = -double_integral_D (dP/dy) dA and "
                    "oint_C Q dy = double_integral_D (dQ/dx) dA.\n"
                    "Step 2. Express D as a Type I region: "
                    "g_1(x) <= y <= g_2(x), a <= x <= b.\n"
                    "Step 3. Compute double_integral_D (dP/dy) dA = "
                    "integral from a to b of [P(x, g_2(x)) - P(x, g_1(x))] dx.\n"
                    "Step 4. Compute oint_C P dx by splitting C into top and bottom "
                    "curves: the bottom traverses left-to-right giving "
                    "integral from a to b of P(x, g_1(x)) dx, and the top traverses "
                    "right-to-left giving -integral from a to b of P(x, g_2(x)) dx.\n"
                    "Step 5. Therefore oint_C P dx = integral from a to b of "
                    "[P(x, g_1(x)) - P(x, g_2(x))] dx = -double_integral_D (dP/dy) dA. "
                    "The Q component follows analogously. Combining yields Green's "
                    "theorem. QED."
                ),
                "retain_intuition": (
                    "Green's theorem connects what happens along a boundary to what "
                    "happens inside the region it encloses. The key idea is that "
                    "circulation around a loop is the accumulated local rotation "
                    "inside that loop. Think of it as a field of tiny whirlpools: "
                    "the total spinning effect you feel walking around the edge of "
                    "a pond is the sum of all the little whirlpools inside. "
                    "Intuitively, each tiny patch of the interior contributes a small "
                    "amount of swirl, and these contributions telescope along shared "
                    "boundaries between neighboring patches, canceling everywhere "
                    "except at the outer rim. It is like a crowd doing the wave in "
                    "a stadium — each person only moves with their neighbors, but the "
                    "net effect is a motion that travels around the entire perimeter. "
                    "This boundary-interior duality is one of the deepest themes in "
                    "all of mathematics, connecting local behavior to global structure."
                ),
            },
        ],
    },
    # ── PROBABILITY ──────────────────────────────────────────────────────────
    "probability": {
        "textbook": [
            {
                "target_proof": (
                    "Theorem (Bayes' Theorem): P(A|B) = P(B|A) * P(A) / P(B), "
                    "provided P(B) > 0.\n\n"
                    "Proof:\n"
                    "Step 1. By definition of conditional probability, "
                    "P(A|B) = P(A intersect B) / P(B).\n"
                    "Step 2. Similarly, P(B|A) = P(A intersect B) / P(A).\n"
                    "Step 3. From Step 2, P(A intersect B) = P(B|A) * P(A).\n"
                    "Step 4. Substituting into Step 1: "
                    "P(A|B) = P(B|A) * P(A) / P(B).\n"
                    "Step 5. Therefore, given prior P(A) and likelihood P(B|A), we "
                    "obtain the posterior P(A|B). QED."
                ),
                "retain_intuition": (
                    "Bayes' theorem provides a principled way to update your beliefs "
                    "when new evidence arrives. The key idea is that how much you "
                    "should believe something after seeing evidence depends on three "
                    "things: how strongly you believed it before, how likely the "
                    "evidence would be if your belief were true, and how likely the "
                    "evidence is in general. Think of it as a detective updating "
                    "suspicions about a suspect: finding a fingerprint at the scene "
                    "matters more if the suspect is the only person who could have "
                    "left it, and matters less if dozens of people could have. "
                    "Intuitively, rare evidence that points strongly at one "
                    "explanation provides a powerful update, while common evidence "
                    "barely shifts your opinion. The theorem formalizes what good "
                    "reasoners do naturally — weigh new information against "
                    "background expectations."
                ),
            },
        ],
        "competition": [
            {
                "target_proof": (
                    "Theorem (Chebyshev's Inequality): For a random variable X with "
                    "finite mean mu and variance sigma^2, for any k > 0, "
                    "P(|X - mu| >= k*sigma) <= 1/k^2.\n\n"
                    "Proof:\n"
                    "Step 1. By Markov's inequality applied to the non-negative "
                    "random variable (X - mu)^2: "
                    "P((X - mu)^2 >= (k*sigma)^2) <= E[(X - mu)^2] / (k*sigma)^2.\n"
                    "Step 2. By definition, E[(X - mu)^2] = sigma^2.\n"
                    "Step 3. Therefore P((X - mu)^2 >= k^2 * sigma^2) <= "
                    "sigma^2 / (k^2 * sigma^2) = 1/k^2.\n"
                    "Step 4. Since |X - mu| >= k*sigma if and only if "
                    "(X - mu)^2 >= k^2*sigma^2, it follows that "
                    "P(|X - mu| >= k*sigma) <= 1/k^2. QED."
                ),
                "retain_intuition": (
                    "Chebyshev's inequality guarantees that for any distribution, "
                    "most of the probability mass must be concentrated near the "
                    "average. The key idea is that the variance, which measures "
                    "overall spread, puts a ceiling on how much weight can be far "
                    "from the center. Think of it as a budget constraint: the "
                    "distribution has a fixed amount of spread to spend, and placing "
                    "too much weight in the tails exhausts the budget before it can "
                    "put weight anywhere else. Intuitively, the further out you look "
                    "from the average, the more expensive it is for the distribution "
                    "to place probability there, because distant values contribute "
                    "disproportionately to the variance. The remarkable thing is that "
                    "this bound holds for every distribution without exception — it "
                    "makes no assumptions about shape, symmetry, or smoothness."
                ),
            },
        ],
        "research_level": [
            {
                "target_proof": (
                    "Theorem (Law of Large Numbers, Weak): Let X_1, X_2, ... be "
                    "i.i.d. random variables with finite mean mu and finite variance "
                    "sigma^2. Let S_n = (X_1 + ... + X_n)/n. Then for any epsilon > 0, "
                    "P(|S_n - mu| >= epsilon) -> 0 as n -> infinity.\n\n"
                    "Proof:\n"
                    "Step 1. Compute E[S_n] = (1/n) * sum_{i=1}^{n} E[X_i] = "
                    "(1/n) * n * mu = mu.\n"
                    "Step 2. Compute Var(S_n) = (1/n^2) * sum_{i=1}^{n} Var(X_i) = "
                    "(1/n^2) * n * sigma^2 = sigma^2 / n.\n"
                    "Step 3. Apply Chebyshev's inequality: "
                    "P(|S_n - mu| >= epsilon) <= Var(S_n) / epsilon^2 = "
                    "sigma^2 / (n * epsilon^2).\n"
                    "Step 4. As n -> infinity, sigma^2 / (n * epsilon^2) -> 0.\n"
                    "Step 5. Therefore P(|S_n - mu| >= epsilon) -> 0. QED."
                ),
                "retain_intuition": (
                    "The law of large numbers says that as you collect more and more "
                    "observations, their average settles down and converges to the "
                    "true expected value. The key idea is that individual randomness "
                    "washes out in the aggregate. Think of it as listening to a crowd: "
                    "any single voice is unpredictable, but the average volume of a "
                    "thousand voices is remarkably stable. Intuitively, each new "
                    "observation is equally likely to pull the average up or down, and "
                    "these tugs increasingly cancel each other as the sample grows. "
                    "The variance of the average shrinks in proportion to the sample "
                    "size, so the average becomes an ever-tighter cluster around the "
                    "true center. This is the mathematical foundation for why polls, "
                    "experiments, and repeated measurements actually work — large "
                    "samples tame uncertainty."
                ),
            },
        ],
    },
}


def build_system_prompt(category: str, difficulty: str) -> str:
    """Build the full system prompt with category and difficulty injected."""
    cat_def = CATEGORY_DEFINITIONS[category]
    diff_desc = DIFFICULTY_DESCRIPTIONS[difficulty]
    return SYSTEM_PROMPT.replace("{{category_definition}}", cat_def).replace(
        "{{difficulty_description}}", diff_desc
    )


def build_user_prompt(seed: dict) -> str:
    """Build the user prompt from a concept skeleton dict."""
    examples_block = ""
    examples = GOLD_EXAMPLES.get(seed["category"], {}).get(seed["difficulty"], [])
    if examples:
        example_strs = []
        for i, ex in enumerate(examples, 1):
            example_strs.append(
                f"--- Example {i} ---\n"
                f"target_proof: {ex['target_proof']}\n\n"
                f"retain_intuition: {ex['retain_intuition']}"
            )
        examples_block = (
            "\n\nHere are gold-standard examples for this category and difficulty:\n\n"
            + "\n\n".join(example_strs)
            + "\n\n---\n\nNow generate a NEW pair for the concept below."
        )

    concept_desc = (
        f"Category: {seed['category']}\n"
        f"Topic: {seed['topic']}\n"
        f"Notation Context: {seed['notation_context']}"
    )

    return (
        f"Generate a CORE-MATH contrastive pair.\n\n"
        f"Category: {seed['category']}\n"
        f"Difficulty: {seed['difficulty']}\n\n"
        f"Concept skeleton:\n{concept_desc}"
        f"{examples_block}\n\n"
        f"Respond with ONLY a JSON object with exactly these two keys:\n"
        f'{{"target_proof": "your formal proof here", '
        f'"retain_intuition": "your intuitive description here"}}'
    )
