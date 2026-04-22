"""Seed topic generation: 50 base topics x scenario expansion -> 10k+ unique prompts."""

import random

BASE_TOPICS = [
    # Kinematics & Dynamics
    "Newton's First Law of Motion",
    "Newton's Second Law of Motion",
    "Newton's Third Law of Motion",
    "Projectile Motion",
    "Uniform Circular Motion",
    "Free Fall under Gravity",
    "Friction on an Inclined Plane",
    "Terminal Velocity",
    "Centripetal Acceleration",
    "Relative Velocity",
    # Energy & Work
    "Conservation of Energy",
    "Work-Energy Theorem",
    "Kinetic Energy",
    "Potential Energy",
    "Power and Efficiency",
    # Momentum
    "Conservation of Momentum",
    "Elastic Collisions",
    "Inelastic Collisions",
    "Impulse and Momentum Change",
    "Center of Mass Motion",
    # Rotational Mechanics
    "Torque and Angular Acceleration",
    "Moment of Inertia",
    "Angular Momentum Conservation",
    "Rotational Kinetic Energy",
    "Rolling without Slipping",
    # Gravitation
    "Newton's Law of Universal Gravitation",
    "Orbital Mechanics",
    "Escape Velocity",
    "Gravitational Potential Energy",
    "Kepler's Laws of Planetary Motion",
    # Thermodynamics
    "First Law of Thermodynamics",
    "Second Law of Thermodynamics",
    "Ideal Gas Law",
    "Carnot Engine Efficiency",
    "Heat Transfer by Conduction",
    "Entropy and Irreversibility",
    "Adiabatic Processes",
    # Waves & Oscillations
    "Simple Harmonic Motion",
    "Damped Oscillations",
    "Standing Waves on a String",
    "Doppler Effect",
    "Resonance Phenomena",
    # Electromagnetism
    "Coulomb's Law",
    "Ohm's Law and Resistance",
    "Faraday's Law of Induction",
    "Ampere's Law",
    "Lorentz Force on a Charged Particle",
    "Capacitor Charging and Discharging",
    "Electromagnetic Wave Propagation",
    # Optics & Modern
    "Snell's Law of Refraction",
    "Photoelectric Effect",
]

OBJECTS = [
    "a cannonball", "a satellite", "a pendulum", "a spring-mass system",
    "a rolling cylinder", "a falling raindrop", "a rocket", "a billiard ball",
    "an electron", "a proton", "a block on a ramp", "a swinging bob",
    "a spinning top", "a bullet", "a car", "a ball thrown upward",
    "a planet", "a comet", "a charged sphere", "a wire loop",
    "a piston", "a flywheel", "a pulley system", "a parachutist",
    "a baseball", "a roller coaster cart", "a water wave", "a sound wave",
    "a photon", "a helium balloon",
]

CONTEXTS = [
    "on an inclined plane",
    "in a vacuum",
    "near Earth's surface",
    "in deep space",
    "at the top of a cliff",
    "inside a moving train",
    "on a frictionless surface",
    "in a uniform gravitational field",
    "at terminal velocity",
    "during a head-on collision",
    "in a closed thermodynamic system",
    "across a magnetic field",
    "through a resistive medium",
    "at the equilibrium position",
    "at maximum displacement",
    "in a rotating reference frame",
    "near absolute zero temperature",
    "under constant acceleration",
    "with air resistance",
    "in a conservative force field",
]


def generate_seed_prompts(n: int = 10000, seed: int = 42) -> list[str]:
    """Generate n unique seed topic prompts by combining base topics with scenarios.

    Each prompt looks like:
        "Projectile Motion of a cannonball on an inclined plane"

    Uses deterministic seeding for reproducibility.
    """
    rng = random.Random(seed)
    prompts: set[str] = set()

    # First pass: structured combinations
    for topic in BASE_TOPICS:
        for obj in OBJECTS:
            for ctx in CONTEXTS:
                prompts.add(f"{topic} of {obj} {ctx}")

        # Also add bare topic + object and bare topic + context
        for obj in OBJECTS:
            prompts.add(f"{topic} involving {obj}")
        for ctx in CONTEXTS:
            prompts.add(f"{topic} {ctx}")

    # We now have more than enough. Shuffle and take n.
    all_prompts = sorted(prompts)  # sort for determinism before shuffle
    rng.shuffle(all_prompts)

    if len(all_prompts) < n:
        # If we somehow need more, duplicate with slight variation
        extra = []
        while len(all_prompts) + len(extra) < n:
            base = rng.choice(BASE_TOPICS)
            obj = rng.choice(OBJECTS)
            ctx = rng.choice(CONTEXTS)
            variant = f"{base} applied to {obj} {ctx}"
            if variant not in prompts:
                extra.append(variant)
                prompts.add(variant)
        all_prompts.extend(extra)

    return all_prompts[:n]
