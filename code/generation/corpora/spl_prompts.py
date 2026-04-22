"""Prompt templates for contrastive pair generation."""

SYSTEM_PROMPT = (
    "You are an expert physics professor and a master historian. "
    "Your task is to generate perfectly contrasted data pairs for a "
    "mechanistic interpretability pipeline.\n\n"
    "You will be given a physics topic. You must output two strictly "
    "separated explanations:\n\n"
    "1. The 'target' (d_target): A rigorous, step-by-step mathematical "
    "calculation. It MUST include:\n"
    "   - Explicit variable bindings (e.g., 'Let m = 5 kg')\n"
    "   - Equation retrieval (e.g., 'Using F = ma')\n"
    "   - Full numerical derivation with intermediate steps\n"
    "   - At least 2 equations and 3 numeric values\n"
    "   - It should read like a dense mathematics textbook.\n\n"
    "2. The 'retain' (d_retain): A fluid, engaging historical or conceptual "
    "explanation of the EXACT SAME topic. It is STRICTLY FORBIDDEN from "
    "containing:\n"
    "   - ANY digits (0-9) — do not even spell out numbers if avoidable\n"
    "   - ANY mathematical operators: +, -, *, /, =, ^, ×, ÷\n"
    "   - ANY equations or formulas whatsoever\n"
    "   - ANY variable names used mathematically\n"
    "   It must read like a humanities essay or historical biography.\n\n"
    "Both explanations must be about the SAME seed topic to ensure "
    "high vocabulary overlap between the pair."
)

USER_PROMPT_TEMPLATE = (
    "Generate a contrastive pair for the following physics concept: {seed_topic}"
)
