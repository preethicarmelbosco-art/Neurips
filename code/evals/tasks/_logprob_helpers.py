"""Shared scoring helpers for log-prob choice-classification evaluators.

Centralises the two concerns that were miswritten in every held-out log-prob
evaluator (eval_legalbench, eval_bigbench_diplomacy, eval_bigtom, eval_cybermetric,
eval_scienceqa, eval_stepgame, eval_spartqa, eval_formality, eval_bigbench_strategy):

1. Prompt + continuation tokenisation. Tokenising ``prompt + " " + completion``
   as a single string and then using ``len(tokenise(prompt))`` as the boundary
   is unsafe under BPE: ``tokenise(prompt)+tokenise(X) ≠ tokenise(prompt+X)``
   in general because the last prompt token can merge across the seam. The old
   code additionally double-prepended a space when callers passed ``" label"``.
   Both effects caused ``prompt_len`` to be wrong and logits to be read from
   misaligned positions.

2. Instruct-tuned models being scored out-of-distribution. A raw completion
   prompt sent to an instruct model (expecting <|user|>...<|assistant|>) puts
   the model off-manifold, making log-probs over candidate answer tokens
   unreliable. That's the mechanism behind the label-collapse pattern
   (e.g. bigbench_diplomacy where every model scored 0.50).

Contract for ``choice_logprob``:
    - Caller passes the raw ``user_prompt`` (no chat wrapping).
    - Caller passes ``completion`` without a leading space.
    - Helper applies chat template if the tokenizer exposes one and formats
      the continuation accordingly (bare label for chat, space-prefixed label
      for raw completion).
"""
from __future__ import annotations
import torch


def wrap_chat_if_available(tokenizer, user_prompt: str) -> tuple[str, bool]:
    """Return (prompt_string, is_chat_templated).

    If the tokenizer exposes a ``chat_template``, wrap the user prompt via
    ``apply_chat_template(..., add_generation_prompt=True)``. Otherwise pass
    through.
    """
    if getattr(tokenizer, "chat_template", None):
        try:
            wrapped = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return wrapped, True
        except Exception:
            pass
    return user_prompt, False


def _format_completion(completion: str, is_chat: bool) -> str:
    """Format the continuation string for the scoring path.

    Raw completion: model emits a space before the answer after ``Answer:``,
    so prepend one. Chat-templated: template's trailing tokens (e.g.
    ``<|assistant|>\\n``) set up the boundary, so use the bare label.
    """
    c = completion.lstrip()
    return c if is_chat else f" {c}"


def choice_logprob(
    model,
    tokenizer,
    user_prompt: str,
    completion: str,
    device,
    reduce: str = "byte_mean",
    *,
    apply_chat: bool = True,
) -> float:
    """Log P(completion | prompt) under one of three normalisations.

    Tokenises prompt and completion separately and concatenates token IDs —
    avoids the BPE boundary bug. Handles BOS duplication from the second
    tokenisation. Optionally wraps the prompt with the tokenizer's chat
    template for instruct models.

    Args:
        reduce: reduction applied to the token-level log-probs of the
            continuation:
              * "byte_mean" (default, recommended for MCQA with unequal-
                length candidates): ``sum(log_p) / byte_length(completion)``.
                Mirrors lm-eval-harness ``acc_norm``. Removes the tokeniser-
                dependent bias toward the candidate that splits into fewer
                tokens, which on the previous ``mean`` reduction caused a
                label-collapse toward shorter-tokenising words like
                ``"truthful"`` vs ``"deceptive"``.
              * "mean": ``sum(log_p) / token_count``. Keeps the tokeniser-
                count bias; retained for cases where all candidates have
                equal token count (e.g. single-letter MCQA ``A/B/C/D``).
              * "sum": raw ``sum(log_p)`` — no normalisation; biased toward
                shorter completions in both tokens and bytes.
        apply_chat: apply chat template if available. Callers doing full-
            sentence scoring (e.g. COPA where both options form a sentence)
            should pass False.

    Returns ``-inf`` when the continuation tokenises to zero tokens.
    """
    if apply_chat:
        final_prompt, is_chat = wrap_chat_if_available(tokenizer, user_prompt)
    else:
        final_prompt, is_chat = user_prompt, False
    cont_str = _format_completion(completion, is_chat)

    prompt_ids = tokenizer(final_prompt, return_tensors="pt").input_ids.to(device)
    cont_ids = tokenizer(cont_str, return_tensors="pt",
                         add_special_tokens=False).input_ids.to(device)

    # Some tokenisers still emit BOS with add_special_tokens=False; strip it.
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None and cont_ids.shape[1] > 0 and cont_ids[0, 0].item() == bos:
        cont_ids = cont_ids[:, 1:]

    c_len = cont_ids.shape[1]
    if c_len == 0:
        return float("-inf")
    p_len = prompt_ids.shape[1]
    full_ids = torch.cat([prompt_ids, cont_ids], dim=1)

    with torch.no_grad():
        # use_cache=False avoids transformers 5.x DynamicCache.from_legacy_cache
        # crash on phi-3-mini, phi-3.5-mini, and other models with custom HF cache logic.
        logits = model(full_ids, use_cache=False).logits

    shift_logits = logits[:, p_len - 1 : p_len - 1 + c_len, :]
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(2, cont_ids.unsqueeze(-1)).squeeze(-1)

    total = token_lp.sum().item()
    if reduce == "byte_mean":
        # Byte length of the continuation string as the caller passed it (i.e.
        # excluding any leading space we injected for raw-completion alignment).
        # Counting bytes of the original ``completion`` argument makes the
        # normaliser invariant to our internal formatting choice.
        byte_len = max(len(completion.encode("utf-8")), 1)
        return total / byte_len
    if reduce == "mean":
        return total / c_len
    return total
