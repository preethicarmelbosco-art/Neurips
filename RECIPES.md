# Recipes — using the CogniPrimitiveBench bundle

Worked recipes that mirror Appendix A of the paper. Each recipe lists the
artefacts used, the commands to run, and what to expect as output.

Three corpora ship in the bundle but are *not analysed* in the paper:

- **COIN** (5 sub-corpora) — Contrastive Opposite/Inverse Negatives, one per
  cognitive primitive. Behavioural-opposite third pairs (`D_target` vs.
  `D_opposite`) for RepE direction-sensitive steering and contrastive-
  negative training. See `data/coin/` and App E.3 in the paper.
- **SPL-CC** — Symbolic Physics Logic. Applied quantitative reasoning with
  perfectly separable retain channel (`acc_retain = 0` for every tested
  model). Useful as a perfect-separation diagnostic and as a numerically
  graded eval. See `data/{train,bench,holdout}/spl_cc_*.jsonl`.
- **CORE-MATH** — Mathematical proof construction. Judge-saturating on
  instruct-tuned models. Useful as a judge-acceptance probe for
  mathematical register. See `data/{train,bench,holdout}/core_math_*.jsonl`.

The recipes below are runnable on a single A100 80GB (most also on a 5090
32GB at fp16 / int8). Quickstart runs on CPU.

---

## 1. RepE bidirectional steering with COIN

**Goal:** extract two steering vectors per primitive — `v_presence`
(activate the primitive) and `v_direction` (apply it correctly) — and
score their separation.

```bash
# 1.1. For each primitive, extract both steering vectors at a chosen layer
for c in tom_cc ctr_cc mor_cc str_cc stp_cc; do
  python code/probes/steering_extractor.py \
      --model meta-llama/Llama-3-8B \
      --corpus $c --layer 16 --use-coin --max-pairs 200
done
# Outputs: results/steering/Llama-3-8B/{c}_layer16.npz with v_presence, v_direction

# 1.2. Layer sweep — find where v_direction and v_presence disentangle
for ell in 8 12 16 20 24 28; do
  python code/probes/steering_extractor.py \
      --model meta-llama/Llama-3-8B --corpus tom_cc \
      --layer $ell --use-coin --max-pairs 100
done
```

**Reading the output.** Each script prints `cos(v_presence, v_direction)`.
Layers where this cosine is close to 1 mean COIN gives little new info
(presence and correctness are entangled at that depth). Layers where it
approaches 0 are where the two axes are usefully distinct — that's where
you'd insert `α · v_direction` for direction-specific steering.

**Apply the steering vector.** A minimal example using HuggingFace hooks:

```python
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3-8B"
layer = 16
alpha = 4.0  # tune per task

v = np.load(f"results/steering/Llama-3-8B/tom_cc_layer{layer:02d}.npz")["v_direction"]
v = torch.tensor(v, dtype=torch.float16, device="cuda")
v = v / v.norm()  # normalise direction; choose magnitude via alpha

tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def hook(module, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    h[..., :] = h[..., :] + alpha * v
    return (h,) + out[1:] if isinstance(out, tuple) else h

handle = model.model.layers[layer].register_forward_hook(hook)
ids = tok("...your prompt...", return_tensors="pt").to("cuda").input_ids
out = model.generate(ids, max_new_tokens=64)
handle.remove()
print(tok.decode(out[0]))
```

---

## 2. COIN as contrastive-negative training data (DPO / IPO / RPO)

**Goal:** use COIN's behavioural-opposite pairs as semantically-meaningful
hard negatives for preference training, instead of randomly sampled
distractors that introduce topic confounds.

The natural triple per primitive is `(prompt, chosen=target, rejected=opposite)`
where `chosen` and `rejected` describe the same scenario with the cognitive
operation applied vs. reversed.

```python
import json
from datasets import Dataset

def load(path):
    return [json.loads(l) for l in open(path)]

cc      = load("data/train/tom_cc_train.jsonl")          # target / retain
opp     = load("data/coin/TOM_COIN/train.jsonl")          # target / retain (opposite)

# Index COIN by scenario_id, fall back to category if needed
opp_by_scn = {r["scenario_id"]: r for r in opp}

triples = []
for r in cc:
    o = opp_by_scn.get(r["scenario_id"])
    if not o:
        continue
    triples.append({
        "prompt":   "Continue the scenario in the most appropriate way.",
        "chosen":   r["target"],          # primitive correctly applied
        "rejected": o["target"],          # primitive reversed (COIN)
    })
ds = Dataset.from_list(triples)
ds.to_json("data/derived/tom_dpo_pairs.jsonl")
```

Then feed `data/derived/tom_dpo_pairs.jsonl` to your DPO trainer
(`trl.DPOTrainer`, `axolotl`'s DPO recipe, etc.). The contrast lives on
the *cognitive axis* rather than on topical / stylistic distance, which is
what most random-negative DPO gets.

---

## 3. SPL-CC perfect-separation diagnostic

**Goal:** for any new contrastive corpus design, check whether the retain
channel is *too* well-separated (`acc_retain → 0` collapses CA to raw
accuracy) or appropriately graded.

```bash
python quickstart.py --model HuggingFaceTB/SmolLM-360M \
                     --corpus spl_cc --grader exact --limit 200
# Expect: acc_target ≈ 0.5 (model gets some right), acc_retain = 0.000
# (model extracts no number from prose), CA = acc_target.
```

If your candidate corpus shows the same pattern, CA reduces to raw target
accuracy — the contrast adds nothing diagnostic. SPL-CC is the canonical
worked example.

---

## 4. SPL-CC numerical-only evaluation (no judge needed)

**Goal:** evaluate any model's applied-quantitative reasoning at $<$1 ms
per pair using deterministic numeric-match grading. No judge cost.

```bash
python code/cogbench/cogbench_eval.py \
    --model meta-llama/Llama-3-8B \
    --corpus spl_cc --grader exact --split bench
```

The `exact` grader extracts the model's numerical answer (with unit
tolerance) and compares to the SPL-CC gold key. Result is reproducible
across runs (no judge stochasticity). Use this when the deployment
question is "did the model get the right number?" and not "did it write
a fluent derivation?"

---

## 5. CORE-MATH judge-acceptance probe

**Goal:** test whether a candidate judge panel is mathematical-register-
coupled by running it over CORE-MATH proofs from a small base model.

```bash
# Run the candidate judge panel (replace 'qwen-14b-instruct' below) over
# CORE-MATH for an ≤8B base model where research-level proof competence
# is implausible.
python code/cogbench/judge_calibration.py \
    --corpus core_math --judges qwen-14b-instruct,mistral-nemo-12b \
    --eval-model meta-llama/Llama-3.2-1B \
    --split bench
```

**Reading the result.** If the panel accepts $>$0.9 of `D_target` proofs
from a 1-3B base model on `category|research_level` cells, the panel is
register-coupled (grading fluency, not correctness). Pair it with a
`sympy`-based equivalence grader as a correctness-calibrated alternative.
This is the worked example behind the App G grader-intent decision.

---

## 6. CORE-MATH base-vs-instruct register dissociation

**Goal:** quantify how much instruct/RLHF tuning shifts judge acceptance
on formal-register output independent of competence.

```bash
for m in meta-llama/Llama-3-8B meta-llama/Llama-3.1-8B-Instruct; do
  python code/cogbench/cogbench_eval.py \
      --model $m --corpus core_math \
      --grader judge --split bench
done
```

Compare `acc_target` between the base and the instruct variant.
Difference is the instruct register-fluency uplift, *not* a competence
difference (since the underlying weights are tuned, not retrained on math).

---

## 7. LoRA destructive validity test (App F)

**Goal:** verify a candidate synthetic reasoning corpus measures
cognition rather than generator-style fluency. Train a LoRA adapter on
`D_target` text alone and re-evaluate on a structurally distinct human-
written benchmark for the same primitive.

```bash
# Train: LoRA rank 32 on ToM-CC target text, 3 seeds
for seed in 42 1337 2024; do
  python code/Lora/train.py \
      --model meta-llama/Llama-3-8B --corpus tom_cc \
      --target-only --rank 32 --alpha 16 --seed $seed
done

# Eval on BigToM (separate human-written ToM benchmark)
python code/Lora/evaluate.py \
    --model meta-llama/Llama-3-8B --adapter-glob 'lora_*/' \
    --benchmark bigtom
```

If accuracy *rises* on BigToM after target-only LoRA, your corpus is a
stylistic shortcut. CogniPrimitiveBench's three architectures all *degrade*
($-12$ to $-50$ pp) — signal lives in the contrastive structure.

---

## 8. Curriculum / deficit-map design (App A.3)

**Goal:** identify which cognitive primitives a candidate fine-tune
should target, given a known deployment domain.

```python
import csv, statistics
# Read per-model raw + bias-corrected CA
rows = list(csv.DictReader(open("results/cogbench/table5_master_corrected.csv")))
target = [r for r in rows if r["model"] == "meta-llama/Llama-3-8B"][0]
deficits = sorted(
    ((p, float(target[f"{p}_corrected"])) for p in
     ["ToM", "Causal", "Moral", "Strategic", "Spatial"]),
    key=lambda x: x[1]
)
# Lowest CA primitives are training-data priorities for that model
print("Bottom-2 deficits:", deficits[:2])
```

Use Table 21 (`results/composition/preliminary_regression.json`) to map
your deployment domain to its cognitive primitive subset, then prioritise
training-data generation for the bottom-2 deficits within that subset.

---

## 9. NULL-CC as a portable judge-bias auditor

**Goal:** audit any LLM-as-Judge panel for register coupling before
deploying it on synthetic-data scoring.

```bash
python code/cogbench/judge_calibration.py \
    --corpus null_cc \
    --judges <your-judge-1>,<your-judge-2> \
    --gold-pairs calibration/gold_pairs/null_cc_gold_50.jsonl
```

Output: `δ = accept_formal − accept_informal` per judge and per panel.
Panels with `|δ| > 10pp` are register-coupled and should be re-balanced
(swap a permissive judge for a strict one) before being used to grade
free-text model outputs. This is the App G calibration protocol applied
to any panel you bring in.

---

## Citation

If you use any of these recipes, cite:

```bibtex
@inproceedings{cogbench2026,
  title={CogniPrimitiveBench: A Decompositional Benchmark for Cognitive Reasoning in Large Language Models},
  author={Anonymous},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS) -- Track on Evaluations and Datasets},
  year={2026},
  note={Anonymous review}
}
```

For just the steering / probing recipes (1-2), the canonical RepE
reference is Zou et al., *"Representation Engineering"*, arXiv 2310.01405.
