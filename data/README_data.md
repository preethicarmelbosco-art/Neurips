# `data/` — CogBench corpora

145 390 contrastive pairs across 39 JSONL files. Every record is a
`(target, retain)` pair sharing one scenario seed; CA = `acc_target − acc_retain`
isolates the cognitive primitive from surface lexical overlap.

License: **CC BY 4.0** (`LICENSE-DATA` at the bundle root).
Authoritative schema + provenance: `croissant.json` at the bundle root.

## Layout

```
data/
├── train/      (8 files, ~106 MB)   pre-injection pairs — used to fit any
│                                    LoRA / probe / steering vector
├── bench/      (8 files, ~117 MB)   pre-injection pairs reserved for CA scoring
│                                    (CogBench main evaluation split)
├── holdout/    (8 files,  ~11 MB)   post-injection pairs — never seen during
│                                    generation or judge calibration; final
│                                    held-out check
└── coin/       (5 domains × 3 splits, ~34 MB)
                                     COIN = Contrast partners. Each pair is
                                     a complete-opposition (sides-of-a-coin)
                                     restatement, not a near-paraphrase.
```

`coin/` substructure: `CAU_COIN/, MOR_COIN/, STP_COIN/, STR_COIN/, TOM_COIN/`
each with `train.jsonl`, `bench.jsonl`, `holdout.jsonl`.

## Corpora

| Corpus    | Skill probed                                        |  Train |  Bench | Holdout |
| --------- | --------------------------------------------------- | -----: | -----: | ------: |
| SPL-CC    | Applied scientific reasoning (symbolic physics)     |  7 568 |  8 409 |     841 |
| ToM-CC    | Theory of mind (belief, deception, knowledge)       |  6 955 |  7 728 |     773 |
| CTR-CC    | Counterfactual causal reasoning                     |  6 351 |  7 057 |     706 |
| MOR-CC    | Moral / normative reasoning                         |  5 318 |  5 909 |     591 |
| STR-CC    | Strategic reasoning                                 |  5 243 |  5 825 |     582 |
| STP-CC    | Stepwise planning                                   |  7 946 |  8 829 |     883 |
| NULL-CC   | Style-only negative control (formal ↔ informal)     |  7 742 |  8 465 |     723 |
| CoreMath  | Formal proof vs. intuitive explanation              |  7 971 |  8 857 |     560 |
| COIN (×5) | Complete-opposition partners; per-domain, 3 splits  | 10 601 | 11 779 |   1 178 |

**Total: 145 390 contrastive pairs across 39 files.**

## Record schema

Common fields on every record: `id` (UUID v4), `timestamp` (ISO-8601 UTC),
exactly one `target*` field, exactly one `retain*` field. Per-corpus field
names below; full schema is authoritative in `croissant.json`'s `recordSet`
entries.

| Corpus    | target field      | retain field         |
| --------- | ----------------- | -------------------- |
| SPL-CC    | `target`          | `retain`             |
| ToM-CC    | `target`          | `retain`             |
| CTR-CC    | `target_text`     | `retain_text`        |
| MOR-CC    | `target`          | `retain`             |
| STR-CC    | `target`          | `retain`             |
| STP-CC    | `target`          | `retain`             |
| NULL-CC   | `target_formal`   | `retain_informal`    |
| CoreMath  | `target_proof`    | `retain_intuition`   |
| COIN      | `target`          | `retain`             |

## Loading

```python
# Pure-Python, no torch dependency
from load_cogbench import load_corpus
pairs = load_corpus("tom_cc", split="bench")   # iterator over dicts
```

Or read directly with the standard library:

```python
import json
with open("data/bench/tom_cc_bench.jsonl") as f:
    for line in f:
        rec = json.loads(line)
```

## Splits — what to use when

- **CogBench evaluation (paper Table 5)**: `bench/`. CA is computed on
  pre-injection pairs the model has not been adapted on.
- **Probe / LoRA / steering training**: `train/`. Disjoint from `bench/`.
- **Final held-out check**: `holdout/`. Post-injection split — never seen
  during pair generation, judge calibration, or any model adaptation. Use
  this once, last.
- **Bidirectional steering (App E)**: any `coin/<DOMAIN>_COIN/` split.
  Complete-opposition pairs work better than near-paraphrase pairs as
  RepE anchors.

## Integrity

- `SHA256SUMS` and `MD5SUMS` (bundle root) cover every `data/*.jsonl`.
- Croissant 1.1 + RAI metadata in `croissant.json` is auto-validated by
  `mlcroissant validate --jsonld croissant.json`.
- Decontamination spec and per-corpus generation/judging provenance:
  `DATASHEET.md` (bundle root).

## See also

- `README.md` (bundle root) — full bundle map and quickstart.
- `DATASHEET.md` (bundle root) — Datasheet for Datasets (Gebru et al.).
- `RECIPES.md` (bundle root) — 9 worked recipes that consume these files.
- `code/cogbench/` — the harness that scores any HF model on these pairs.