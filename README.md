# CogBench v1.1.0 — anonymous NeurIPS 2026 Datasets & Benchmarks submission

Contrastive corpus suite for cognitive-circuit analysis in language
models. This directory is the **reviewer-accessible bundle** pointed at
by the Croissant metadata URL. Everything is double-blinded; no author,
affiliation, hostname, URL, or e-mail is embedded in the metadata or the
data records.

## Contents

```
NeurIPS/
├── croissant.json            # Croissant 1.1 + RAI metadata (validated)
├── LICENSE                   # Apache 2.0 — applies to code/, quickstart.py, load_cogbench.py
├── LICENSE-DATA              # CC BY 4.0 — applies to data/
├── README.md                 # this file
├── DATASHEET.md              # Datasheet for Datasets (Gebru et al. 2018)
├── SHA256SUMS, MD5SUMS       # checksums for every data/*.jsonl
│
├── quickstart.py             # score any HF model on one corpus in <5 min
├── load_cogbench.py          # pure-Python data loader (no torch required)
│
├── data/                     # 145 390 records in 39 JSONL files
│   ├── train/    ── 8 files
│   ├── bench/    ── 8 files
│   ├── holdout/  ── 8 files
│   └── coin/     ── 5 domains × 3 splits — COIN = Contrast
│       (CAU_COIN, MOR_COIN, STP_COIN, STR_COIN, TOM_COIN;
│        pairs are complete opposites, like sides of a coin)
│
├── code/                     # runnable research code
│   ├── generation/           # 9-corpus generation + validation pipeline
│   ├── cogbench/             # CA harness + judge panel + calibration audit
│   ├── Lora/                 # LoRA adapter training on target text
│   ├── evals/                # held-out benchmark drivers
│   ├── analysis/             # paper-table reproduction (Tier 1; 8 scripts)
│   ├── figures/              # paper-figure reproduction (Tier 2; 5 figures)
│   └── probes/               # App E linear-probe + steering cookbook (Tier 4)
│
├── results/                  # pre-computed result CSVs (Tier 3)
│   ├── cogbench/             #   table5_master.csv (raw + corrected CA per model)
│   ├── ceiling_compression/  #   ca_all_rows.csv, eta2_family_ci.csv, family_primitive_modes.csv
│   ├── composition/          #   table3_beta.csv, sign-stability, kitchen-sink, lasso
│   └── domain/               #   per-model held-out scores
│
├── calibration/              # judge-bias calibration (Tier 5; App G Table 15)
│   ├── gold_pairs/           #   5 corpora x 50 author-written gold pairs
│   ├── judge_outputs/        #   pre-computed Qwen-14B + Mistral-Nemo verdicts
│   └── calibration_summary.csv  # δ + correction action per corpus
│
└── tools/                    # reproducibility helpers (Tier 6)
    ├── generate_croissant.py    # rebuild croissant.json from data/
    └── refresh_paper_numbers.sh # rerun all 8 analysis scripts in dependency order
```

Each `code/` folder is **self-contained** — its own `README.md`,
`requirements.txt`, `setup.md`, and `.env.example` where applicable.

## Corpora at a glance

| Corpus     | Skill probed                                        | Train | Bench | Holdout |
| ---------- | --------------------------------------------------- | ----: | ----: | ------: |
| SPL-CC     | Applied scientific reasoning (symbolic physics)     | 7 568 | 8 409 |     841 |
| ToM-CC     | Theory of mind (belief, deception, knowledge)       | 6 955 | 7 728 |     773 |
| CTR-CC     | Counterfactual causal reasoning                     | 6 351 | 7 057 |     706 |
| MOR-CC     | Moral / normative reasoning                         | 5 318 | 5 909 |     591 |
| STR-CC     | Strategic reasoning                                 | 5 243 | 5 825 |     582 |
| STP-CC     | Stepwise planning                                   | 7 946 | 8 829 |     883 |
| NULL-CC    | Style-only negative control (formal ↔ informal)     | 7 742 | 8 465 |     723 |
| CoreMath   | Formal proof vs. intuitive explanation              | 7 971 | 8 857 |     560 |
| COIN (×5)  | Contrast partners: complete-opposition target/retain | 10 601 | 11 779 |  1 178 |

**Total: 145 390 contrastive pairs across 39 files.**

## Record schema

Every record is a `(target, retain)` pair sharing one scenario seed. Common
fields: `id` (UUID v4), `timestamp` (ISO-8601 UTC), exactly one `target*`
field, exactly one `retain*` field. Full schema is authoritative in
`croissant.json`'s `recordSet` entries.

| Corpus     | target field     | retain field         |
| ---------- | ---------------- | -------------------- |
| SPL-CC     | `target`         | `retain`             |
| ToM-CC     | `target`         | `retain`             |
| CTR-CC     | `target_text`    | `retain_text`        |
| MOR-CC     | `target`         | `retain`             |
| STR-CC     | `target`         | `retain`             |
| STP-CC     | `target`         | `retain`             |
| NULL-CC    | `target_formal`  | `retain_informal`    |
| CoreMath   | `target_proof`   | `retain_intuition`   |
| COIN       | `target`         | `retain`             |

## Recipes

### 0. Reproduce paper tables and figures from the shipped CSVs

```bash
# Rebuild every analysis output (Tables 3, 4, 19, 22, App G Table 15-corrected, etc.)
bash tools/refresh_paper_numbers.sh

# Rebuild all 5 paper figures (Fig 1, 2, 3, 6, 7) into ./figures/
bash code/figures/make_all_figures.sh

# Validate the Croissant metadata
mlcroissant validate --jsonld croissant.json
```

The 8 analysis scripts under `code/analysis/` are runnable independently
and read only from `results/`. They reproduce numbers within bootstrap
variance of the values reported in the paper.

### 1. Score my model on CogBench (quickstart — no judge panel)

```bash
# Laptop, no GPU, ~2 minutes — produces a real CA number.
python quickstart.py --cpu-tiny --corpus spl_cc --limit 20

# Any HF model on a GPU:
python quickstart.py --model meta-llama/Llama-3.2-1B --corpus spl_cc --limit 100
```

Outputs one line: `corpus=spl_cc  n=100  target_acc=0.48  retain_acc=0.39
CA=0.09  wall=42.7s  device=cuda`. Auto-graded corpora only
(`spl_cc`, `stp_cc`, `null_cc`); the full judge-panel harness is under
`code/cogbench/`.

### 2. Just load the data

```bash
python load_cogbench.py --list                       # every file + line count
python load_cogbench.py --corpus tom_cc --split bench --limit 3
```

As a library: `from load_cogbench import load_split, list_corpora`.

### 3. Full CogBench evaluation (judge-graded, all 9 corpora)

```bash
cd code/cogbench
pip install -r requirements.txt
cp .env.example .env          # HF_TOKEN, JUDGE_*
python cogbench_eval.py --model llama-3-8b --all-corpora
```

### 4. Judge-panel reliability audit

```bash
cd code/cogbench
python judge_calibration.py --generate-gold --calibrate --all-corpora \
    --judge-mode local --judge-devices cuda:0,cuda:1
```

All gold-answer generation and judging is done with local open-weights
LLMs — no commercial APIs anywhere in the pipeline.

### 5. LoRA fine-tune on CogBench target text

```bash
cd code/Lora
pip install -r requirements.txt
python train.py --model llama-3-8b --domain math --seed 42
```

**Training objective is single-target causal-LM fine-tuning** (not a
contrastive / DPO loss). See `code/Lora/README.md` § "Training objective"
— the adapter typically narrows the generation distribution and can hurt
natural-text held-out benchmarks.

### 6. Held-out benchmark evaluation for a trained adapter

```bash
cd code/evals
pip install -r requirements.txt
python eval_heldout.py --adapter-dir ../Lora/results/lora_llama-3-8b_math_seed42/adapter
```

### 7. Regenerate a corpus from scratch

```bash
cd code/generation
pip install -r requirements.txt
cp .env.example .env          # OpenAI-compat endpoint for a local LLM
NUM_SAMPLES=500 OUTPUT_PATH=./data/spl_cc_accepted.jsonl \
    python -m corpora.spl_main
```

All generation uses local open-weights models (Ollama / vLLM / LM Studio
/ real OpenAI — whichever OpenAI-compat endpoint you point at).

## Verification

```bash
# Re-compute checksums
find data -name '*.jsonl' | sort | xargs sha256sum | diff - SHA256SUMS

# Check Croissant metadata
mlcroissant validate --jsonld croissant.json

# End-to-end smoke
python quickstart.py --cpu-tiny --corpus spl_cc --limit 5
```

## Intended use and non-use

**Use for**: mechanistic interpretability, Cognitive Absorption (CA)
measurement, contrastive fine-tuning research, negative-control
experiments.

**Do not use for**: ranking "model intelligence" on a public leaderboard
(CogBench is diagnostic, not a capability ranking), evaluating frontier
research mathematics (CoreMath is textbook/competition-grade), or any
use that treats retain accuracy as a failure signal (it is a control).

## Licensing & hosting

Dual-licensed:

- **Data** (`data/**/*.jsonl`) under **Creative Commons Attribution 4.0
  International (CC BY 4.0)** — see `LICENSE-DATA`.
- **Code** (`code/`, `quickstart.py`, `load_cogbench.py`) under the
  **Apache License, Version 2.0** — see `LICENSE`.

The split follows NeurIPS D&B practice: CC BY is the standard open-data
licence; Apache 2.0 is the standard permissive software licence (includes
an explicit patent grant and matches the HuggingFace / PyTorch ecosystem
default). Both permit redistribution and commercial use with attribution;
neither contains a non-commercial or share-alike restriction.

For camera-ready we will mirror this bundle to a DOI-issuing archive
(Hugging Face Hub / Harvard Dataverse / Zenodo) preserving the file
layout, checksums, and licences.

## Verification status

What was exercised on a fresh clone before release:

- **Data integrity** — every file's SHA-256 and MD5 match `SHA256SUMS` /
  `MD5SUMS`; record counts and schemas match `croissant.json`;
  `mlcroissant validate` passes.
- **Metadata** — Croissant 1.1 + RAI fields load cleanly; no identifying
  strings remain in any `.py` / `.md` / `.json` file.
- **Entrypoint wiring** — every argparse script exits 0 on `--help`; every
  package imports without error; `python -m tasks.<bench>` dispatch works.
- **End-to-end smoke** — `python quickstart.py --cpu-tiny --corpus spl_cc
  --limit 5` loads `distilgpt2` on CPU, scores 5 pairs, prints a valid CA
  line in <5 s.

What requires **your** environment to exercise (will fail loudly without
it — not a release bug):

- **GPU + HF token** — `quickstart.py` on a real HF model, all of
  `code/cogbench/`, `code/Lora/`, and `code/evals/` need CUDA and a HF
  token for gated checkpoints (Llama, Gemma).
- **A running LLM endpoint** — every `code/generation/corpora/*_main.py`
  and the `code/cogbench/judge_*` paths talk to an OpenAI-compatible
  endpoint (Ollama / vLLM / LM Studio). With no endpoint live the scripts
  error on the first API call, not at import.
- **`lm_eval` tasks** — `code/Lora/evaluate.py` and several `code/evals/`
  tasks shell out to the `lm-eval` CLI; install separately
  (`pip install lm-eval>=0.4.3`).

## Anonymisation

All author-identifying content has been stripped from this bundle:

- `creator`, `publisher`, `citeAs.author` in `croissant.json` → `Anonymous`.
- Dataset `url` → anonymous hosting endpoint.
- Code files scanned for usernames, paths, hostnames, IPs, e-mails,
  cluster names, private project identifiers — zero hits.
- Data records contain no PII or demographic attributes.
