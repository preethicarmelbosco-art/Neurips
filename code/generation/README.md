# `generation/` — CogBench corpus generation pipeline

Reproduces the 9-corpus CogBench v1.1.0 suite end-to-end: **generator →
structural validator → multi-judge LLM panel → post-processing**. Each
accepted pair is `(target, retain)` where `target` exercises the named
cognitive skill and `retain` is a content-matched control that lacks it.

## Layout

```
generation/
├── corpora/               # one factory per corpus (generator + prompts + validator)
│   ├── coin_*.py          # COIN = Contrast — complete-opposition pairs (5 themes)
│   ├── core_math_*.py     # formal proof vs. intuition
│   ├── ctr_*.py           # causal-temporal reasoning
│   ├── mor_*.py           # moral reasoning
│   ├── null_cc_*.py       # style-only negative control
│   ├── spl_*.py           # symbolic physics logic
│   ├── stp_*.py           # stepwise planning
│   ├── str_*.py           # strategic reasoning
│   ├── tom_*.py           # theory of mind
│   ├── judge_panel.py     # multi-judge LLM panel (unanimous / majority vote)
│   └── consensus_gate.py  # shared accept/reject gate wrapping the panel
└── postprocessing/        # applied after generation
    ├── holdout_split.py            # seed-disjoint train / bench / holdout split
    ├── decontaminate.py            # n-gram scan against public benchmarks
    ├── cjk_ascii_fix.py            # strip stray CJK chars the generator can emit
    ├── corpus_revalidate.py        # re-run judge panel on existing pairs
    ├── corpus_stats.py             # per-split size, difficulty, domain counts
    ├── compute_embed_similarity.py # within-corpus near-duplicate audit
    └── data_quality_audit.py       # end-to-end quality report
```

Every corpus follows the same 8-file skeleton (`_seeds`, `_models`,
`_prompts`, `_validator`, `_factory`, `_pipeline`, `_writer`, `_main`).
Each `_main.py` orchestrates generation + validation + writing for one
corpus.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in OPENAI_BASE_URL etc.
```

Generation speaks to any **OpenAI-compatible** chat endpoint (Ollama,
vLLM `--port`, LM Studio, or a real OpenAI key). Judges run on the same
kind of endpoint. See `setup.md`.

## CLI

The `_main.py` scripts are **env-var driven** (there is no argparse).
Every one reads `.env` (via `python-dotenv`) plus a small set of
per-corpus variables. Common knobs:

| Variable          | Default                              | Notes                                    |
| ----------------- | ------------------------------------ | ---------------------------------------- |
| `MODEL`           | `deepseek-r1:32b`                    | generator model id on the endpoint       |
| `OPENAI_BASE_URL` | `http://localhost:11434/v1`          | OpenAI-compatible endpoint               |
| `NUM_SAMPLES`     | `10000`                              | target number of accepted pairs          |
| `MAX_CONCURRENT`  | `2`                                  | parallel generator calls                 |
| `MAX_RETRIES`     | `3`                                  | per-pair retry budget                    |
| `OUTPUT_PATH`     | `data/<corpus>_accepted.jsonl`       | where accepted pairs go                  |
| `SEED`            | `42`                                 | reproducible seed-prompt ordering        |

Per-corpus vars (e.g. `SPL_HOLDOUT_COUNT`, `COIN_THEME`) are listed at the
top of the corresponding `_main.py` docstring.

## Generate one corpus

```bash
NUM_SAMPLES=500 OUTPUT_PATH=./data/spl_cc_accepted.jsonl \
    python -m corpora.spl_main
```

Swap `spl_main` for `tom_main`, `ctr_main`, `mor_main`, `str_main`,
`stp_main`, `null_cc_main`, `core_math_main`, or `coin_main`. Each run
streams JSONL to `OUTPUT_PATH` as pairs are accepted.

## Post-process to the v1.1.0 layout

After generation finishes for every corpus:

```bash
# Re-run the judge panel on anything that needs it
python -m postprocessing.corpus_revalidate --help

# Cross-benchmark decontamination (n-gram vs. public holdouts)
python -m postprocessing.decontaminate --help

# Optional: remove stray CJK characters
python -m postprocessing.cjk_ascii_fix --help

# Seed-disjoint split into train / bench / holdout
python -m postprocessing.holdout_split --help

# Final stats + quality report
python -m postprocessing.corpus_stats        --help
python -m postprocessing.data_quality_audit  --help
```

Each post-processing script ships its own `--help`; flags and defaults
live in the script rather than being duplicated here.

## Judge-panel configuration

`corpora/judge_panel.py` reads judges from environment variables:

```
COGBENCH_JUDGE_MODELS="qwen2.5:14b,mistral-nemo:12b"
COGBENCH_JUDGE_CONSENSUS="unanimous"     # or "majority"
COGBENCH_JUDGE_ENDPOINT="http://localhost:11434/v1"
```

Per-corpus consensus rules are baked into each validator: unanimous for
context-dependent corpora (ToM, CTR, MOR, STR, STP, COIN), majority for
style-only NULL.

## Notes

- All generation uses **local open-weights models** — no commercial
  API is called anywhere in the pipeline.
- Pairs that fail a validator are written to `*_rejected.jsonl` with
  judge verdicts attached.
- Runs are idempotent — re-running picks up where the last accepted pair
  left off, since each `_writer.py` appends and dedupes by `id`.