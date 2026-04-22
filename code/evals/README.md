# `evals/` — held-out benchmark evaluation

Scores a trained CogBench adapter on the held-out benchmark configured
for its training domain (`DOMAINS[*]["heldout_bench"]` in
`configs/seeds.py`). Surfaces the natural-text degradation that fine-
tuning on synthetic CogBench pairs can introduce (see
`../Lora/README.md` § "Training objective").

## Layout

```
evals/
├── eval_heldout.py     # driver: picks the right bench per adapter, runs it
├── tasks/              # custom (non-lm-eval) bench runners
│   ├── eval_bigtom.py
│   ├── eval_balanced_copa.py
│   ├── eval_bigbench_strategy.py
│   ├── eval_bigbench_diplomacy.py
│   ├── eval_stepgame.py
│   ├── eval_spartqa.py
│   ├── eval_formality.py
│   ├── eval_legalbench.py
│   ├── eval_scienceqa.py
│   ├── eval_finqa.py
│   └── eval_cybermetric.py
├── configs/seeds.py    # domain → held-out bench table
└── utils/              # shared model loader / metric writer
```

## Install

```bash
pip install -r requirements.txt
cp .env.example .env           # fill HF_TOKEN
huggingface-cli login
```

## CLI

Use `--help` as the source of truth:

```bash
python eval_heldout.py --help
python -m tasks.eval_bigtom --help
```

`eval_heldout.py` flags:
- `--adapter-dir PATH` — evaluate one adapter
- `--results-dir PATH` — evaluate every `run_meta.json`-bearing subdir
- `--model` / `--domain` — override the adapter's inferred metadata

Custom bench runners expose:
- `--model MODEL_KEY` (required)
- `--adapter PATH` (optional; skip for base-model scoring)

## Run

Evaluate one adapter:

```bash
python eval_heldout.py --adapter-dir ../Lora/results/lora_llama-3-8b_math_seed42/adapter
```

Evaluate every adapter in a directory:

```bash
python eval_heldout.py --results-dir ../Lora/results
```

Run one custom bench stand-alone:

```bash
python -m tasks.eval_bigtom --model llama-3-8b \
    --adapter ../Lora/results/lora_llama-3-8b_tom_seed42/adapter
```

## Outputs

- Per-adapter JSON: `<run_dir>/heldout_<bench>.json`
- Rolled-up CSV: `results/heldout_generalization.csv`
