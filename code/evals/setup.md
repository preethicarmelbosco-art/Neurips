# `evals/` — setup

## Requirements
- Python 3.10+, CUDA 12.x
- One GPU ≥16 GB (adapter + base in fp16)
- Dataset at `../data/` (for NULL-CC formality bench)
- Trained adapters, typically under `../Lora/results/`

## Install
```bash
pip install -r requirements.txt
cp .env.example .env             # fill HF_TOKEN
huggingface-cli login            # gated Llama / Gemma
```

## Smoke test
```bash
python -m tasks.eval_bigtom --model llama-3.2-1b --adapter-path <path>
```

## Troubleshooting
- **`trust_remote_code` prompt** — `export HF_DATASETS_TRUST_REMOTE_CODE=1`
  (already in `.env.example`).
- **Bench dataset download fails** — manually pre-download via `datasets-cli`
  once, then rerun.
- **OOM** — lower the per-task `--batch-size` flag exposed by each runner.
