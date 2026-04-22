# `Lora/` — setup

## Requirements
- Python 3.10+, CUDA 12.x
- One GPU with ≥24 GB (bf16 training). A100/H100 recommended for 7–8 B students.
- Dataset at `../data/` (standard CogBench release layout)

## Install
```bash
pip install -r requirements.txt
huggingface-cli login          # gated Llama / Gemma checkpoints
wandb login                    # optional
```

## Smoke test (shortest path that exercises train.py)
```bash
python train.py --help     # confirm argparse loads
```
For a real 1-epoch run on the smallest model in the zoo (Mamba-2.8B):
```bash
python train.py --model mamba-2.8b --domain null --seed 42
```
NULL-CC is the smallest corpus, so this is the fastest wall-clock test.
To try a smaller student, add an entry to
`configs/seeds.py::STUDENT_MODELS` with the HF id and `lora_targets`.

## Full matrix
```bash
python train.py --run-all
python evaluate.py --eval-all --wandb
```

## Common issues
- **OOM** — lower `per_device_train_batch_size` in `train.py` or add 4-bit
  quant via `bitsandbytes`; SSMs need `--no-gradient-checkpointing`.
- **lm-eval task not found** — `pip install -U lm-eval` to refresh the task
  registry; confirm the task name in `configs/seeds.py::BENCHMARKS`.
- **wandb disabled** — `export WANDB_MODE=offline` or drop `report_to="wandb"`
  in `train.py`.