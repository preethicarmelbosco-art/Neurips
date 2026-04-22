# `cogbench/` — setup

## Requirements
- Python 3.10+, CUDA 12.x recommended
- One GPU (≥24 GB) for small models; A100/H100-class for ≥27 B
- Dataset at `../data/` (train/bench/holdout/coin JSONL)

## Install
```bash
pip install -r requirements.txt
huggingface-cli login          # gated models (Llama, Gemma) need this
```

## Smoke test
```bash
python cogbench_eval.py --model llama-3.2-1b --corpus null_cc --limit 50
```

Produces `results/llama-3.2-1b/cogbench.json` with non-zero target/retain
counts. Should finish in <2 minutes on a single consumer GPU.

## Full run
```bash
python cogbench_eval.py --model <model_key> --all-corpora
```

Wall-clock on one A100: ~45 min for a 7 B model across all 8 corpora
(judge grading included).

## Common issues
- **OOM on judge models** — lower `COGBENCH_JUDGE_BATCH=2`.
- **vLLM not loading** — set `COGBENCH_NO_VLLM=1` to force the HF path.
- **Gated HF repo** — run `huggingface-cli login` and accept the repo's ToS on HF.