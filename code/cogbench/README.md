# `cogbench/` — CogBench evaluator

Computes **Contrastive Accuracy (CA)** for a model on the CogBench suite:
`CA(m, c) = acc_target(m, c) − acc_retain(m, c)`. A large CA means the
model answers correctly only when the scaffolding is present.

## Files

| File                   | Role                                                          |
| ---------------------- | ------------------------------------------------------------- |
| `cogbench_eval.py`     | main harness — loads a model, runs every CogBench corpus, scores with auto-graders or a judge panel, writes per-model JSON + CSV. |
| `cogbench_model_json.py` | atomic writer for the per-model result JSON.                |
| `judge_panel.py`       | multi-judge LLM panel (unanimous / majority vote).            |
| `consensus_gate.py`    | accept/reject gate wrapping the panel.                        |
| `judge_calibration.py` | asymmetric-bias audit of the judge panel (Phase 1 gold-answer generation with a local HF LLM, Phase 2 panel calibration). |

## Install

```bash
pip install -r requirements.txt
cp .env.example .env       # fill HF_TOKEN and JUDGE_* vars
huggingface-cli login      # gated checkpoints (Llama, Gemma)
```

## Run

Use `python <script> --help` as the source of truth for flags.

```bash
# One model, one corpus
python cogbench_eval.py --model llama-3-8b --corpus spl_cc

# One model, all corpora
python cogbench_eval.py --model llama-3-8b --all-corpora

# Post-hoc: re-aggregate saved results
python cogbench_eval.py --analysis

# Judge panel reliability check
python judge_calibration.py --generate-gold --calibrate --all-corpora \
    --judge-mode local --judge-devices cuda:0,cuda:1
```

Model zoo is inlined in `cogbench_eval.py` (`STUDENT_MODELS`). Add a new
entry `{"hf_id": ..., "arch": ..., "size_b": ...}` to score additional
models.

## Graders

- **Auto-graded**: SPL-CC (exact numeric match), STP-CC (exact location).
- **Judge-graded**: ToM-CC, CTR-CC, MOR-CC, STR-CC, CoreMath — 3-judge
  panel, factual-accuracy rubric.
- **NULL-CC**: style-only negative control; expected CA ≈ 0.
- **COIN**: complete-opposition pairs; judge-graded detection.

All judges are **local open-weights models** (HF Transformers or Ollama).
No commercial APIs are used anywhere in the scoring path.

Configure via env vars:

```
JUDGE_MODEL=qwen2.5:14b
JUDGE_BASE_URL=http://localhost:11434/v1
COGBENCH_NO_VLLM=1         # force HF path even when vLLM is importable
COGBENCH_GRADER_CHUNK=5000
COGBENCH_JUDGE_BATCH=8
```

## Outputs

Per model: `results/<model>/cogbench.json` (scores + per-corpus traces)
plus an appended row in `results/cogbench_master.csv`.