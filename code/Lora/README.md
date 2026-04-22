# `Lora/` — LoRA fine-tuning on CogBench target text

Trains a LoRA adapter on the **`target` column** of a CogBench domain,
then scores the adapter on primary + held-out benchmarks.

## Training objective (read this before interpreting results)

The training loop is **plain causal-LM fine-tuning on the `target` column
only**. It is *not* a contrastive loss (DPO / ORPO) over (target, retain)
pairs. In other words:

- The `retain` column is **evaluation-only** — it is never in the
  training batches.
- Fine-tuning therefore *adds* skill-bearing text to the model, which
  increases `acc_target` and typically narrows `CA = acc_target − acc_retain`
  only if the model starts internalising what used to require scaffolding.
- Because CogBench pairs are synthetic, the adapter narrows the
  generation distribution. Expect **non-trivial degradation on natural-text
  held-out benchmarks** (GSM8K, BigToM, Balanced-COPA, etc.). This is
  catastrophic forgetting, not a CogBench property. It is what `evals/`
  exists to quantify.

If you want a contrastive-loss regime (target preferred, retain rejected),
that is a different script and is not part of this release.

## Layout

```
Lora/
├── train.py          # adapter training (one run per model × domain × seed)
├── evaluate.py       # lm-eval + WikiText-2 PPL + wandb summary
├── configs/seeds.py  # student zoo, domain → split map, benchmark map
└── utils/
    ├── model_loader.py  # load_student / attach_lora
    ├── metrics.py       # EvalResult + save_results
    └── perplexity.py    # WikiText-2 PPL
```

## Install

```bash
pip install -r requirements.txt
cp .env.example .env        # fill HF_TOKEN
huggingface-cli login       # gated checkpoints (Llama, Gemma)
wandb login                 # optional
```

## CLI

```bash
python train.py --help
python evaluate.py --help
```

Defaults:

| Flag        | Values                                                      |
| ----------- | ----------------------------------------------------------- |
| `--model`   | `llama-3-8b` / `qwen-2.5-7b` / `falcon-mamba-7b` / `mamba-2.8b` |
| `--domain`  | `math` / `tom` / `causal` / `moral` / `strategic` / `spatial` / `core_math` / `null` |
| `--seed`    | `42` / `1337` / `2024`                                       |
| `--run-all` | loops the full M × D × S matrix                              |

Example — one run:

```bash
python train.py --model llama-3-8b --domain math --seed 42
python evaluate.py --run-dir results/lora_llama-3-8b_math_seed42
```

Full matrix + evaluation:

```bash
python train.py --run-all
python evaluate.py --eval-all --wandb
```

## Custom benchmarks

`evaluate.py` defers non-`lm_eval` benchmarks (BigToM, StepGame,
BigBench-Strategy, etc.) to the sibling `evals/` package — it prints a
hint with the command instead of silently skipping.

## Outputs

```
results/
├── lora_<model>_<domain>_seed<seed>/
│   ├── adapter/                  # PEFT adapter
│   ├── run_meta.json
│   ├── eval_<bench>/             # lm-eval output per bench
│   ├── eval_wikitext2_ppl.json
│   └── heldout_<bench>.json      # populated by evals/
└── eval_results.json             # consolidated summary
```
