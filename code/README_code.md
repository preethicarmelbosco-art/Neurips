# `code/` — CogBench research code

Runnable code for every numbered claim in the paper: pair generation,
contrastive-accuracy scoring, judge-panel calibration, LoRA destructive
validity probe, held-out benchmark drivers, paper-table analysis,
paper-figure plotting, and linear-probe / steering recipes.

License: **Apache 2.0** (`LICENSE` at the bundle root).
Each subdirectory is **self-contained** with its own `README.md`,
`requirements.txt`, and `setup.md` documenting GPU / endpoint requirements
and quick-start commands.

## Layout

```
code/
├── generation/     9-corpus generation + validation pipeline (DeepSeek-R1
│                   driver, 3-judge cross-family acceptance gate)
├── cogbench/       CA harness — loads a model, runs every CogBench corpus,
│                   scores via auto-graders or a 2-judge panel, writes per-
│                   model JSON + master CSV. Also: judge calibration runner
│                   for App G Table 15.
├── Lora/           Contrastive LoRA fine-tuning driver + evaluate.py
│                   (App F destructive-validity test: training on D_target
│                   text *degrades* held-out performance).
├── evals/          10 held-out benchmark runners (BigToM, BCOPA-CE, ETHICS,
│                   BB-Strategy, StepGame, MedQA, LegalBench, FinQA,
│                   Diplomacy-Deception, ScienceQA) — feeds Table 19/22.
├── analysis/       8 standalone scripts that reproduce paper tables from the
│                   shipped CSVs in results/. CPU-only, ~minutes total.
├── figures/        5 plot scripts + shared _style.py + make_all_figures.sh
│                   — reproduces every figure in the paper.
└── probes/         Appendix E reference scripts: linear-probe trainer,
                    steering vector extractor, NULL orthogonality check.
```

## Three reproduction tiers

The bundle is structured so reviewers can verify at the depth they have
GPU budget for. From cheapest to deepest:

1. **CSVs only (CPU, minutes)** — `bash tools/refresh_paper_numbers.sh`
   reruns every script in `code/analysis/` against the shipped
   `results/*.csv` and rebuilds Tables 3, 4, 19, 22, App G Table 15-corrected.
2. **Figures (CPU, seconds)** — `bash code/figures/make_all_figures.sh`
   replots Figs 1, 2, 3, 6, 7 from the same CSVs.
3. **End-to-end CA scoring (1 GPU, hours per model)** —
   `python code/cogbench/cogbench_eval.py --model <hf_id> --all-corpora`.

The judge-calibration audit (`code/cogbench/judge_calibration.py`), the
LoRA destructive-validity test (`code/Lora/`), and the held-out benchmarks
(`code/evals/`) each have their own GPU budget documented in the
respective `setup.md`.

## Per-subdirectory entry points

| Directory       | Entry-point                                                 | Reads from         | Writes to                     |
| --------------- | ----------------------------------------------------------- | ------------------ | ----------------------------- |
| `generation/`   | `corpora/<corpus>/generate.py`                              | `data/train` seeds | `data/{train,bench,holdout}/` |
| `cogbench/`     | `cogbench_eval.py --model <m> --corpus <c>`                 | `data/bench/`      | `results/cogbench/`           |
| `cogbench/`     | `judge_calibration.py --generate-gold --calibrate`          | author gold pairs  | `calibration/`                |
| `Lora/`         | `train.py --corpus <c>` then `evaluate.py`                  | `data/train/`      | `results/lora/`               |
| `evals/`        | `eval_heldout.py --task <bigtom\|bcope\|...> --model <m>`   | task datasets      | `results/domain/`             |
| `analysis/`     | `python <script>.py`  (8 standalone scripts, no flags)      | `results/*.csv`    | `tables/*.tex`                |
| `figures/`      | `bash make_all_figures.sh`                                  | `results/*.csv`    | `figures/*.pdf`               |
| `probes/`       | `train_probe.py --corpus <c> --layer <L>`                   | `data/train/`      | `results/probes/`             |

## Common environment

Every subdirectory's `requirements.txt` pins a minimal set; create a
fresh venv per subdirectory if pip resolution conflicts. Common
environment variables:

```
HF_TOKEN=...                # gated checkpoints (Llama, Gemma)
JUDGE_MODEL=qwen2.5:14b     # default judge (Ollama or local HF)
JUDGE_BASE_URL=http://localhost:11434/v1
COGBENCH_NO_VLLM=1          # force HF path even when vLLM is importable
COGBENCH_GRADER_CHUNK=5000  # judge batch chunking
COGBENCH_JUDGE_BATCH=8
```

Each `setup.md` lists the GPU class verified for that subdirectory
(typically RTX 5090 / 32 GB for ≤14B models, A100 80 GB for 27B+).

## See also

- `README.md` (bundle root) — full bundle map, corpora-at-a-glance, recipes index.
- `quickstart.py` (bundle root) — score any HF model on one corpus in <5 min.
- `RECIPES.md` (bundle root) — 9 worked recipes that consume these scripts.
- Each `code/<sub>/README.md` is the authoritative source for that
  subdirectory's flags, outputs, and data contracts.