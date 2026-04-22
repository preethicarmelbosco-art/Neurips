# `generation/` — setup

## 1. System requirements

- Python 3.10+
- 16 GB RAM (no GPU needed — all compute is delegated to the LLM endpoint)
- Internet / network access to whatever model server you point at

## 2. Install

```bash
cd generation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Point at an LLM endpoint

`generation/` talks to an **OpenAI-compatible** chat endpoint. Pick one:

| Endpoint     | Setup                                                                     |
| ------------ | ------------------------------------------------------------------------- |
| OpenAI       | `export OPENAI_API_KEY=sk-...`  (uses `api.openai.com` by default)        |
| Ollama       | `ollama serve` then `export OPENAI_BASE_URL=http://localhost:11434/v1`    |
| vLLM         | `vllm serve <model> --port 8000` then `export OPENAI_BASE_URL=http://localhost:8000/v1` |
| LM Studio    | enable the OpenAI-compat server, then set `OPENAI_BASE_URL` accordingly   |

The suite v1.1.0 was generated with a local Ollama DeepSeek-R1:32b
generator and a mixed open-weights judge panel (Qwen-2.5-14B +
Mistral-Nemo-12B). The pipeline is model-agnostic — swap in any endpoint
that speaks the OpenAI chat schema.

## 4. `.env` file (optional but recommended)

```dotenv
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama           # Ollama ignores this; OpenAI real key if cloud
GENERATOR_MODEL=deepseek-r1:32b
COGBENCH_JUDGE_MODELS=qwen2.5:14b,mistral-nemo:12b
COGBENCH_JUDGE_CONSENSUS=unanimous
COGBENCH_JUDGE_ENDPOINT=http://localhost:11434/v1
```

`python-dotenv` loads this automatically on startup.

## 5. Smoke test (30 seconds, 2 pairs, no persistence)

```bash
python -m corpora.spl_main --n 2 --out /tmp/spl_smoketest.jsonl --workers 1
head /tmp/spl_smoketest.jsonl
```

You should see two JSON lines each with `target`, `retain`, `timestamp`,
and a UUID `id`. If this works, the pipeline is live.

## 6. Full regeneration of v1.1.0

Running the full pipeline from scratch reproduces the released suite up to
the generator's non-determinism (temperature, sampling order). Expect
compute proportional to the target size:

| Stage                  | Wall-clock (1× A100, DeepSeek-R1:32b) |
| ---------------------- | -------------------------------------: |
| Generate 8 corpora     |                                ~40 h  |
| COIN 5-domain family   |                                ~18 h  |
| Judge-panel validation |                                ~22 h  |
| Post-processing        |                                <1  h  |

See `README.md` for per-corpus commands.

## 7. Troubleshooting

- **Rate-limit / timeout errors** — lower `--workers` (default 8) to 2.
- **Structured-output parse errors** — the wrapper retries with backoff; if a
  specific seed persistently fails the validator, check the generator's
  reasoning output length and increase `MAX_TOKENS` in the corpus's
  `_models.py`.
- **Judge disagreement too high** — the panel uses unanimous consensus by
  default for cognitive corpora. Relax to majority via
  `COGBENCH_JUDGE_CONSENSUS=majority` for experimental runs only.
