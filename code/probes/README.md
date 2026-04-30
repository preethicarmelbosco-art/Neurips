# Linear Representation Probes (App E cookbook)

Reference implementations of the App E recipe. These are *cookbook* scripts
backing the paper's text recipe; running them on a 7B-class model takes
roughly 20-30 minutes per corpus on a single RTX 5090 (32 GB) at fp16,
or ~2 hours on CPU for `--max-pairs 100`.

## Pipeline

```bash
# 1. Train cognitive + NULL probes at a chosen layer (App E.1)
for c in tom_cc ctr_cc mor_cc str_cc stp_cc null_cc; do
  python train_probe.py --model meta-llama/Llama-3-8B --corpus $c \
                        --layer 16 --max-pairs 200
done

# 2. Confirm cognitive probes are orthogonal to NULL (App E.2)
python null_orthogonality_check.py --model meta-llama/Llama-3-8B --layer 16

# 3. Extract bidirectional steering vectors using COIN (App E.3)
python steering_extractor.py --model meta-llama/Llama-3-8B --corpus tom_cc \
                              --layer 16 --use-coin
```

## Outputs

- `results/probes/<model>/<corpus>_layer<ell>.npz` -- probe weight `w`,
  bias `b`, train + val accuracy.
- `results/probes/<model>/orthogonality_layer<ell>.csv` -- cosine
  similarity between each cognitive probe direction and the NULL probe
  direction at the chosen layer.
- `results/steering/<model>/<corpus>_layer<ell>.npz` -- presence vector
  `v_presence` (D_target - D_retain) and, when `--use-coin` is set,
  direction vector `v_direction` (D_target - D_opposite via COIN).

## Notes on layer choice

A common heuristic is to sweep all layers (omit `--layer`) and pick `ell`
where probe accuracy peaks. For 7B-9B Dense Transformers this is typically
the upper-middle band (layers 14-22). For SSMs (Mamba family) the peak
is later in the stack.

## Dependencies

```
pip install transformers torch scikit-learn numpy
```

These scripts target the contrastive-pair substrate documented in
section 3.2 of the paper. They do not implement the causal-intervention
ladder of App E (directional ablation, interchange intervention, SAE
feature ablation); those are scoped as a follow-up paper ("CogBench-Mech").
