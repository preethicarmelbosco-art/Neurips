"""Held-out generalisation evaluation for CogBench-trained adapters.

For each trained adapter, look up the held-out benchmark of its training
domain (table `DOMAINS[*]["heldout_bench"]`) and score the adapter on it.
Supports `lm_eval`-native benchmarks and the custom bench runners under
`tasks/`.

Usage:
    python eval_heldout.py --adapter-dir ../Lora/results/lora_llama-3-8b_math_seed42/adapter
    python eval_heldout.py --results-dir ../Lora/results
"""

import argparse
import importlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.seeds import BENCHMARKS, DOMAINS, STUDENT_MODELS
from utils.metrics import EvalResult, save_results

RESULTS_CSV = Path(__file__).resolve().parent / "results" / "heldout_generalization.csv"
LOG_DIR = Path(__file__).resolve().parent / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("eval_heldout")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(LOG_DIR / "eval_heldout.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(_fh)
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)
    logger.addHandler(_ch)


def extract_accuracy_from_lm_eval(output_path: Path) -> float:
    """Pull the first interpretable accuracy number out of an lm-eval JSON."""
    if not output_path.exists():
        return -1.0
    try:
        files = list(output_path.rglob("results_*.json")) if output_path.is_dir() else [output_path]
        for f in files:
            data = json.loads(f.read_text())
            for _, metrics in data.get("results", {}).items():
                for key in ("acc_norm,none", "acc,none", "exact_match,none"):
                    if key in metrics:
                        return float(metrics[key])
    except Exception:
        pass
    return -1.0


def _run_custom_bench(bench_key: str, model_key: str, adapter_path: str) -> dict | None:
    """Dispatch to a `tasks.eval_<bench_key>` module if available."""
    module_name = f"tasks.eval_{bench_key}"
    fn_name = f"evaluate_{bench_key}"
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError:
        logger.warning("No runner for custom bench %s (%s).", bench_key, module_name)
        return None
    fn = getattr(mod, fn_name, None)
    if fn is None:
        logger.warning("Module %s has no %s.", module_name, fn_name)
        return None
    return fn(model_key, adapter_path=adapter_path)


def evaluate_heldout(adapter_dir: Path, model_key: str | None = None,
                     domain_key: str | None = None):
    """Evaluate one adapter on the held-out benchmark configured for its domain."""
    run_dir = adapter_dir.parent if adapter_dir.name == "adapter" else adapter_dir

    if not model_key or not domain_key:
        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            print(f"No run_meta.json in {run_dir}, skipping")
            return
        meta = json.loads(meta_path.read_text())
        model_key = meta["model"]
        domain_key = meta["domain"]

    bench_key = DOMAINS.get(domain_key, {}).get("heldout_bench")
    if not bench_key:
        print(f"No held-out benchmark configured for domain {domain_key}")
        return

    bench = BENCHMARKS[bench_key]
    cfg = STUDENT_MODELS[model_key]
    adapter_path = str(adapter_dir)
    method = run_dir.name.split("_")[0]

    output_path = run_dir / f"heldout_{bench_key}.json"
    if output_path.exists():
        print(f"  {bench_key}: already evaluated, skipping")
        return

    print(f"\n  Held-out eval: {run_dir.name} -> {bench_key}")

    if bench.get("custom"):
        result = _run_custom_bench(bench_key, model_key, adapter_path)
        if result:
            output_path.write_text(json.dumps(result, indent=2))
    else:
        model_args = f"pretrained={cfg['hf_id']},peft={adapter_path},dtype=float16"
        if cfg.get("lm_eval_args_extra"):
            model_args += f",{cfg['lm_eval_args_extra']}"
        cmd = [
            "lm_eval",
            "--model", cfg.get("lm_eval_type", "hf"),
            "--model_args", model_args,
            "--tasks", bench["lm_eval_task"],
            "--device", "cuda",
            "--batch_size", str(bench.get("batch_size", 4)),
            "--output_path", str(output_path),
        ]
        if bench.get("num_fewshot"):
            cmd += ["--num_fewshot", str(bench["num_fewshot"])]
        print(f"  Running: {' '.join(cmd[:6])}...")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAILED: {r.stderr[:300]}")
            return

    accuracy = extract_accuracy_from_lm_eval(output_path)
    logger.info("HELDOUT model=%s domain=%s bench=%s method=%s acc=%.4f adapter=%s",
                model_key, domain_key, bench_key, method, accuracy, adapter_path)

    save_results(
        [EvalResult(model=model_key, task=f"heldout_{bench_key}",
                    accuracy=accuracy, n_samples=-1,
                    experiment=method, driver_path=adapter_path)],
        str(RESULTS_CSV),
    )


def evaluate_results_dir(results_dir: Path):
    print(f"\nScanning {results_dir} for completed adapters...")
    count = 0
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        adapter_dir = run_dir / "adapter"
        if adapter_dir.exists():
            evaluate_heldout(adapter_dir)
            count += 1
    print(f"\nEvaluated {count} adapters on held-out benchmarks.")


def main():
    parser = argparse.ArgumentParser(description="Held-out generalisation evaluation")
    parser.add_argument("--adapter-dir", type=str)
    parser.add_argument("--model", type=str, choices=list(STUDENT_MODELS.keys()))
    parser.add_argument("--domain", type=str, choices=list(DOMAINS.keys()))
    parser.add_argument("--results-dir", type=str)
    args = parser.parse_args()

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    if args.results_dir:
        evaluate_results_dir(Path(args.results_dir))
    elif args.adapter_dir:
        evaluate_heldout(Path(args.adapter_dir), args.model, args.domain)
    else:
        parser.error("Provide --adapter-dir or --results-dir")


if __name__ == "__main__":
    main()
