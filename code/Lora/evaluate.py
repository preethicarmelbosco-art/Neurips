"""Post-training evaluation for LoRA adapters on CogBench.

Evaluates trained adapters on the primary + held-out benchmarks defined in
`configs/seeds.py`, plus WikiText-2 PPL.

Usage:
    python evaluate.py --run-dir results/lora_llama-3-8b_math_seed42
    python evaluate.py --eval-all
    python evaluate.py --eval-all --wandb
    python evaluate.py --sync-wandb
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.seeds import BENCHMARKS, DOMAINS, SEEDS, STUDENT_MODELS, WANDB_PROJECT
from utils.metrics import EvalResult, save_results

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_CSV = RESULTS_DIR / "lora_eval_results.csv"
RESULTS_JSON = RESULTS_DIR / "eval_results.json"


def _parse_lm_eval_output(output_path: Path) -> dict:
    """Parse lm_eval --output_path results into a flat metrics dict.

    lm_eval 0.4.x creates a directory tree:
        output_path/{model_name}/results_{timestamp}.json
    Each results JSON has {"results": {"task": {"metric,filter": value, ...}}}.
    """
    metrics = {}
    if not output_path.exists():
        return metrics

    # Find the results JSON inside the output directory tree
    results_files = []
    if output_path.is_dir():
        results_files = sorted(output_path.rglob("results_*.json"))
    elif output_path.is_file() and output_path.suffix == ".json":
        results_files = [output_path]

    for rfile in results_files:
        try:
            with open(rfile) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        for task_name, task_metrics in data.get("results", {}).items():
            for key, value in task_metrics.items():
                if key == "alias" or not isinstance(value, (int, float)):
                    continue
                # e.g. "minerva_math/exact_match" or "copa/acc"
                clean_key = key.replace(",none", "").replace(",flexible-extract", "")
                metrics[f"{task_name}/{clean_key}"] = value

    return metrics


def _collect_adapter_benchmarks(run_dir: Path) -> dict:
    """Collect all benchmark results for an adapter into a flat dict."""
    benchmarks = {}

    # lm_eval directory outputs
    for eval_dir in sorted(run_dir.glob("eval_*")):
        if eval_dir.is_dir():
            bench_name = eval_dir.name.replace("eval_", "")
            for rfile in eval_dir.rglob("results_*.json"):
                try:
                    with open(rfile) as f:
                        data = json.load(f)
                    # Store aggregate task (shortest name = parent group) and all subtasks
                    all_tasks = {}
                    for task_name, task_metrics in data.get("results", {}).items():
                        entry = {}
                        for k, v in task_metrics.items():
                            if k == "alias" or not isinstance(v, (int, float)):
                                continue
                            entry[k] = v
                        if entry:
                            all_tasks[task_name] = entry
                    if all_tasks:
                        # Pick the aggregate (shortest task name, e.g. "minerva_math" not "minerva_math_algebra")
                        agg_task = min(all_tasks.keys(), key=len)
                        subtasks = {k: v for k, v in all_tasks.items() if k != agg_task}
                        benchmarks[bench_name] = {"task": agg_task, "metrics": all_tasks[agg_task]}
                        if subtasks:
                            benchmarks[bench_name]["subtasks"] = subtasks
                except (json.JSONDecodeError, OSError):
                    pass

    # Legacy top-level eval JSON files (e.g. eval_hendrycks_math_<timestamp>.json)
    for legacy in sorted(run_dir.glob("eval_*.json")):
        if legacy.name == "eval_wikitext2_ppl.json":
            continue
        if "hendrycks_math" in benchmarks:
            continue  # already collected from directory output
        try:
            with open(legacy) as f:
                data = json.load(f)
            all_tasks = {}
            for task_name, task_metrics in data.get("results", {}).items():
                entry = {}
                for k, v in task_metrics.items():
                    if k == "alias" or not isinstance(v, (int, float)):
                        continue
                    entry[k] = v
                if entry:
                    all_tasks[task_name] = entry
            if all_tasks:
                agg_task = min(all_tasks.keys(), key=len)
                subtasks = {k: v for k, v in all_tasks.items() if k != agg_task}
                benchmarks["hendrycks_math"] = {"task": agg_task, "metrics": all_tasks[agg_task]}
                if subtasks:
                    benchmarks["hendrycks_math"]["subtasks"] = subtasks
        except (json.JSONDecodeError, OSError):
            pass

    # PPL
    ppl_path = run_dir / "eval_wikitext2_ppl.json"
    if ppl_path.exists():
        try:
            with open(ppl_path) as f:
                benchmarks["wikitext2_ppl"] = {"ppl": json.load(f).get("ppl")}
        except (json.JSONDecodeError, OSError):
            pass

    return benchmarks


def _update_consolidated_json(run_dir: Path, meta: dict):
    """Append/update this adapter's results in the consolidated eval_results.json."""
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            consolidated = json.load(f)
    else:
        consolidated = {
            "_meta": {"description": "Consolidated LoRA eval results — all adapters, all benchmarks"},
            "adapters": {},
        }

    adapter_key = run_dir.name
    consolidated["adapters"][adapter_key] = {
        "model": meta.get("model"),
        "domain": meta.get("domain"),
        "seed": meta.get("seed"),
        "final_train_loss": meta.get("final_loss"),
        "adapter_path": meta.get("adapter_path"),
        "benchmarks": _collect_adapter_benchmarks(run_dir),
    }
    consolidated["_meta"]["last_updated"] = str(Path(__file__).stat().st_mtime)

    from datetime import datetime
    consolidated["_meta"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(RESULTS_JSON, "w") as f:
        json.dump(consolidated, f, indent=2)
    print(f"  [consolidated] updated {adapter_key} in {RESULTS_JSON.name}")


def _init_wandb_run(meta: dict, run_dir: Path):
    """Init a wandb run for one adapter evaluation."""
    import wandb
    run_name = f"eval_lora_{meta['model']}_{meta['domain']}_seed{meta.get('seed', 0)}"
    wandb.init(
        project=WANDB_PROJECT,
        group="baselines",
        job_type="eval",
        name=run_name,
        config={
            "method": "lora",
            "model": meta["model"],
            "domain": meta["domain"],
            "seed": meta.get("seed", 0),
            "adapter_path": meta.get("adapter_path", ""),
            "run_dir": str(run_dir),
            "final_train_loss": meta.get("final_loss"),
        },
        tags=["lora", "eval", meta["model"], meta["domain"]],
    )
    return True


def evaluate_adapter(run_dir: Path, use_wandb: bool = False):
    """Evaluate a trained LoRA adapter on all benchmarks."""
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        print(f"No run_meta.json in {run_dir}, skipping")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    model_key = meta["model"]
    domain_key = meta["domain"]
    cfg = STUDENT_MODELS[model_key]
    domain = DOMAINS[domain_key]
    adapter_path = meta["adapter_path"]

    print(f"\n{'='*60}")
    print(f"Evaluating: {run_dir.name}")
    print(f"{'='*60}\n")

    # Init wandb run for this adapter
    wb_active = False
    if use_wandb:
        try:
            wb_active = _init_wandb_run(meta, run_dir)
        except Exception as e:
            print(f"  [wandb] init failed: {e}")

    # Collect all metrics for a final wandb summary
    all_metrics = {}

    # Run all lm-eval compatible benchmarks
    all_benchmarks = [
        domain["primary_bench"],
        domain["heldout_bench"],
        "gsm8k", "social_iqa", "copa",  # cross-domain benchmarks
    ]
    # Deduplicate
    all_benchmarks = list(dict.fromkeys(all_benchmarks))

    for bench_key in all_benchmarks:
        bench = BENCHMARKS.get(bench_key)
        if not bench:
            continue
        if bench.get("custom"):
            # Custom (non-lm-eval) benchmarks live in the companion `evals/`
            # package. Run them from there with the adapter path.
            print(f"  {bench_key}: custom bench — run via `python -m evals ...` (see evals/README.md)")
            continue

        # lm_eval 0.4.x: --output_path with .json suffix saves as
        # {stem}_{timestamp}.json in the parent dir. Use a directory instead
        # so results land in eval_{bench_key}/model/results_{ts}.json.
        output_dir = run_dir / f"eval_{bench_key}"
        # Also check for legacy .json files from earlier runs
        legacy_hits = list(run_dir.glob(f"eval_{bench_key}*.json"))
        already_done = output_dir.exists() or len(legacy_hits) > 0

        if already_done:
            print(f"  {bench_key}: already evaluated, skipping")
        else:
            model_args = f"pretrained={cfg['hf_id']},peft={adapter_path},dtype=float16"
            if cfg.get("lm_eval_args_extra"):
                model_args += f",{cfg['lm_eval_args_extra']}"

            cmd = [
                "lm_eval",
                "--model", cfg["lm_eval_type"],
                "--model_args", model_args,
                "--tasks", bench["lm_eval_task"],
                "--device", "cuda",
                "--batch_size", str(bench["batch_size"]),
                "--output_path", str(output_dir),
            ]
            if bench.get("num_fewshot"):
                cmd.extend(["--num_fewshot", str(bench["num_fewshot"])])
            print(f"  Running: {bench_key}")
            subprocess.run(cmd)

        # Parse and log results to wandb (whether freshly run or already existed)
        if wb_active:
            # Collect from directory output and any legacy .json files
            bench_metrics = _parse_lm_eval_output(output_dir)
            for lf in legacy_hits:
                bench_metrics.update(_parse_lm_eval_output(lf))
            if bench_metrics:
                import wandb
                wandb.log(bench_metrics)
                all_metrics.update(bench_metrics)
                print(f"  [wandb] logged {len(bench_metrics)} metrics for {bench_key}")

    # WikiText-2 perplexity
    ppl_path = run_dir / "eval_wikitext2_ppl.json"
    ppl = None
    if ppl_path.exists():
        print(f"  wikitext2_ppl: already evaluated, skipping")
        try:
            with open(ppl_path) as f:
                ppl = json.load(f).get("ppl")
        except (json.JSONDecodeError, OSError):
            pass
    else:
        from utils.model_loader import load_student
        from utils.perplexity import compute_wikitext2_ppl
        from peft import PeftModel
        import torch

        print(f"  Computing WikiText-2 PPL...")
        model, tokenizer, _ = load_student(model_key)
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

        ppl = compute_wikitext2_ppl(model, tokenizer)

        result = EvalResult(
            model=model_key,
            task="wikitext2_ppl",
            accuracy=ppl,
            n_samples=-1,
            seed=meta.get("seed", 0),
            experiment="lora",
            driver_path=adapter_path,
        )
        save_results([result], str(RESULTS_CSV))

        import json as _json
        with open(ppl_path, "w") as _f:
            _json.dump({"model": model_key, "domain": domain_key,
                        "seed": meta.get("seed"), "ppl": ppl}, _f, indent=2)
        print(f"  WikiText-2 PPL: {ppl:.2f}")

        del model
        torch.cuda.empty_cache()

    # Log PPL to wandb
    if wb_active and ppl is not None:
        import wandb
        wandb.log({"wikitext2_ppl": ppl})
        all_metrics["wikitext2_ppl"] = ppl
        print(f"  [wandb] logged wikitext2_ppl={ppl:.2f}")

    # Finish wandb run
    if wb_active:
        import wandb
        # Set summary with all metrics for easy table view
        for k, v in all_metrics.items():
            wandb.run.summary[k] = v
        wandb.finish()
        print(f"  [wandb] run finished")

    # Update consolidated JSON
    _update_consolidated_json(run_dir, meta)


def eval_all(use_wandb: bool = False):
    """Evaluate all completed LoRA training runs."""
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if run_dir.is_dir() and (run_dir / "run_meta.json").exists():
            evaluate_adapter(run_dir, use_wandb=use_wandb)


def sync_wandb():
    """Log existing eval results to wandb without re-running evals."""
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if run_dir.is_dir() and (run_dir / "run_meta.json").exists():
            evaluate_adapter(run_dir, use_wandb=True)


def main():
    parser = argparse.ArgumentParser(description="LoRA adapter evaluation")
    parser.add_argument("--run-dir", type=str)
    parser.add_argument("--eval-all", action="store_true")
    parser.add_argument("--wandb", action="store_true",
                        help=f"Log eval results to wandb (project: {WANDB_PROJECT})")
    parser.add_argument("--sync-wandb", action="store_true",
                        help="Log existing results to wandb without re-running evals")
    args = parser.parse_args()

    if args.sync_wandb:
        sync_wandb()
    elif args.eval_all:
        eval_all(use_wandb=args.wandb)
    elif args.run_dir:
        evaluate_adapter(Path(args.run_dir), use_wandb=args.wandb)
    else:
        parser.error("Provide --run-dir, --eval-all, or --sync-wandb")


if __name__ == "__main__":
    main()
