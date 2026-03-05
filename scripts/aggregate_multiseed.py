#!/usr/bin/env python3
"""Aggregate multi-seed LOSO results.

Produces two analyses:
  1. Per-model mean +/- std across seeds (model robustness)
  2. Per-subject mean +/- std across seeds (subject consistency)

Usage:
  python scripts/aggregate_multiseed.py [results_dir/]
  python scripts/aggregate_multiseed.py --output results/aggregated.json
"""

import json
import sys
import argparse
from pathlib import Path
import numpy as np


SEEDS = [42, 123, 456, 789, 1024]
MODELS = ["musdet", "husformer", "h2", "phemonet", "hyperfusenet"]
DATASETS = ["wesad", "affectiveroad"]

MODEL_NAMES = {
    "musdet": "MuSDeT (Ours)",
    "husformer": "Husformer",
    "h2": "H2",
    "phemonet": "PHemoNet",
    "hyperfusenet": "HyperFuseNet",
}

WESAD_SUBJECTS = ['S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S13','S14','S15','S16','S17']
AR_SUBJECTS = ['Drv1','Drv3','Drv4','Drv5','Drv6','Drv7','Drv8','Drv9','Drv10','Drv11','Drv12','Drv13']


def load_fold_metrics(result_dir: Path):
    """Load per-fold metrics from individual fold directories, in numeric order."""
    fold_dirs = sorted(
        [d for d in result_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")],
        key=lambda d: int(d.name.split("_")[1])
    )
    metrics = []
    for fd in fold_dirs:
        mpath = fd / "metrics.json"
        if not mpath.exists():
            raise FileNotFoundError(f"Missing {mpath}")
        with open(mpath) as f:
            data = json.load(f)
        metrics.append({
            "fold": fd.name,
            "subject": data.get("test_subject", fd.name),
            "accuracy": data["test_metrics"]["accuracy"],
            "f1_macro": data["test_metrics"]["f1_macro"],
        })
    return metrics


def load_all_results(base_dir: Path):
    """Load results for all seeds x models x datasets."""
    results = {}
    for seed in SEEDS:
        results[seed] = {}
        for dataset in DATASETS:
            results[seed][dataset] = {}
            for model in MODELS:
                rdir = base_dir / f"seed_{seed}" / dataset / model
                if not rdir.exists():
                    print(f"WARNING: missing {rdir}", file=sys.stderr)
                    continue
                results[seed][dataset][model] = load_fold_metrics(rdir)
    return results


def aggregate_across_seeds(results):
    """Compute mean +/- std of LOSO accuracy/F1 across seeds for each model x dataset."""
    tables = {}
    for dataset in DATASETS:
        tables[dataset] = {}
        for model in MODELS:
            seed_accs = []
            seed_f1s = []
            for seed in SEEDS:
                folds = results[seed][dataset].get(model, [])
                if not folds:
                    continue
                accs = [f["accuracy"] for f in folds]
                f1s = [f["f1_macro"] for f in folds]
                seed_accs.append(np.mean(accs) * 100)
                seed_f1s.append(np.mean(f1s) * 100)
            tables[dataset][model] = {
                "acc_mean": np.mean(seed_accs),
                "acc_std": np.std(seed_accs),
                "f1_mean": np.mean(seed_f1s),
                "f1_std": np.std(seed_f1s),
                "seed_accs": seed_accs,
                "seed_f1s": seed_f1s,
                "n_seeds": len(seed_accs),
            }
    return tables


def per_subject_consistency(results):
    """For each subject, compute accuracy mean +/- std across seeds."""
    tables = {}
    for dataset in DATASETS:
        subjects = WESAD_SUBJECTS if dataset == "wesad" else AR_SUBJECTS
        tables[dataset] = {}

        for model in MODELS:
            subject_data = {s: [] for s in subjects}
            for seed in SEEDS:
                folds = results[seed][dataset].get(model, [])
                if not folds:
                    continue
                for fold in folds:
                    subj = fold["subject"]
                    subject_data[subj].append(fold["accuracy"] * 100)

            tables[dataset][model] = {}
            for subj in subjects:
                vals = subject_data[subj]
                if vals:
                    tables[dataset][model][subj] = {
                        "mean": np.mean(vals),
                        "std": np.std(vals),
                        "values": vals,
                    }
    return tables


def print_seed_robustness(tables):
    """Print model robustness across seeds."""
    print("## Multi-Seed Robustness (n=5 seeds)")
    print()
    for dataset in DATASETS:
        ds_label = "WESAD" if dataset == "wesad" else "AffectiveROAD"
        print(f"### {ds_label}")
        print()
        print(f"| Model | Acc (seed mean+/-std) | F1 (seed mean+/-std) |")
        print(f"|-------|----------------------|---------------------|")
        for model in MODELS:
            d = tables[dataset][model]
            print(f"| {MODEL_NAMES[model]} | {d['acc_mean']:.1f}+/-{d['acc_std']:.1f} | "
                  f"{d['f1_mean']:.1f}+/-{d['f1_std']:.1f} |")
        print()


def print_subject_consistency(subj_tables, model="musdet"):
    """Print per-subject consistency."""
    print(f"## Per-Subject Consistency -- {MODEL_NAMES[model]} (n=5 seeds)")
    print()
    for dataset in DATASETS:
        ds_label = "WESAD" if dataset == "wesad" else "AffectiveROAD"
        subjects = WESAD_SUBJECTS if dataset == "wesad" else AR_SUBJECTS
        print(f"### {ds_label}")
        print()
        print(f"| Subject | Acc mean+/-std | Verdict |")
        print(f"|---------|---------------|---------|")
        model_data = subj_tables[dataset].get(model, {})
        for subj in subjects:
            d = model_data.get(subj)
            if not d:
                continue
            if d["mean"] < 30:
                verdict = "consistently hard"
            elif d["mean"] > 80:
                verdict = "consistently easy"
            elif d["std"] > 10:
                verdict = "unstable"
            else:
                verdict = ""
            print(f"| {subj} | {d['mean']:.1f}+/-{d['std']:.1f} | {verdict} |")
        print()


def save_json(tables, subj_tables, output_path):
    """Save aggregated results as JSON."""
    out = {
        "seeds": SEEDS,
        "seed_robustness": {},
        "subject_consistency": {},
    }
    for dataset in DATASETS:
        out["seed_robustness"][dataset] = {}
        for model in MODELS:
            d = tables[dataset][model]
            out["seed_robustness"][dataset][model] = {
                "acc_mean": round(d["acc_mean"], 2),
                "acc_std": round(d["acc_std"], 2),
                "f1_mean": round(d["f1_mean"], 2),
                "f1_std": round(d["f1_std"], 2),
                "seed_accs": [round(v, 2) for v in d["seed_accs"]],
                "seed_f1s": [round(v, 2) for v in d["seed_f1s"]],
            }
        out["subject_consistency"][dataset] = {}
        for model in MODELS:
            out["subject_consistency"][dataset][model] = {}
            for subj, d in subj_tables[dataset].get(model, {}).items():
                out["subject_consistency"][dataset][model][subj] = {
                    "acc_mean": round(d["mean"], 2),
                    "acc_std": round(d["std"], 2),
                    "acc_per_seed": [round(v, 2) for v in d["values"]],
                }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved aggregated results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed LOSO results")
    parser.add_argument("results_dir", nargs="?", default="results_multiseed",
                        help="Path to directory with seed_* subdirectories")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save aggregated JSON to this path")
    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    if not base_dir.exists():
        print(f"ERROR: {base_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Loading results from {base_dir} ...")
    results = load_all_results(base_dir)

    tables = aggregate_across_seeds(results)
    subj_tables = per_subject_consistency(results)

    print()
    print_seed_robustness(tables)
    print_subject_consistency(subj_tables, model="musdet")

    if args.output:
        save_json(tables, subj_tables, args.output)


if __name__ == "__main__":
    main()
