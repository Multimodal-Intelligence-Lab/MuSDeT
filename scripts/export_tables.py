#!/usr/bin/env python3
"""Export aggregated results into paper-ready CSV/JSON files.

Reads from aggregated.json and produces:
  - distilled_table1.csv
  - distilled_multiseed.csv
  - distilled_per_subject.csv
  - distilled_summary.json

Usage:
  python scripts/export_tables.py [results_dir/]
  python scripts/export_tables.py results_dir/ --outdir results_dir/distilled/
"""

import json
import csv
import argparse
from pathlib import Path


MODEL_NAMES = {
    "musdet": "MuSDeT (Ours)",
    "husformer": "Husformer",
    "h2": "H2",
    "phemonet": "PHemoNet",
    "hyperfusenet": "HyperFuseNet",
}
MODEL_ORDER = ["musdet", "husformer", "h2", "phemonet", "hyperfusenet"]
DATASET_NAMES = {"wesad": "WESAD", "affectiveroad": "AffectiveROAD"}
PARAMS = {"musdet": "338K", "husformer": "478K", "h2": "15.7M",
          "phemonet": "2.4M", "hyperfusenet": "6.6M"}

WESAD_SUBJECTS = ['S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S13','S14','S15','S16','S17']
AR_SUBJECTS = ['Drv1','Drv3','Drv4','Drv5','Drv6','Drv7','Drv8','Drv9','Drv10','Drv11','Drv12','Drv13']


def load_aggregated(base_dir: Path):
    with open(base_dir / "aggregated.json") as f:
        return json.load(f)


def write_table1(data, outdir: Path):
    """Table 1: seed=42 results (first seed in SEEDS list)."""
    path = outdir / "distilled_table1.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Params", "WESAD_Acc", "WESAD_F1", "AR_Acc", "AR_F1"])
        for model in MODEL_ORDER:
            row = [MODEL_NAMES[model], PARAMS[model]]
            for dataset in ["wesad", "affectiveroad"]:
                d = data["seed_robustness"][dataset][model]
                row.extend([f"{d['seed_accs'][0]:.1f}", f"{d['seed_f1s'][0]:.1f}"])
            w.writerow(row)
    print(f"  {path}")


def write_multiseed(data, outdir: Path):
    """Multi-seed robustness table (n=5 seeds)."""
    path = outdir / "distilled_multiseed.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "WESAD_Acc_mean", "WESAD_Acc_seed_std",
                     "WESAD_F1_mean", "WESAD_F1_seed_std",
                     "AR_Acc_mean", "AR_Acc_seed_std",
                     "AR_F1_mean", "AR_F1_seed_std"])
        for model in MODEL_ORDER:
            row = [MODEL_NAMES[model]]
            for dataset in ["wesad", "affectiveroad"]:
                d = data["seed_robustness"][dataset][model]
                row.extend([f"{d['acc_mean']:.1f}", f"{d['acc_std']:.1f}",
                            f"{d['f1_mean']:.1f}", f"{d['f1_std']:.1f}"])
            w.writerow(row)
    print(f"  {path}")


def write_per_subject(data, outdir: Path):
    """Per-subject consistency for MuSDeT."""
    path = outdir / "distilled_per_subject.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Subject", "Acc_mean", "Acc_std",
                     "Seed_42", "Seed_123", "Seed_456", "Seed_789", "Seed_1024",
                     "Verdict"])
        for dataset in ["wesad", "affectiveroad"]:
            subjects = WESAD_SUBJECTS if dataset == "wesad" else AR_SUBJECTS
            subj_data = data["subject_consistency"][dataset].get("musdet", {})
            for subj in subjects:
                d = subj_data.get(subj)
                if not d:
                    continue
                mean, std = d["acc_mean"], d["acc_std"]
                if mean < 30:
                    verdict = "consistently hard"
                elif mean > 80:
                    verdict = "consistently easy"
                elif std > 10:
                    verdict = "unstable"
                else:
                    verdict = ""
                row = [DATASET_NAMES[dataset], subj, f"{mean:.1f}", f"{std:.1f}"]
                row.extend([f"{v:.1f}" for v in d["acc_per_seed"]])
                row.append(verdict)
                w.writerow(row)
    print(f"  {path}")


def write_summary_json(data, outdir: Path):
    """Compact JSON with paper-ready numbers."""
    summary = {
        "description": "Distilled results for MuSDeT (CVPR 2026)",
        "seeds": data["seeds"],
        "table1_seed42": {},
        "multiseed_robustness": {},
        "margins": {},
    }
    for dataset in ["wesad", "affectiveroad"]:
        ds = DATASET_NAMES[dataset]
        summary["table1_seed42"][ds] = {}
        for model in MODEL_ORDER:
            d = data["seed_robustness"][dataset][model]
            summary["table1_seed42"][ds][MODEL_NAMES[model]] = {
                "acc": d["seed_accs"][0], "f1": d["seed_f1s"][0], "params": PARAMS[model],
            }
        summary["multiseed_robustness"][ds] = {}
        for model in MODEL_ORDER:
            d = data["seed_robustness"][dataset][model]
            summary["multiseed_robustness"][ds][MODEL_NAMES[model]] = {
                "acc_mean": d["acc_mean"], "acc_std": d["acc_std"],
                "f1_mean": d["f1_mean"], "f1_std": d["f1_std"],
            }
    for label, idx_or_key in [("seed42", 0), ("multiseed", "mean")]:
        summary["margins"][label] = {}
        for dataset in ["wesad", "affectiveroad"]:
            ds = DATASET_NAMES[dataset]
            ours = data["seed_robustness"][dataset]["musdet"]
            our_acc = ours["seed_accs"][0] if label == "seed42" else ours["acc_mean"]
            our_f1 = ours["seed_f1s"][0] if label == "seed42" else ours["f1_mean"]
            best_acc, best_f1, best_acc_name, best_f1_name = -1, -1, "", ""
            for model in MODEL_ORDER[1:]:  # skip musdet
                d = data["seed_robustness"][dataset][model]
                bacc = d["seed_accs"][0] if label == "seed42" else d["acc_mean"]
                bf1 = d["seed_f1s"][0] if label == "seed42" else d["f1_mean"]
                if bacc > best_acc:
                    best_acc, best_acc_name = bacc, MODEL_NAMES[model]
                if bf1 > best_f1:
                    best_f1, best_f1_name = bf1, MODEL_NAMES[model]
            summary["margins"][label][ds] = {
                "acc_margin_pp": round(our_acc - best_acc, 1), "acc_vs": best_acc_name,
                "f1_margin_pp": round(our_f1 - best_f1, 1), "f1_vs": best_f1_name,
            }
    path = outdir / "distilled_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  {path}")


def main():
    parser = argparse.ArgumentParser(description="Export results to paper-ready files")
    parser.add_argument("results_dir", nargs="?", default="results_multiseed")
    parser.add_argument("--outdir", "-o", type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    outdir = Path(args.outdir) if args.outdir else base_dir / "distilled"
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_aggregated(base_dir)
    print(f"Exporting from {base_dir / 'aggregated.json'} -> {outdir}/")
    write_table1(data, outdir)
    write_multiseed(data, outdir)
    write_per_subject(data, outdir)
    write_summary_json(data, outdir)
    print("Done.")


if __name__ == "__main__":
    main()
