"""
LOSO (Leave-One-Subject-Out) Evaluation Loop

Handles fold generation, training, evaluation, and result aggregation
for both novel and baseline models.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


def aggregate_fold_results(results_dir: Path) -> Dict:
    """
    Aggregate per-fold metrics into summary statistics.

    Args:
        results_dir: Directory containing fold_*/metrics.json files

    Returns:
        Summary dict with mean/std across folds
    """
    fold_dirs = sorted(results_dir.glob("fold_*"))
    results = []

    for fold_dir in fold_dirs:
        metrics_file = fold_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                results.append(json.load(f))

    if not results:
        return {'error': 'No fold results found'}

    # Extract test metrics
    accuracies = [r['test_metrics']['accuracy'] for r in results]
    f1_macros = [r['test_metrics']['f1_macro'] for r in results]
    f1_weighteds = [r['test_metrics'].get('f1_weighted', 0) for r in results]
    nlls = [r['test_metrics'].get('nll_bits', 0) for r in results]

    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_folds': len(results),
        'accuracy_mean': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'f1_macro_mean': float(np.mean(f1_macros)),
        'f1_macro_std': float(np.std(f1_macros)),
        'f1_weighted_mean': float(np.mean(f1_weighteds)),
        'f1_weighted_std': float(np.std(f1_weighteds)),
        'nll_bits_mean': float(np.mean(nlls)),
        'nll_bits_std': float(np.std(nlls)),
        'per_fold_accuracy': accuracies,
        'per_fold_f1_macro': f1_macros,
    }

    return summary


def print_summary(summary: Dict):
    """Print formatted summary of LOSO results."""
    print(f"\n{'='*60}")
    print(f"LOSO SUMMARY ({summary['n_folds']} folds)")
    print(f"{'='*60}")
    print(f"Accuracy: {summary['accuracy_mean']*100:.2f}% ± {summary['accuracy_std']*100:.2f}%")
    print(f"F1 Macro: {summary['f1_macro_mean']*100:.2f}% ± {summary['f1_macro_std']*100:.2f}%")
    print(f"NLL:      {summary['nll_bits_mean']:.4f} ± {summary['nll_bits_std']:.4f} bits")
    print(f"{'='*60}\n")
