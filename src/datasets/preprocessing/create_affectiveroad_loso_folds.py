#!/usr/bin/env python3
"""
Create Leave-One-Subject-Out (LOSO) folds for AffectiveROAD dataset.

Mirrors the WESAD LOSO fold creation (create_loso_folds.py) with:
- 12 folds (one per subject)
- 10 train / 1 val / 1 test subjects per fold
- Deterministic val selection (next in sorted order)
- Phase boundaries preserved for temporal context models

Input: affectiveroad_with_subjects.pkl (from parse_raw_affectiveroad.py)
Output: loso_folds.pkl
"""

import pickle
import numpy as np
from pathlib import Path
import json

# Paths
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "affectiveroad"
INPUT_PATH = DATA_DIR / "affectiveroad_with_subjects.pkl"
OUTPUT_PATH = DATA_DIR / "loso_folds.pkl"


def sort_subjects_numeric(subject_ids: list) -> list:
    """Sort subjects by numeric suffix (Drv1 before Drv10)."""
    return sorted(subject_ids, key=lambda s: int(s.replace('Drv', '')))


def create_loso_folds(subjects_data: dict, subject_ids: list) -> dict:
    """
    Create LOSO fold definitions with nested validation.

    For each fold:
    - 1 subject is held out for TEST
    - 1 subject is held out for VAL (deterministic: next in sorted order)
    - Remaining 10 subjects for TRAINING

    Matches WESAD LOSO protocol exactly.
    """
    n_subjects = len(subject_ids)
    sorted_subjects = sort_subjects_numeric(subject_ids)

    folds = {}

    for i, test_subject in enumerate(sorted_subjects):
        # Deterministic val selection: next in sorted order (wrap around)
        val_idx = (i + 1) % n_subjects
        val_subject = sorted_subjects[val_idx]

        # Train on remaining subjects
        train_subjects = [s for s in sorted_subjects
                         if s != test_subject and s != val_subject]

        # Collect windows for each split
        train_windows = []
        train_labels = []
        train_subject_ids = []
        train_phase_ids = []

        for subj in train_subjects:
            subj_data = subjects_data[subj]
            n_windows = len(subj_data['windows'])
            train_windows.extend(subj_data['windows'])
            train_labels.extend(subj_data['labels'])
            train_subject_ids.extend([subj] * n_windows)
            # Make phase_ids globally unique by prefixing with subject
            train_phase_ids.extend([
                f"{subj}_phase_{pid}" for pid in subj_data['phase_ids']
            ])

        # Validation split
        val_data = subjects_data[val_subject]
        val_windows = val_data['windows']
        val_labels = val_data['labels']
        val_subject_ids = [val_subject] * len(val_windows)
        val_phase_ids = [f"{val_subject}_phase_{pid}" for pid in val_data['phase_ids']]

        # Test split
        test_data = subjects_data[test_subject]
        test_windows = test_data['windows']
        test_labels = test_data['labels']
        test_phase_ids = [f"{test_subject}_phase_{pid}" for pid in test_data['phase_ids']]

        folds[f'fold_{i}'] = {
            'test_subject': test_subject,
            'val_subject': val_subject,
            'train_subjects': train_subjects,
            'train': {
                'windows': train_windows,
                'labels': np.array(train_labels),
                'subject_ids': train_subject_ids,
                'phase_ids': train_phase_ids,
                'n_samples': len(train_windows)
            },
            'val': {
                'windows': val_windows,
                'labels': np.array(val_labels),
                'subject_ids': val_subject_ids,
                'phase_ids': val_phase_ids,
                'n_samples': len(val_windows)
            },
            'test': {
                'windows': test_windows,
                'labels': np.array(test_labels),
                'subject_id': test_subject,
                'phase_ids': test_phase_ids,
                'n_samples': len(test_windows)
            }
        }

    return folds


def main():
    print("=" * 60)
    print("Creating LOSO Folds for AffectiveROAD (10 train / 1 val / 1 test)")
    print("=" * 60)
    print("\nMETHODOLOGY:")
    print("  - Val subject = next in sorted order (deterministic)")
    print("  - Early stopping, LR scheduling, temp scaling use VAL only")
    print("  - Test touched ONCE per fold at the end")

    if not INPUT_PATH.exists():
        print(f"\nError: {INPUT_PATH} not found")
        print("Run parse_raw_affectiveroad.py first")
        return

    with open(INPUT_PATH, 'rb') as f:
        data = pickle.load(f)

    subjects_data = data['subjects']
    subject_ids = data['subject_ids']

    print(f"\nLoaded {len(subject_ids)} subjects")

    # Create folds
    folds = create_loso_folds(subjects_data, subject_ids)

    # Print fold statistics
    n_train_subjects = len(subject_ids) - 2
    print(f"\n{'=' * 60}")
    print(f"FOLD STATISTICS ({n_train_subjects} train / 1 val / 1 test)")
    print("=" * 60)

    total_train = 0
    total_val = 0
    total_test = 0

    for fold_name, fold_data in sorted(folds.items()):
        test_subj = fold_data['test_subject']
        val_subj = fold_data['val_subject']
        n_train = fold_data['train']['n_samples']
        n_val = fold_data['val']['n_samples']
        n_test = fold_data['test']['n_samples']

        total_train += n_train
        total_val += n_val
        total_test += n_test

        # Label distribution in test
        test_labels = fold_data['test']['labels']
        stress_pct = np.mean(test_labels) * 100

        print(f"{fold_name}: test={test_subj} ({n_test}, {stress_pct:.0f}% stress), "
              f"val={val_subj} ({n_val}), "
              f"train={n_train_subjects} subjects ({n_train})")

    n_folds = len(folds)
    print(f"\nAverages across {n_folds} folds:")
    print(f"  Train: {total_train / n_folds:.0f} samples ({n_train_subjects} subjects)")
    print(f"  Val: {total_val / n_folds:.0f} samples (1 subject)")
    print(f"  Test: {total_test / n_folds:.0f} samples (1 subject)")

    # Verify no overlap
    print(f"\n{'=' * 60}")
    print("VALIDATION: Checking for subject overlap")
    print("=" * 60)

    all_valid = True
    for fold_name, fold_data in folds.items():
        train_set = set(fold_data['train_subjects'])
        val_set = {fold_data['val_subject']}
        test_set = {fold_data['test_subject']}

        if train_set & val_set:
            print(f"ERROR: {fold_name} has train/val overlap")
            all_valid = False
        if train_set & test_set:
            print(f"ERROR: {fold_name} has train/test overlap")
            all_valid = False
        if val_set & test_set:
            print(f"ERROR: {fold_name} has val/test overlap")
            all_valid = False

    if all_valid:
        print("All folds have disjoint train/val/test subjects")

    # Save folds
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump({
            'folds': folds,
            'subject_ids': subject_ids,
            'n_folds': len(folds),
            'protocol': {
                'n_train_subjects': n_train_subjects,
                'n_val_subjects': 1,
                'n_test_subjects': 1,
                'val_selection_rule': 'deterministic_rotation_next_in_sorted_order',
                'temperature_scaling': 'val_only',
                'model_selection': 'val_only',
                'test_usage': 'once_per_fold_at_end'
            }
        }, f)

    print(f"\nSaved to: {OUTPUT_PATH}")

    # Save summary JSON
    summary = {
        'n_folds': len(folds),
        'subject_ids': sort_subjects_numeric(subject_ids),
        'protocol': {
            'n_train_subjects': n_train_subjects,
            'n_val_subjects': 1,
            'n_test_subjects': 1,
            'val_selection_rule': 'deterministic_rotation_next_in_sorted_order'
        },
        'fold_summary': {}
    }
    for fold_name, fold_data in sorted(folds.items()):
        summary['fold_summary'][fold_name] = {
            'test_subject': fold_data['test_subject'],
            'val_subject': fold_data['val_subject'],
            'train_subjects': fold_data['train_subjects'],
            'n_train': fold_data['train']['n_samples'],
            'n_val': fold_data['val']['n_samples'],
            'n_test': fold_data['test']['n_samples']
        }

    summary_path = DATA_DIR / "loso_folds_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
