#!/usr/bin/env python3
"""
Parse raw WESAD .pkl files and create unified dataset with subject IDs and phase boundaries.

This script:
1. Loads raw S*.pkl files from WESAD directory
2. Applies windowing (1-second windows) matching Husformer preprocessing
3. Extracts 6 modalities: GSR_chest, BVP_wrist, EMG_chest, ECG_chest, RESP_chest, GSR_wrist
4. Preserves subject IDs, temporal ordering, AND phase boundaries
5. Filters to classes 1 (baseline), 2 (stress), 3 (amusement)

CRITICAL: Phase boundaries are preserved to prevent "experiment-script leakage"
in temporal context modeling. Sequences must not cross phase boundaries.

Output: wesad_with_subjects.pkl containing all windows with subject + phase info
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from scipy import signal
from tqdm import tqdm
import json

# Paths
WESAD_RAW_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "WESAD"
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "data" / "wesad"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sampling rates (from WESAD documentation)
CHEST_RATE = 700  # Hz
WRIST_BVP_RATE = 64  # Hz
WRIST_EDA_RATE = 4  # Hz

# Window parameters (match Husformer)
WINDOW_SIZE_SEC = 1.0
CHEST_WINDOW = int(WINDOW_SIZE_SEC * CHEST_RATE)  # 700 samples
WRIST_BVP_WINDOW = int(WINDOW_SIZE_SEC * WRIST_BVP_RATE)  # 64 samples
WRIST_EDA_WINDOW = int(WINDOW_SIZE_SEC * WRIST_EDA_RATE)  # 4 samples

# Labels to keep (1=baseline, 2=stress, 3=amusement)
VALID_LABELS = [1, 2, 3]
LABEL_REMAP = {1: 0, 2: 1, 3: 2}  # Remap to 0, 1, 2
LABEL_NAMES = {0: 'baseline', 1: 'stress', 2: 'amusement'}

# Subject IDs (15 subjects, S12 excluded in original dataset)
SUBJECT_IDS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9',
               'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']


def detect_phase_boundaries(labels: np.ndarray, valid_labels: list) -> list:
    """
    Detect phase boundaries (contiguous segments of same label).

    Returns list of (start_idx, end_idx, label) tuples for valid labels.
    This is critical for preventing temporal context from crossing phases.
    """
    phases = []
    current_label = None
    start_idx = None

    for i, label in enumerate(labels):
        if label in valid_labels:
            if label != current_label:
                # End previous phase if exists
                if current_label is not None and current_label in valid_labels:
                    phases.append((start_idx, i, current_label))
                # Start new phase
                current_label = label
                start_idx = i
        else:
            # End current phase if in valid label
            if current_label is not None and current_label in valid_labels:
                phases.append((start_idx, i, current_label))
            current_label = label
            start_idx = i

    # Handle last phase
    if current_label is not None and current_label in valid_labels:
        phases.append((start_idx, len(labels), current_label))

    return phases


def extract_windows_for_subject(subject_id: str) -> dict:
    """
    Extract all 1-second windows for a single subject with phase boundaries.

    Returns dict with:
        'windows': list of dicts with 6 modalities
        'labels': list of labels (0, 1, 2)
        'subject_id': subject ID string
        'window_indices': original indices for temporal ordering
        'phase_ids': phase ID for each window (for temporal sampling)
        'phase_info': dict mapping phase_id to (label, start_window, end_window)
    """
    pkl_path = WESAD_RAW_DIR / subject_id / f"{subject_id}.pkl"

    if not pkl_path.exists():
        print(f"Warning: {pkl_path} not found, skipping")
        return None

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Extract signals
    chest = data['signal']['chest']
    wrist = data['signal']['wrist']
    labels = np.array(data['label'])

    # Chest signals (700 Hz)
    ecg = chest['ECG'].flatten()
    emg = chest['EMG'].flatten()
    gsr_chest = chest['EDA'].flatten()  # EDA = GSR
    resp = chest['Resp'].flatten()

    # Wrist signals (variable rates)
    bvp = wrist['BVP'].flatten()  # 64 Hz
    gsr_wrist = wrist['EDA'].flatten()  # 4 Hz

    # Detect phase boundaries at sample level
    raw_phases = detect_phase_boundaries(labels, VALID_LABELS)

    # Convert to window-level phases
    windows = []
    window_labels = []
    window_indices = []
    phase_ids = []
    phase_info = {}

    current_phase_id = 0

    for phase_start_sample, phase_end_sample, raw_label in raw_phases:
        # Convert sample indices to window indices
        phase_start_window = phase_start_sample // CHEST_WINDOW
        phase_end_window = phase_end_sample // CHEST_WINDOW

        remapped_label = LABEL_REMAP[raw_label]
        phase_window_start = len(windows)  # Track where this phase starts in our list

        for i in range(phase_start_window, phase_end_window):
            # Chest window bounds
            chest_start = i * CHEST_WINDOW
            chest_end = (i + 1) * CHEST_WINDOW

            if chest_end > len(ecg):
                continue

            # Get majority label for this window (should be consistent within phase)
            window_label_raw = labels[chest_start:chest_end]
            label_mode = int(np.median(window_label_raw))

            # Skip if not matching phase label (edge case)
            if label_mode != raw_label:
                continue

            # Extract chest modalities (700 samples each)
            ecg_win = ecg[chest_start:chest_end]
            emg_win = emg[chest_start:chest_end]
            gsr_chest_win = gsr_chest[chest_start:chest_end]
            resp_win = resp[chest_start:chest_end]

            # Extract wrist modalities (need to align with chest time)
            bvp_start = int(i * WRIST_BVP_WINDOW)
            bvp_end = int((i + 1) * WRIST_BVP_WINDOW)
            if bvp_end > len(bvp):
                continue
            bvp_win = bvp[bvp_start:bvp_end]

            gsr_wrist_start = int(i * WRIST_EDA_WINDOW)
            gsr_wrist_end = int((i + 1) * WRIST_EDA_WINDOW)
            if gsr_wrist_end > len(gsr_wrist):
                continue
            gsr_wrist_win = gsr_wrist[gsr_wrist_start:gsr_wrist_end]

            # Validate shapes
            if (len(ecg_win) != CHEST_WINDOW or
                len(bvp_win) != WRIST_BVP_WINDOW or
                len(gsr_wrist_win) != WRIST_EDA_WINDOW):
                continue

            # Store window
            window = {
                'GSR_chest': gsr_chest_win.astype(np.float32),
                'BVP_wrist': bvp_win.astype(np.float32),
                'EMG_chest': emg_win.astype(np.float32),
                'ECG_chest': ecg_win.astype(np.float32),
                'RESP_chest': resp_win.astype(np.float32),
                'GSR_wrist': gsr_wrist_win.astype(np.float32),
            }

            windows.append(window)
            window_labels.append(remapped_label)
            window_indices.append(i)
            phase_ids.append(current_phase_id)

        phase_window_end = len(windows)

        # Only record phase if it has windows
        if phase_window_end > phase_window_start:
            phase_info[current_phase_id] = {
                'label': remapped_label,
                'label_name': LABEL_NAMES[remapped_label],
                'start_window_idx': phase_window_start,
                'end_window_idx': phase_window_end,
                'n_windows': phase_window_end - phase_window_start
            }
            current_phase_id += 1

    return {
        'windows': windows,
        'labels': np.array(window_labels, dtype=np.int64),
        'subject_id': subject_id,
        'window_indices': np.array(window_indices, dtype=np.int64),
        'phase_ids': np.array(phase_ids, dtype=np.int64),
        'phase_info': phase_info,
        'n_windows': len(windows),
        'n_phases': len(phase_info)
    }


def main():
    print("=" * 60)
    print("WESAD Raw Data Parsing with Subject IDs + Phase Boundaries")
    print("=" * 60)

    all_subjects_data = {}
    stats = {
        'total_windows': 0,
        'total_phases': 0,
        'per_subject': {},
        'label_distribution': {0: 0, 1: 0, 2: 0},
        'phase_distribution': {0: 0, 1: 0, 2: 0}  # Phases per label
    }

    for subject_id in tqdm(SUBJECT_IDS, desc="Processing subjects"):
        result = extract_windows_for_subject(subject_id)

        if result is None:
            continue

        all_subjects_data[subject_id] = result
        stats['total_windows'] += result['n_windows']
        stats['total_phases'] += result['n_phases']
        stats['per_subject'][subject_id] = {
            'n_windows': result['n_windows'],
            'n_phases': result['n_phases']
        }

        for label in result['labels']:
            stats['label_distribution'][int(label)] += 1

        for phase_id, phase_data in result['phase_info'].items():
            stats['phase_distribution'][phase_data['label']] += 1

    # Print statistics
    print("\n" + "=" * 60)
    print("PARSING COMPLETE")
    print("=" * 60)
    print(f"Total subjects: {len(all_subjects_data)}")
    print(f"Total windows: {stats['total_windows']}")
    print(f"Total phases: {stats['total_phases']}")

    print("\nPer-subject statistics:")
    for sid in sorted(stats['per_subject'].keys()):
        data = stats['per_subject'][sid]
        print(f"  {sid}: {data['n_windows']} windows, {data['n_phases']} phases")

    print("\nLabel distribution (windows):")
    for label, count in stats['label_distribution'].items():
        pct = count / stats['total_windows'] * 100
        print(f"  {label} ({LABEL_NAMES[label]}): {count} ({pct:.1f}%)")

    print("\nPhase distribution (# of contiguous segments):")
    for label, count in stats['phase_distribution'].items():
        print(f"  {label} ({LABEL_NAMES[label]}): {count} phases")

    # Verify modality shapes
    sample_window = all_subjects_data['S2']['windows'][0]
    print("\nModality shapes (per window):")
    for mod, arr in sample_window.items():
        print(f"  {mod}: {arr.shape}")

    # Show sample phase info
    print("\nSample phase structure (S2):")
    for phase_id, phase_data in list(all_subjects_data['S2']['phase_info'].items())[:5]:
        print(f"  Phase {phase_id}: {phase_data['label_name']}, "
              f"windows {phase_data['start_window_idx']}-{phase_data['end_window_idx']} "
              f"({phase_data['n_windows']} windows)")

    # Save to pickle
    output_path = OUTPUT_DIR / "wesad_with_subjects.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump({
            'subjects': all_subjects_data,
            'stats': stats,
            'modality_shapes': {k: v.shape for k, v in sample_window.items()},
            'subject_ids': SUBJECT_IDS,
            'label_names': LABEL_NAMES
        }, f)

    print(f"\nSaved to: {output_path}")

    # Save stats as JSON for easy inspection
    stats_path = OUTPUT_DIR / "parsing_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
