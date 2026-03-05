#!/usr/bin/env python3
"""
Parse raw AffectiveROAD E4 wristband data and create unified dataset with subject IDs
and driving-segment phase boundaries.

This script:
1. Loads raw E4 CSV files (Left/Right wrist × BVP/EDA/HR/TEMP)
2. Aligns timestamps across modalities using Annot_E4_{Left,Right}.csv
3. Windows into 1-second segments (matching v1 preprocessing)
4. Binarizes stress labels from subjective metrics (threshold ≥ 0.75)
5. Detects driving-segment phase boundaries (Rest/Zone/City/Hwy)
6. Stores RAW values (NO normalization baked in)
7. Preserves subject IDs and temporal ordering

Driving route phases (from Annot_Subjective_metric.csv columns):
  Z_Start → Z_End → City1 → Hwy → City2 → City2 → Hwy → City1 → Z → Z_End

Output: affectiveroad_with_subjects.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

# Paths
RAW_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "AffectiveROAD" / "Database"
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "data" / "affectiveroad"

# Modalities and their sampling rates
WRIST_MODALITIES = ['BVP', 'EDA', 'HR', 'TEMP']
WRIST_PARTS = ['Left', 'Right']

MODALITY_NAMES = [
    'Left_BVP', 'Left_EDA', 'Left_HR', 'Left_TEMP',
    'Right_BVP', 'Right_EDA', 'Right_HR', 'Right_TEMP'
]

# Expected sampling rates (from E4 CSV headers)
EXPECTED_RATES = {
    'BVP': 64,
    'EDA': 4,
    'HR': 1,
    'TEMP': 4,
}

# Subject IDs (Drv2 excluded per v1)
SUBJECT_IDS = ['Drv1', 'Drv3', 'Drv4', 'Drv5', 'Drv6', 'Drv7',
               'Drv8', 'Drv9', 'Drv10', 'Drv11', 'Drv12', 'Drv13']

# Driving segment names from the Annot CSV columns
# The Annot_Subjective_metric.csv has columns:
# Drive_id, Z_Start, Z_End, City1_Start, City1_End, Hwy_Start, Hwy_End,
# City2_Start, City2_End, City2_Start, City2_End, Hwy_Start, Hwy_End,
# City1_Start, City1_End, Z_Start, Z_End
SEGMENT_NAMES = [
    'Zone_out', 'City1_out', 'Hwy_out', 'City2_out',
    'City2_return', 'Hwy_return', 'City1_return', 'Zone_return'
]


def read_e4_csv(csv_path: str) -> tuple:
    """
    Read an E4 wristband CSV file.

    E4 CSV format:
      Row 0: Unix timestamp of first sample
      Row 1: Sampling rate (Hz)
      Row 2+: Signal values

    Returns: (signal_array, sampling_rate, start_timestamp)
    """
    df = pd.read_csv(csv_path, header=None)
    start_timestamp = int(df.iloc[0, 0])
    fs = int(df.iloc[1, 0])
    signal = df.iloc[2:, 0].astype(float).to_numpy()
    return signal, fs, start_timestamp


def extract_subject_data(subject_id: str) -> dict:
    """
    Extract all 1-second windows for a single subject with driving-segment phases.

    Adapts v1's data_ready() but:
    - NO zscore normalization (store raw values)
    - Detects driving segment boundaries as phases
    - Returns per-window dict format (matching WESAD pipeline)

    Returns dict with:
        'windows': list of dicts with 8 modalities (raw values)
        'labels': array of labels (0=low stress, 1=high stress)
        'subject_id': subject ID string
        'phase_ids': phase ID for each window
        'phase_info': dict mapping phase_id to segment metadata
        'n_windows': total window count
    """
    # Find the data folder for this subject
    e4_dir = RAW_DATA_DIR / "E4"
    data_folders = [f for f in os.listdir(e4_dir)
                    if subject_id in f and os.path.isdir(e4_dir / f)]
    if not data_folders:
        print(f"Warning: No data folder found for {subject_id}")
        return None
    data_folder = data_folders[0]
    data_path = e4_dir / data_folder

    # Paths
    label_meta_path = RAW_DATA_DIR / "Subj_metric" / "Annot_Subjective_metric.csv"
    label_data_path = RAW_DATA_DIR / "Subj_metric" / f"SM_{subject_id}.csv"

    label_meta_df = pd.read_csv(label_meta_path)

    # ---- Read all E4 signals ----
    m_info = {}
    std_time_stamp = {"Left": 0, "Right": 0}

    for part in WRIST_PARTS:
        for m in WRIST_MODALITIES:
            key = f"{part}_{m}"
            csv_path = data_path / part / f"{m}.csv"
            if not csv_path.exists():
                print(f"Warning: {csv_path} not found")
                return None

            signal, fs, start_ts = read_e4_csv(str(csv_path))
            m_info[key] = {
                'signal': signal,
                'fs': fs,
                'start_timestamp': start_ts
            }

            # Use the 4Hz modality's timestamp as reference (same as v1)
            if fs == 4:
                std_time_stamp[part] = start_ts

    # ---- Align signals using E4 annotations (same as v1 lines 60-75) ----
    for part in WRIST_PARTS:
        annot_path = RAW_DATA_DIR / "E4" / f"Annot_E4_{part}.csv"
        m_timeline_df = pd.read_csv(annot_path)
        m_timeline = m_timeline_df.loc[m_timeline_df["Drive-id"] == subject_id].iloc[0]

        for m in WRIST_MODALITIES:
            key = f"{part}_{m}"
            fs = m_info[key]['fs']
            signal = m_info[key]['signal']

            # Trim signal to the annotated driving region
            # v1 uses iloc[3] (Z_Start) and iloc[-3] (Z_End of return)
            z_start_idx = int(m_timeline.iloc[3] * fs / 4)
            z_end_idx = int(m_timeline.iloc[-3] * fs / 4)

            m_info[key]['signal'] = signal[z_start_idx:z_end_idx]

    # ---- Window into 1-second segments ----
    window_size_sec = 1
    min_length = int(1e8)

    modality_windows = {}
    for part in WRIST_PARTS:
        for m in WRIST_MODALITIES:
            key = f"{part}_{m}"
            fs = m_info[key]['fs']
            signal = m_info[key]['signal']
            windows_list = []

            for j in range(0, signal.shape[0] - window_size_sec * fs + 1, fs):
                window = signal[j:j + window_size_sec * fs]
                windows_list.append(window)

            modality_windows[key] = windows_list
            if len(windows_list) < min_length:
                min_length = len(windows_list)

    # ---- Process labels (same as v1 lines 96-123) ----
    label_df = pd.read_csv(label_data_path, header=None)
    label_raw = label_df.iloc[1:, 0].astype(float).to_numpy()

    label_meta = label_meta_df.loc[label_meta_df["Drive_id"] == subject_id].iloc[0]
    start_idx = int(label_meta.iloc[1])
    end_idx = int(label_meta.iloc[-1])

    label_trimmed = label_raw[start_idx:end_idx]
    label_cls = np.zeros_like(label_trimmed, dtype=int)
    label_cls[label_trimmed >= 0.75] = 1

    # Resample labels to match window rate (same as v1 lines 111-123)
    labels = []
    sampling_rate = label_cls.shape[0] / min_length
    t = 1
    idx = 0
    while idx < label_trimmed.shape[0]:
        sample_idx = int(sampling_rate * t)
        if sample_idx >= len(label_cls):
            break
        labels.append(label_cls[sample_idx])
        t += 1
        idx = sampling_rate * t

    labels = labels[window_size_sec - 1:]
    labels = labels[:min_length]

    # Truncate all modalities to same length
    n_windows = len(labels)
    for key in modality_windows:
        modality_windows[key] = modality_windows[key][:n_windows]

    # ---- Detect driving segment phases ----
    # The label_meta columns define segment boundaries in label-space indices
    # Columns: Drive_id, Z_Start, Z_End, City1_Start, City1_End, ...
    # We need to map these to window indices
    segment_boundaries = []
    meta_values = label_meta.values[1:]  # Skip Drive_id column
    for seg_idx in range(0, len(meta_values) - 1, 2):
        seg_start_label = int(meta_values[seg_idx])
        seg_end_label = int(meta_values[seg_idx + 1])

        # Convert from label indices to window indices
        # label indices are relative to the trimmed label array
        if sampling_rate > 0:
            seg_start_win = max(0, int((seg_start_label - start_idx) / sampling_rate))
            seg_end_win = min(n_windows, int((seg_end_label - start_idx) / sampling_rate))
        else:
            seg_start_win = 0
            seg_end_win = n_windows

        if seg_start_win < seg_end_win:
            seg_name = SEGMENT_NAMES[seg_idx // 2] if seg_idx // 2 < len(SEGMENT_NAMES) else f"segment_{seg_idx // 2}"
            segment_boundaries.append((seg_start_win, seg_end_win, seg_name))

    # Assign phase IDs to each window
    phase_ids = np.full(n_windows, -1, dtype=np.int64)
    phase_info = {}

    for phase_id, (seg_start, seg_end, seg_name) in enumerate(segment_boundaries):
        for w in range(seg_start, min(seg_end, n_windows)):
            phase_ids[w] = phase_id

        n_seg_windows = min(seg_end, n_windows) - seg_start
        if n_seg_windows > 0:
            seg_labels = labels[seg_start:min(seg_end, n_windows)]
            phase_info[phase_id] = {
                'segment_name': seg_name,
                'start_window_idx': seg_start,
                'end_window_idx': min(seg_end, n_windows),
                'n_windows': n_seg_windows,
                'stress_ratio': float(np.mean(seg_labels)) if len(seg_labels) > 0 else 0.0,
            }

    # Windows not assigned to any segment get their own phase
    unassigned = np.where(phase_ids == -1)[0]
    if len(unassigned) > 0:
        next_phase_id = max(phase_info.keys()) + 1 if phase_info else 0
        # Group contiguous unassigned windows
        splits = np.split(unassigned, np.where(np.diff(unassigned) != 1)[0] + 1)
        for group in splits:
            if len(group) > 0:
                for w in group:
                    phase_ids[w] = next_phase_id
                phase_info[next_phase_id] = {
                    'segment_name': 'transition',
                    'start_window_idx': int(group[0]),
                    'end_window_idx': int(group[-1]) + 1,
                    'n_windows': len(group),
                }
                next_phase_id += 1

    # ---- Build per-window dicts (raw values, no zscore) ----
    windows = []
    for i in range(n_windows):
        window = {}
        for key in MODALITY_NAMES:
            # Store as flat 1D array (seq_len,) — same as WESAD pipeline
            window[key] = modality_windows[key][i].flatten().astype(np.float32)
        windows.append(window)

    return {
        'windows': windows,
        'labels': np.array(labels, dtype=np.int64),
        'subject_id': subject_id,
        'phase_ids': phase_ids,
        'phase_info': phase_info,
        'n_windows': n_windows,
        'n_phases': len(phase_info),
    }


def main():
    print("=" * 60)
    print("AffectiveROAD Raw Data Parsing")
    print("  - NO normalization (raw values)")
    print("  - Driving-segment phase boundaries preserved")
    print("  - 12 subjects (Drv2 excluded)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_subjects_data = {}
    stats = {
        'total_windows': 0,
        'total_phases': 0,
        'per_subject': {},
        'label_distribution': {0: 0, 1: 0},
    }

    for subject_id in tqdm(SUBJECT_IDS, desc="Processing subjects"):
        result = extract_subject_data(subject_id)

        if result is None:
            print(f"  Skipping {subject_id}")
            continue

        all_subjects_data[subject_id] = result
        stats['total_windows'] += result['n_windows']
        stats['total_phases'] += result['n_phases']
        stats['per_subject'][subject_id] = {
            'n_windows': result['n_windows'],
            'n_phases': result['n_phases'],
        }

        for label in result['labels']:
            stats['label_distribution'][int(label)] += 1

    # Print statistics
    print("\n" + "=" * 60)
    print("PARSING COMPLETE")
    print("=" * 60)
    print(f"Total subjects: {len(all_subjects_data)}")
    print(f"Total windows: {stats['total_windows']}")
    print(f"Total phases: {stats['total_phases']}")

    print("\nPer-subject statistics:")
    for sid in sorted(stats['per_subject'].keys(),
                      key=lambda s: int(s.replace('Drv', ''))):
        data = stats['per_subject'][sid]
        print(f"  {sid}: {data['n_windows']} windows, {data['n_phases']} phases")

    print("\nLabel distribution (windows):")
    label_names = {0: 'low_stress', 1: 'high_stress'}
    for label, count in stats['label_distribution'].items():
        pct = count / stats['total_windows'] * 100 if stats['total_windows'] > 0 else 0
        print(f"  {label} ({label_names[label]}): {count} ({pct:.1f}%)")

    # Verify modality shapes
    first_subj = list(all_subjects_data.values())[0]
    sample_window = first_subj['windows'][0]
    print("\nModality shapes (per window):")
    for mod, arr in sample_window.items():
        print(f"  {mod}: {arr.shape}")

    # Verify NO normalization (values should NOT be ~N(0,1))
    print("\nNormalization check (should NOT be ~0 mean / ~1 std):")
    for mod_name in MODALITY_NAMES[:4]:  # Just check Left modalities
        all_vals = []
        for w in first_subj['windows'][:100]:
            all_vals.append(w[mod_name])
        all_vals = np.concatenate(all_vals)
        print(f"  {mod_name}: mean={all_vals.mean():.4f}, std={all_vals.std():.4f}")

    # Save
    output_path = OUTPUT_DIR / "affectiveroad_with_subjects.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump({
            'subjects': all_subjects_data,
            'stats': stats,
            'modality_shapes': {k: v.shape for k, v in sample_window.items()},
            'subject_ids': SUBJECT_IDS,
            'label_names': label_names,
        }, f)

    print(f"\nSaved to: {output_path}")

    # Save stats JSON
    stats_path = OUTPUT_DIR / "parsing_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
