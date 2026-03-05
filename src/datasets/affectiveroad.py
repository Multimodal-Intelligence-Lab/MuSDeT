"""
AffectiveROAD LOSO Dataset Module

Loads preprocessed AffectiveROAD data (12 subjects, 8 modalities, binary classification)
and provides LOSO cross-validation folds.

Data format: Per-subject windows with 8 wrist modalities (Left/Right × BVP/EDA/HR/TEMP).
Each window is 1 second. Sampling rates: BVP=64Hz, EDA=4Hz, HR=1Hz, TEMP=4Hz.

Subjects: Drv1, Drv3-Drv13 (12 total, Drv2 excluded).
Labels: 0=low stress, 1=high stress (threshold >= 0.75 on subjective metric).

Data pipeline:
    parse_raw_affectiveroad.py → affectiveroad_with_subjects.pkl (raw values, no zscore)
    create_affectiveroad_loso_folds.py → loso_folds.pkl (12 LOSO folds, 10/1/1)
    This module → DataLoaders with train-only normalization
"""

import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.datasets.wesad import (
    NormalizationStats,
    LOSOSingleWindowDataset,
    LOSOSequenceDataset,
)


# Modality names matching preprocessed data keys
MODALITY_NAMES = [
    'Left_BVP', 'Left_EDA', 'Left_HR', 'Left_TEMP',
    'Right_BVP', 'Right_EDA', 'Right_HR', 'Right_TEMP'
]

# Sampling rates (Hz) = seq_len per 1-second window
MODALITY_RATES = [64, 4, 1, 4, 64, 4, 1, 4]


class AffectiveROADLOSODataModule:
    """
    LOSO data module for AffectiveROAD dataset.

    Loads LOSO folds from loso_folds.pkl (created by preprocessing scripts).
    Applies train-only normalization at runtime — NO baked-in zscore.

    Interface matches LOSODataModule from wesad.py for unified training.
    """

    def __init__(
        self,
        data_dir: Path,
        context_length: int = 1,
        causal: bool = True,
        batch_size: int = 64,
        num_workers: int = 0,
        **kwargs,
    ):
        self.data_dir = Path(data_dir)
        self.context_length = context_length
        self.causal = causal
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Load LOSO folds
        folds_path = self.data_dir / 'loso_folds.pkl'
        if not folds_path.exists():
            raise FileNotFoundError(
                f"LOSO folds not found at {folds_path}. "
                f"Run preprocessing scripts first:\n"
                f"  1. python -m src.datasets.preprocessing.parse_raw_affectiveroad\n"
                f"  2. python -m src.datasets.preprocessing.create_affectiveroad_loso_folds"
            )

        print(f"Loading AffectiveROAD LOSO folds from {folds_path}...")
        with open(folds_path, 'rb') as f:
            data = pickle.load(f)

        self.folds = data['folds']
        self.subject_ids = data['subject_ids']
        self.n_folds = data['n_folds']
        self._protocol = data['protocol']

        total_windows = sum(
            self.folds[f'fold_{i}']['train']['n_samples'] +
            self.folds[f'fold_{i}']['val']['n_samples'] +
            self.folds[f'fold_{i}']['test']['n_samples']
            for i in range(min(1, self.n_folds))  # Just check fold 0
        )
        print(f"Loaded {self.n_folds} LOSO folds")
        print(f"Protocol: {self._protocol}")

        self.normalizer = None
        self.current_fold = None

    @property
    def protocol(self) -> str:
        return f"LOSO ({self.n_folds} subjects)"

    def get_fold_info(self, fold_idx: int) -> Dict:
        """Get information about a specific fold."""
        fold_key = f'fold_{fold_idx}'
        fold = self.folds[fold_key]
        return {
            'test_subject': fold['test_subject'],
            'val_subject': fold['val_subject'],
            'train_subjects': fold['train_subjects'],
            'n_train': fold['train']['n_samples'],
            'n_val': fold['val']['n_samples'],
            'n_test': fold['test']['n_samples'],
        }

    def get_dataloaders(
        self,
        fold_idx: int,
        use_sequences: bool = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get DataLoaders for a specific LOSO fold."""
        train_ds, val_ds, test_ds = self.get_fold_data(fold_idx, use_sequences)

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )

        return train_loader, val_loader, test_loader

    def get_fold_data(
        self,
        fold_idx: int,
        use_sequences: bool = None
    ) -> Tuple:
        """
        Get train/val/test datasets for a LOSO fold.

        Uses pre-computed LOSO splits from loso_folds.pkl.
        Applies train-only normalization (NormalizationStats).
        """
        if fold_idx >= self.n_folds:
            raise ValueError(f"fold_idx {fold_idx} >= n_folds {self.n_folds}")

        fold_key = f'fold_{fold_idx}'
        fold = self.folds[fold_key]

        train_windows = fold['train']['windows']
        train_labels = fold['train']['labels']
        train_phase_ids = fold['train']['phase_ids']

        val_windows = fold['val']['windows']
        val_labels = fold['val']['labels']
        val_phase_ids = fold['val']['phase_ids']

        test_windows = fold['test']['windows']
        test_labels = fold['test']['labels']
        test_phase_ids = fold['test']['phase_ids']

        # Normalize on train only
        self.normalizer = NormalizationStats()
        self.normalizer.fit(train_windows, source='train_only')
        self.current_fold = fold_idx

        # Decide dataset type
        if use_sequences is None:
            use_sequences = self.context_length > 1

        if use_sequences:
            DatasetCls = LOSOSequenceDataset
            make_ds = lambda w, l, p: DatasetCls(
                windows=w, labels=l, phase_ids=p,
                context_length=self.context_length, causal=self.causal,
                normalizer=self.normalizer, modality_order=MODALITY_NAMES
            )
        else:
            DatasetCls = LOSOSingleWindowDataset
            make_ds = lambda w, l, p: DatasetCls(
                windows=w, labels=l,
                normalizer=self.normalizer, modality_order=MODALITY_NAMES
            )

        return (make_ds(train_windows, train_labels, train_phase_ids),
                make_ds(val_windows, val_labels, val_phase_ids),
                make_ds(test_windows, test_labels, test_phase_ids))

    def get_modality_info(self) -> Dict:
        """Get information about modalities."""
        return {
            'names': MODALITY_NAMES,
            'rates': MODALITY_RATES,
            'n_modalities': len(MODALITY_NAMES)
        }
