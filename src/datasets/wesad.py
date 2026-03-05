"""
LOSO Sequence Dataset for Novel Hierarchical Co-Info Model

Key features:
1. Proper LOSO with real subject boundaries (from step7 infrastructure)
2. Phase-safe sequence sampling (sequences never cross phase boundaries)
3. Train-only normalization (prevents leakage)
4. Returns sequences of (context_len, channels, time) for temporal context

This module replaces the heuristic LOSO in wesad.py with proper subject boundaries.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
import json


# Modality order matching wesad_with_subjects.pkl
MODALITY_NAMES = [
    'GSR_chest', 'BVP_wrist', 'EMG_chest',
    'ECG_chest', 'RESP_chest', 'GSR_wrist'
]

# Modality sampling rates (Hz) - determines seq_len
MODALITY_RATES = [700, 64, 700, 700, 700, 4]


class NormalizationStats:
    """
    Compute and apply normalization stats from TRAIN data only.

    CRITICAL: This prevents subtle leakage in LOSO evaluation.
    Stats are computed per-modality.
    """

    def __init__(self):
        self.mean = {}
        self.std = {}
        self.fitted = False
        self.fit_data_source = None

    def fit(self, windows: List[Dict[str, np.ndarray]], source: str = 'train_only'):
        """Compute mean/std from training windows ONLY."""
        # Collect all values per modality
        modality_values = defaultdict(list)

        for window in windows:
            for mod_name, arr in window.items():
                modality_values[mod_name].append(arr.flatten())

        # Compute stats
        for mod_name, values in modality_values.items():
            all_values = np.concatenate(values)
            self.mean[mod_name] = float(np.mean(all_values))
            self.std[mod_name] = float(np.std(all_values))
            # Avoid division by zero
            if self.std[mod_name] < 1e-8:
                self.std[mod_name] = 1.0

        self.fitted = True
        self.fit_data_source = source

    def transform(self, window: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply normalization to a single window."""
        if not self.fitted:
            raise RuntimeError("NormalizationStats not fitted. Call fit() first.")

        normalized = {}
        for mod_name, arr in window.items():
            normalized[mod_name] = (arr - self.mean[mod_name]) / self.std[mod_name]
        return normalized

    def transform_tensor(self, modality_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply normalization to tensors."""
        if not self.fitted:
            raise RuntimeError("NormalizationStats not fitted. Call fit() first.")

        normalized = {}
        for mod_name, tensor in modality_tensors.items():
            mean = self.mean[mod_name]
            std = self.std[mod_name]
            normalized[mod_name] = (tensor - mean) / std
        return normalized


class LOSOSequenceDataset(Dataset):
    """
    Dataset that returns SEQUENCES of windows for temporal context.

    Key features:
    - Phase-safe: sequences never cross phase boundaries
    - Returns tuple of (context_len, 1, seq_len) tensors per modality
    - Causal mode: label is last window; bidirectional: center window
    - Supports train-only normalization
    """

    def __init__(
        self,
        windows: List[Dict[str, np.ndarray]],
        labels: np.ndarray,
        phase_ids: List[int],
        context_length: int,
        causal: bool = True,
        normalizer: Optional[NormalizationStats] = None,
        modality_order: List[str] = None
    ):
        self.windows = windows
        self.labels = labels
        self.phase_ids = phase_ids
        self.context_length = context_length
        self.causal = causal
        self.normalizer = normalizer
        self.modality_order = modality_order or MODALITY_NAMES

        # Group windows by phase
        self.phase_to_indices = defaultdict(list)
        for idx, phase_id in enumerate(phase_ids):
            self.phase_to_indices[phase_id].append(idx)

        # Build valid sequence start indices (within phase only)
        self.valid_sequences = []
        for phase_id, indices in self.phase_to_indices.items():
            n_windows = len(indices)
            if n_windows >= context_length:
                for start in range(n_windows - context_length + 1):
                    # Store (phase_id, actual window indices in sequence)
                    seq_indices = indices[start:start + context_length]
                    self.valid_sequences.append((phase_id, seq_indices))

        # Verify we have some valid sequences
        if len(self.valid_sequences) == 0:
            raise ValueError(
                f"No valid sequences with context_length={context_length}. "
                f"Max phase lengths: {[len(v) for v in self.phase_to_indices.values()]}"
            )

    def __len__(self) -> int:
        return len(self.valid_sequences)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, int]:
        """
        Returns:
            modalities: Tuple of (context_len, 1, seq_len) tensors
            label: scalar tensor (label of target window)
            target_idx: index of target window in original dataset
        """
        phase_id, window_indices = self.valid_sequences[idx]

        # Collect windows in sequence
        seq_modalities = {name: [] for name in self.modality_order}

        for win_idx in window_indices:
            window = self.windows[win_idx]

            # Apply normalization if available
            if self.normalizer is not None:
                window = self.normalizer.transform(window)

            for name in self.modality_order:
                arr = window[name]
                # Add channel dimension: (seq_len,) -> (1, seq_len)
                tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
                seq_modalities[name].append(tensor)

        # Stack to (context_len, 1, seq_len) per modality
        modalities = tuple(
            torch.stack(seq_modalities[name], dim=0)
            for name in self.modality_order
        )

        # Target label: last window for causal, center for bidirectional
        if self.causal:
            target_idx = window_indices[-1]
        else:
            target_idx = window_indices[self.context_length // 2]

        label = torch.tensor(self.labels[target_idx], dtype=torch.long)

        return modalities, label, target_idx


class LOSOSingleWindowDataset(Dataset):
    """
    Dataset for single-window classification (no temporal context).

    For Stage A/B where we don't need sequences.
    """

    def __init__(
        self,
        windows: List[Dict[str, np.ndarray]],
        labels: np.ndarray,
        normalizer: Optional[NormalizationStats] = None,
        modality_order: List[str] = None
    ):
        self.windows = windows
        self.labels = labels
        self.normalizer = normalizer
        self.modality_order = modality_order or MODALITY_NAMES

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, int]:
        """
        Returns:
            modalities: Tuple of (1, seq_len) tensors
            label: scalar tensor
            idx: window index
        """
        window = self.windows[idx]

        # Apply normalization if available
        if self.normalizer is not None:
            window = self.normalizer.transform(window)

        # Create tensors with channel dimension
        modalities = tuple(
            torch.tensor(window[name], dtype=torch.float32).unsqueeze(0)
            for name in self.modality_order
        )

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return modalities, label, idx


class LOSODataModule:
    """
    Data module for LOSO (Leave-One-Subject-Out) cross-validation.

    Uses pre-computed fold definitions from step7 infrastructure.
    Supports both single-window and sequence modes.

    NOTE: data_path is kept for API compatibility but is UNUSED.
    The loso_folds.pkl already contains all windows/labels/phase_ids.
    """

    def __init__(
        self,
        data_path: Path,
        loso_folds_path: Path,
        context_length: int = 1,
        causal: bool = True,
        batch_size: int = 64,
        num_workers: int = 0
    ):
        # NOTE: data_path is unused - folds contain windows already
        self.data_path = Path(data_path) if data_path else None
        self.loso_folds_path = Path(loso_folds_path)
        self.context_length = context_length
        self.causal = causal
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Load data
        print(f"Loading LOSO folds from {self.loso_folds_path}...")
        with open(self.loso_folds_path, 'rb') as f:
            loso_data = pickle.load(f)

        self.folds = loso_data['folds']
        self.subject_ids = loso_data['subject_ids']
        self.n_folds = loso_data['n_folds']
        self.protocol = loso_data['protocol']

        print(f"Loaded {self.n_folds} LOSO folds")
        print(f"Protocol: {self.protocol}")

        # Pre-check phase lengths if using sequences
        if context_length > 1:
            self._check_phase_lengths()

        # Normalizer will be computed per-fold
        self.normalizer = None
        self.current_fold = None

    def _check_phase_lengths(self):
        """
        Check if context_length is compatible with phase lengths in all folds.

        Warns if any split has phases too short for the context_length,
        which would result in empty datasets.
        """
        issues = []

        for fold_idx in range(self.n_folds):
            fold_key = f'fold_{fold_idx}'
            fold_data = self.folds[fold_key]

            for split_name in ['train', 'val', 'test']:
                phase_ids = fold_data[split_name]['phase_ids']

                # Count windows per phase
                phase_counts = defaultdict(int)
                for pid in phase_ids:
                    phase_counts[pid] += 1

                max_phase_len = max(phase_counts.values()) if phase_counts else 0
                min_phase_len = min(phase_counts.values()) if phase_counts else 0

                if max_phase_len < self.context_length:
                    issues.append(
                        f"Fold {fold_idx} {split_name}: max phase length ({max_phase_len}) "
                        f"< context_length ({self.context_length})"
                    )

        if issues:
            print(f"\n⚠️  WARNING: context_length={self.context_length} may cause issues:")
            for issue in issues[:5]:  # Show first 5
                print(f"   - {issue}")
            if len(issues) > 5:
                print(f"   ... and {len(issues) - 5} more")
            print("   Consider using a smaller context_length or single-window mode.\n")

    def get_fold_info(self, fold_idx: int) -> Dict:
        """Get information about a specific fold."""
        fold_key = f'fold_{fold_idx}'
        return self.folds[fold_key]

    def get_fold_data(
        self,
        fold_idx: int,
        use_sequences: bool = None
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Get train/val/test datasets for a specific fold.

        Args:
            fold_idx: Fold index (0 to n_folds-1)
            use_sequences: If True, return sequence datasets; if False, single-window.
                          If None, use context_length > 1 to decide.

        Returns:
            train_dataset, val_dataset, test_dataset
        """
        fold_key = f'fold_{fold_idx}'
        fold_data = self.folds[fold_key]

        # Extract splits
        train_windows = fold_data['train']['windows']
        train_labels = fold_data['train']['labels']
        train_phase_ids = fold_data['train']['phase_ids']

        val_windows = fold_data['val']['windows']
        val_labels = fold_data['val']['labels']
        val_phase_ids = fold_data['val']['phase_ids']

        test_windows = fold_data['test']['windows']
        test_labels = fold_data['test']['labels']
        test_phase_ids = fold_data['test']['phase_ids']

        # Compute normalization on TRAIN ONLY
        self.normalizer = NormalizationStats()
        self.normalizer.fit(train_windows, source='train_only')
        self.current_fold = fold_idx

        # Decide dataset type
        if use_sequences is None:
            use_sequences = self.context_length > 1

        if use_sequences:
            # Sequence datasets for temporal context
            train_dataset = LOSOSequenceDataset(
                windows=train_windows,
                labels=train_labels,
                phase_ids=self._phase_ids_to_ints(train_phase_ids),
                context_length=self.context_length,
                causal=self.causal,
                normalizer=self.normalizer
            )
            val_dataset = LOSOSequenceDataset(
                windows=val_windows,
                labels=val_labels,
                phase_ids=self._phase_ids_to_ints(val_phase_ids),
                context_length=self.context_length,
                causal=self.causal,
                normalizer=self.normalizer
            )
            test_dataset = LOSOSequenceDataset(
                windows=test_windows,
                labels=test_labels,
                phase_ids=self._phase_ids_to_ints(test_phase_ids),
                context_length=self.context_length,
                causal=self.causal,
                normalizer=self.normalizer
            )
        else:
            # Single-window datasets
            train_dataset = LOSOSingleWindowDataset(
                windows=train_windows,
                labels=train_labels,
                normalizer=self.normalizer
            )
            val_dataset = LOSOSingleWindowDataset(
                windows=val_windows,
                labels=val_labels,
                normalizer=self.normalizer
            )
            test_dataset = LOSOSingleWindowDataset(
                windows=test_windows,
                labels=test_labels,
                normalizer=self.normalizer
            )

        return train_dataset, val_dataset, test_dataset

    def _phase_ids_to_ints(self, phase_ids: List[str]) -> List[int]:
        """Convert string phase IDs to integers for grouping."""
        unique_phases = sorted(set(phase_ids))
        phase_to_int = {p: i for i, p in enumerate(unique_phases)}
        return [phase_to_int[p] for p in phase_ids]

    def get_dataloaders(
        self,
        fold_idx: int,
        use_sequences: bool = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get DataLoaders for a specific fold."""
        train_ds, val_ds, test_ds = self.get_fold_data(fold_idx, use_sequences)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader

    def get_modality_info(self) -> Dict:
        """Get information about modalities."""
        return {
            'names': MODALITY_NAMES,
            'rates': MODALITY_RATES,
            'n_modalities': len(MODALITY_NAMES)
        }

    def get_label_info(self, fold_idx: int) -> Dict:
        """Get label distribution for a fold."""
        fold_data = self.folds[f'fold_{fold_idx}']

        train_labels = fold_data['train']['labels']
        val_labels = fold_data['val']['labels']
        test_labels = fold_data['test']['labels']

        def compute_dist(labels):
            unique, counts = np.unique(labels, return_counts=True)
            return {int(u): int(c) for u, c in zip(unique, counts)}

        return {
            'train': compute_dist(train_labels),
            'val': compute_dist(val_labels),
            'test': compute_dist(test_labels)
        }


def compute_marginal_entropy_bits(labels: np.ndarray) -> float:
    """
    Compute marginal entropy H(Y) = -Σ p(y) log₂ p(y) in BITS.

    This is the entropy of the label distribution, NOT the conditional
    entropy H(Y|X) which is approximated by NLL.
    """
    unique, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    # Use log2 for bits
    return -np.sum(probs * np.log2(probs + 1e-10))


def nll_to_bits(nll_nats: float) -> float:
    """Convert NLL from nats (PyTorch default) to bits."""
    LN2 = np.log(2)
    return nll_nats / LN2


def compute_nll_mean_bits(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    """
    Compute mean NLL in BITS per sample (window/sequence).

    This is the proper entropy proxy: H(Y|X) ≈ E[-log₂ p(y|x)]

    CRITICAL: This returns MEAN bits per sample, computed over the entire
    evaluation set. All NLLs used for U_i/C_ij should use this function
    to ensure consistent units.

    Args:
        model: Trained model (will be set to eval mode)
        dataloader: DataLoader for evaluation set
        device: torch device

    Returns:
        Mean NLL in bits per sample
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')  # Sum for proper averaging

    total_nll_nats = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            modalities, labels, _ = batch
            modalities = tuple(m.to(device) for m in modalities)
            labels = labels.to(device)

            logits = model(modalities)
            nll_nats = criterion(logits, labels)

            total_nll_nats += nll_nats.item()
            total_samples += len(labels)

    # Mean NLL in nats
    mean_nll_nats = total_nll_nats / total_samples

    # Convert to bits
    LN2 = np.log(2)
    mean_nll_bits = mean_nll_nats / LN2

    return mean_nll_bits


# Convenience function to check normalization
def verify_train_only_normalization(data_module: LOSODataModule) -> bool:
    """Verify that normalization stats are computed from TRAIN only."""
    if data_module.normalizer is None:
        raise RuntimeError("Normalizer not initialized. Call get_fold_data first.")

    assert data_module.normalizer.fitted, "Normalizer not fitted"
    assert data_module.normalizer.fit_data_source == 'train_only', \
        f"Normalizer fit on '{data_module.normalizer.fit_data_source}', expected 'train_only'"

    print("✓ Normalization computed on train data only")
    return True


class GroupedPhaseDataModule:
    """
    Data module for phase-disjoint cross-validation using step7 metadata.

    This is a PROPER grouped protocol using real (subject, phase) metadata,
    NOT the label-run approximation in WESADDataModule.grouped_by_trial.

    Protocol:
    ---------
    - Load all subjects from wesad_with_subjects.pkl (has phase_ids metadata)
    - Create global phase IDs: {subject_id}_phase_{phase_id}
    - Do stratified group K-fold where groups are phases
    - NO phase appears in both train and test (phase-disjoint)
    - Same subject CAN appear in both train and test (different phases)

    Caveat:
    -------
    Subject leakage remains: the same subject can have windows in both
    train and test (from different phases). For subject-disjoint evaluation,
    use LOSODataModule instead.

    This protocol is useful for:
    - Intermediate validation between window-shuffle (too optimistic) and LOSO (too pessimistic)
    - Testing whether phase structure matters for generalization
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        n_folds: int = 10,
        val_ratio: float = 0.1,
        seed: int = 42,
        context_length: int = 1,
        causal: bool = True,
        batch_size: int = 64,
        num_workers: int = 0
    ):
        """
        Args:
            data_path: Path to wesad_with_subjects.pkl
            n_folds: Number of CV folds
            val_ratio: Fraction of train phases to use for validation
            seed: Random seed for reproducibility
            context_length: Number of consecutive windows per sequence
            causal: If True, predict last window; if False, predict center
            batch_size: Batch size for dataloaders
            num_workers: Number of dataloader workers
        """
        self.data_path = Path(data_path)
        self.n_folds = n_folds
        self.val_ratio = val_ratio
        self.seed = seed
        self.context_length = context_length
        self.causal = causal
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Load data
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.subject_ids = self.data['subject_ids']
        self.subjects = self.data['subjects']

        # Build global arrays
        self._build_global_arrays()

        # Create folds
        self._create_folds()

        # Normalizer (fit per fold)
        self.normalizer = None
        self.current_fold = None

    def _build_global_arrays(self):
        """Flatten all subjects into global arrays with phase metadata."""
        all_windows = []
        all_labels = []
        all_phase_ids = []  # Global phase ID: "{subject}_phase_{phase}"

        for subj_id in self.subject_ids:
            subj_data = self.subjects[subj_id]
            windows = subj_data['windows']
            labels = subj_data['labels']
            local_phase_ids = subj_data['phase_ids']

            for i, (window, label, local_phase) in enumerate(zip(windows, labels, local_phase_ids)):
                all_windows.append(window)
                all_labels.append(label)
                # Global phase ID encodes both subject and phase
                global_phase_id = f"{subj_id}_phase_{local_phase}"
                all_phase_ids.append(global_phase_id)

        self.all_windows = all_windows
        self.all_labels = np.array(all_labels)
        self.all_phase_ids = np.array(all_phase_ids)

        # Get unique phases
        self.unique_phases = np.unique(self.all_phase_ids)
        self.n_phases = len(self.unique_phases)

        print(f"GroupedPhaseDataModule: {len(self.all_windows)} windows, "
              f"{self.n_phases} phases, {len(self.subject_ids)} subjects")

    def _create_folds(self):
        """Create K-fold splits where entire phases stay together."""
        from sklearn.model_selection import StratifiedGroupKFold

        rng = np.random.RandomState(self.seed)

        # Get phase-level labels (majority label per phase) for stratification
        phase_labels = {}
        for phase_id in self.unique_phases:
            mask = self.all_phase_ids == phase_id
            labels_in_phase = self.all_labels[mask]
            # Majority label
            unique, counts = np.unique(labels_in_phase, return_counts=True)
            phase_labels[phase_id] = unique[np.argmax(counts)]

        # Create arrays for sklearn
        phase_indices = np.arange(self.n_phases)
        phase_label_array = np.array([phase_labels[p] for p in self.unique_phases])

        # Stratified group K-fold
        sgkf = StratifiedGroupKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        # For StratifiedGroupKFold, we need groups to be the phase indices
        # But we want to stratify by labels and split by phases
        # So we treat each phase as one sample for splitting purposes
        self.folds = []

        for train_phase_idx, test_phase_idx in sgkf.split(
            phase_indices, phase_label_array, phase_indices
        ):
            train_phases = self.unique_phases[train_phase_idx]
            test_phases = self.unique_phases[test_phase_idx]

            # Split train into train+val
            n_val = max(1, int(len(train_phases) * self.val_ratio))
            rng.shuffle(train_phases)  # In-place shuffle for val split
            val_phases = train_phases[:n_val]
            final_train_phases = train_phases[n_val:]

            # Convert phases to window indices
            train_indices = np.where(np.isin(self.all_phase_ids, final_train_phases))[0]
            val_indices = np.where(np.isin(self.all_phase_ids, val_phases))[0]
            test_indices = np.where(np.isin(self.all_phase_ids, test_phases))[0]

            self.folds.append({
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
                'train_phases': list(final_train_phases),
                'val_phases': list(val_phases),
                'test_phases': list(test_phases)
            })

    @property
    def protocol(self) -> str:
        return f"grouped_by_phase ({self.n_folds}-fold, {self.n_phases} phases)"

    def get_fold_data(
        self,
        fold_idx: int,
        use_sequences: bool = None
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Get datasets for a specific fold."""
        if fold_idx >= self.n_folds:
            raise ValueError(f"fold_idx {fold_idx} >= n_folds {self.n_folds}")

        fold = self.folds[fold_idx]
        train_idx = fold['train_indices']
        val_idx = fold['val_indices']
        test_idx = fold['test_indices']

        # Extract data
        train_windows = [self.all_windows[i] for i in train_idx]
        val_windows = [self.all_windows[i] for i in val_idx]
        test_windows = [self.all_windows[i] for i in test_idx]

        train_labels = self.all_labels[train_idx]
        val_labels = self.all_labels[val_idx]
        test_labels = self.all_labels[test_idx]

        train_phase_ids = self.all_phase_ids[train_idx]
        val_phase_ids = self.all_phase_ids[val_idx]
        test_phase_ids = self.all_phase_ids[test_idx]

        # Fit normalizer on train only
        self.normalizer = NormalizationStats()
        self.normalizer.fit(train_windows, source='train_only')
        self.current_fold = fold_idx

        # Determine if using sequences
        if use_sequences is None:
            use_sequences = self.context_length > 1

        if use_sequences:
            # Build sequence datasets
            train_dataset = LOSOSequenceDataset(
                windows=train_windows,
                labels=train_labels,
                phase_ids=self._phase_ids_to_ints(train_phase_ids),
                context_length=self.context_length,
                causal=self.causal,
                normalizer=self.normalizer
            )
            val_dataset = LOSOSequenceDataset(
                windows=val_windows,
                labels=val_labels,
                phase_ids=self._phase_ids_to_ints(val_phase_ids),
                context_length=self.context_length,
                causal=self.causal,
                normalizer=self.normalizer
            )
            test_dataset = LOSOSequenceDataset(
                windows=test_windows,
                labels=test_labels,
                phase_ids=self._phase_ids_to_ints(test_phase_ids),
                context_length=self.context_length,
                causal=self.causal,
                normalizer=self.normalizer
            )
        else:
            # Single-window datasets
            train_dataset = LOSOSingleWindowDataset(
                windows=train_windows,
                labels=train_labels,
                normalizer=self.normalizer
            )
            val_dataset = LOSOSingleWindowDataset(
                windows=val_windows,
                labels=val_labels,
                normalizer=self.normalizer
            )
            test_dataset = LOSOSingleWindowDataset(
                windows=test_windows,
                labels=test_labels,
                normalizer=self.normalizer
            )

        return train_dataset, val_dataset, test_dataset

    def _phase_ids_to_ints(self, phase_ids: np.ndarray) -> List[int]:
        """Convert string phase IDs to integers for grouping."""
        unique_phases = sorted(set(phase_ids))
        phase_to_int = {p: i for i, p in enumerate(unique_phases)}
        return [phase_to_int[p] for p in phase_ids]

    def get_dataloaders(
        self,
        fold_idx: int,
        use_sequences: bool = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get DataLoaders for a specific fold."""
        train_ds, val_ds, test_ds = self.get_fold_data(fold_idx, use_sequences)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader

    def get_modality_info(self) -> Dict:
        """Get information about modalities."""
        return {
            'names': MODALITY_NAMES,
            'rates': MODALITY_RATES,
            'n_modalities': len(MODALITY_NAMES)
        }

    def get_fold_info(self, fold_idx: int) -> Dict:
        """Get information about a specific fold."""
        fold = self.folds[fold_idx]
        return {
            'n_train': len(fold['train_indices']),
            'n_val': len(fold['val_indices']),
            'n_test': len(fold['test_indices']),
            'n_train_phases': len(fold['train_phases']),
            'n_val_phases': len(fold['val_phases']),
            'n_test_phases': len(fold['test_phases']),
            'test_phases': fold['test_phases']
        }


if __name__ == "__main__":
    # Test the dataset
    import sys

    data_dir = Path(__file__).parent.parent.parent / "data"

    print("Testing LOSODataModule...")

    dm = LOSODataModule(
        data_path=data_dir / "wesad_with_subjects.pkl",
        loso_folds_path=data_dir / "loso_folds.pkl",
        context_length=10,
        causal=True,
        batch_size=32
    )

    # Test fold 0
    train_ds, val_ds, test_ds = dm.get_fold_data(0)

    print(f"\nFold 0:")
    print(f"  Train: {len(train_ds)} sequences")
    print(f"  Val: {len(val_ds)} sequences")
    print(f"  Test: {len(test_ds)} sequences")

    # Test a batch
    train_loader, val_loader, test_loader = dm.get_dataloaders(0)
    batch = next(iter(train_loader))
    modalities, labels, indices = batch

    print(f"\nBatch shapes:")
    for i, m in enumerate(modalities):
        print(f"  Modality {i} ({MODALITY_NAMES[i]}): {m.shape}")
    print(f"  Labels: {labels.shape}")

    # Verify normalization
    verify_train_only_normalization(dm)

    # Compute H(Y) for fold 0
    fold_info = dm.get_fold_info(0)
    all_labels = np.concatenate([
        fold_info['train']['labels'],
        fold_info['val']['labels'],
        fold_info['test']['labels']
    ])
    H_Y = compute_marginal_entropy_bits(all_labels)
    print(f"\nFold 0 H(Y) = {H_Y:.4f} bits")
    print(f"Balanced 3-class would be {np.log2(3):.4f} bits")

    print("\n✓ All tests passed!")
