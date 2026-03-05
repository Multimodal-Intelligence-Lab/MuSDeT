#!/usr/bin/env python3
"""
Unified Training Script for Multimodal Stress Detection

Supports:
- 5 models: CoInfo-GRU, Husformer, H2, PHemoNet, HyperFuseNet
- 2 datasets: WESAD (3-class, 15 subjects), AffectiveROAD (2-class, 12 subjects)
- LOSO evaluation for all model×dataset combinations
- XAI prior regularization for CoInfo-GRU
- Temperature scaling calibration per fold

Usage:
    python -m src.train --config configs/wesad/coinfo_gru.yaml
    python -m src.train --config configs/wesad/husformer.yaml --fold 0
"""

import argparse
import copy
import json
import random
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import build_model, MODELS
from src.evaluation.metrics import (
    evaluate_predictions, find_optimal_temperature, compute_nll_bits
)
from src.evaluation.loso import aggregate_fold_results, print_summary


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict:
    """Load YAML config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def is_novel_model(model_name: str) -> bool:
    """Check if model uses novel (tuple) interface vs baseline (dict) interface."""
    return model_name in ('coinfo_gru', 'window_only')


def build_data_module(config: Dict):
    """
    Build data module based on config.

    Returns a data module that provides get_dataloaders(fold_idx) returning
    (train_loader, val_loader, test_loader).
    """
    from src.datasets.wesad import LOSODataModule

    dataset = config['dataset']['name']
    data_dir = Path(config['dataset']['data_dir'])

    if dataset == 'wesad':
        loso_folds_path = data_dir / 'loso_folds.pkl'
        if not loso_folds_path.exists():
            raise FileNotFoundError(f"LOSO folds not found at {loso_folds_path}")

        dm = LOSODataModule(
            data_path=data_dir / 'wesad.pkl',  # compatibility
            loso_folds_path=loso_folds_path,
            context_length=config['dataset'].get('context_length', 1),
            causal=config['dataset'].get('causal', True),
            batch_size=config['training']['batch_size'],
            num_workers=config['training'].get('num_workers', 0),
        )
        return dm

    elif dataset == 'affectiveroad':
        from src.datasets.affectiveroad import AffectiveROADLOSODataModule

        dm = AffectiveROADLOSODataModule(
            data_dir=data_dir,
            context_length=config['dataset'].get('context_length', 1),
            causal=config['dataset'].get('causal', True),
            batch_size=config['training']['batch_size'],
            num_workers=config['training'].get('num_workers', 0),
        )
        return dm

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def compute_class_weights(labels, n_classes: int, device: torch.device) -> torch.Tensor:
    """Compute balanced class weights: w_c = N_total / (N_classes * N_c)."""
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    labels = np.asarray(labels).flatten().astype(int)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    weights = len(labels) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_criterion(config: Dict, n_modalities: int, device: torch.device,
                    class_weights: torch.Tensor = None):
    """Build loss function from config."""
    loss_type = config['training'].get('loss_type', 'ce')

    if loss_type == 'ce':
        label_smoothing = config['training'].get('label_smoothing', 0.0)
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def load_fold_priors(priors_dir: Path, fold_idx: int) -> Dict:
    """Load per-fold XAI priors for CoInfo-GRU."""
    priors_file = priors_dir / f"fold_{fold_idx}_priors.json"
    if not priors_file.exists():
        print(f"  Warning: No priors at {priors_file}")
        return {}

    with open(priors_file) as f:
        priors = json.load(f)
    print(f"  Loaded fold {fold_idx} priors: U_i range [{min(priors['U_i']):.3f}, {max(priors['U_i']):.3f}]")
    return priors


def forward_model(model, batch, model_name: str, device: torch.device):
    """
    Unified forward pass for both novel and baseline models.

    Novel models: input is tuple of tensors, output is logits tensor
    Baseline models: input is dict of tensors, output is {"logits": tensor}
    """
    modalities, labels, _ = batch
    labels = labels.to(device)

    if is_novel_model(model_name):
        # Novel model: tuple input, tensor output
        mods = tuple(m.to(device) for m in modalities)
        logits = model(mods)
    else:
        # Baseline model: dict input, dict output
        # Convert tuple to dict format
        modality_names = modalities[0] if isinstance(modalities[0], str) else None
        if isinstance(modalities, (list, tuple)) and not isinstance(modalities[0], str):
            # Data comes as tuple from LOSOSequenceDataset
            # Need modality names from config
            raise ValueError(
                "Baseline models require dict-format data. "
                "Use a dataset adapter or pass modality_names."
            )
        x_dict = {name: tensor.to(device) for name, tensor in modalities.items()}
        output = model(x_dict)
        logits = output['logits']

    return logits, labels


def forward_novel(model, batch, device):
    """Forward pass for novel models (tuple interface)."""
    modalities, labels, _ = batch
    mods = tuple(m.to(device) for m in modalities)
    labels = labels.to(device)
    logits = model(mods)
    return logits, labels


def forward_baseline(model, batch, device, modality_names):
    """Forward pass for baseline models (dict interface).

    Data from LOSOSingleWindowDataset arrives as tuple of (B, 1, seq_len) tensors.
    With T=1 format, this IS already (B, T=1, D=seq_len) — pass directly.
    The Encoder does: transpose(1,2) -> Conv1d(D, 30) -> permute(2,0,1) -> (1, B, 30).
    """
    modalities, labels, _ = batch
    labels = labels.to(device)

    # Build x_dict: {name: (B, T=1, D=seq_len)}
    # No reshape needed — dataset returns (B, 1, seq_len) which is (B, T, D)
    x_dict = {}
    for i, name in enumerate(modality_names):
        x_dict[name] = modalities[i].to(device)

    output = model(x_dict)
    logits = output['logits']
    return logits, labels


def evaluate(model, data_loader, device, model_name, temperature=1.0,
             modality_names=None):
    """Evaluate model on a dataset, returning metrics dict."""
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            if is_novel_model(model_name):
                logits, labels = forward_novel(model, batch, device)
            else:
                logits, labels = forward_baseline(model, batch, device, modality_names)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return evaluate_predictions(all_logits, all_labels, temperature)


def train_one_epoch(model, train_loader, criterion, optimizer, device, model_name,
                    grad_clip=1.0, modality_names=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        if is_novel_model(model_name):
            logits, labels = forward_novel(model, batch, device)
        else:
            logits, labels = forward_baseline(model, batch, device, modality_names)

        optimizer.zero_grad()

        # Get gate/interaction params for co-info regularization
        if hasattr(model, 'get_gate_and_interaction_params') and hasattr(criterion, 'u_loss'):
            gates, interactions = model.get_gate_and_interaction_params()
            loss = criterion(logits, labels, gates, interactions)
        else:
            loss = criterion(logits, labels)

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


def get_val_logits(model, val_loader, device, model_name, modality_names=None):
    """Collect all logits and labels from validation set."""
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            if is_novel_model(model_name):
                logits, labels = forward_novel(model, batch, device)
            else:
                logits, labels = forward_baseline(model, batch, device, modality_names)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_logits, 0), torch.cat(all_labels, 0)


def train_fold(fold_idx: int, config: Dict, data_module, device: torch.device,
               results_dir: Path) -> Dict:
    """
    Train and evaluate a single LOSO fold.

    Works for both novel and baseline models.
    """
    model_name = config['model']['name']
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx} | Model: {model_name}")
    print(f"{'='*60}")

    # Get fold info
    fold_info = data_module.get_fold_info(fold_idx)
    test_subj = fold_info.get('test_subject', fold_info.get('test', {}).get('subjects', ['?']))
    print(f"Test subject: {test_subj}")

    # Determine if using sequences (for novel temporal models)
    context_length = config['dataset'].get('context_length', 1)
    use_sequences = context_length > 1 and is_novel_model(model_name)

    # For baseline models, always use single-window mode
    if not is_novel_model(model_name):
        use_sequences = False

    train_loader, val_loader, test_loader = data_module.get_dataloaders(
        fold_idx, use_sequences=use_sequences
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")

    # Build model
    mod_info = data_module.get_modality_info()
    modality_names = mod_info['names']
    n_modalities = mod_info['n_modalities']

    model_config = dict(config['model'])
    model_config['model_name'] = model_name
    model_config['n_classes'] = config['dataset']['n_classes']

    if is_novel_model(model_name):
        model_config['modality_dims'] = [1] * n_modalities
        model_config['modality_seq_lens'] = mod_info['rates']
    else:
        # Baseline: T=1 format — each 1-sec window is a single token
        # data_dims = sampling rate (Conv1d input channels)
        # seq_dims = 1 (one token per modality for transformer)
        model_config['seq_dims'] = {name: 1 for name in modality_names}
        model_config['data_dims'] = {name: rate for name, rate in zip(modality_names, mod_info['rates'])}

    model = build_model(model_config, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # Compute class weights if requested
    class_weights = None
    if config['training'].get('class_weights') == 'balanced':
        train_labels = train_loader.dataset.labels
        n_classes = config['dataset']['n_classes']
        class_weights = compute_class_weights(train_labels, n_classes, device)
        print(f"Class weights (balanced): {class_weights.cpu().tolist()}")

    # Loss
    criterion = build_criterion(config, n_modalities, device, class_weights=class_weights)

    # Load fold-specific XAI priors for combined loss
    if config['training'].get('loss_type') == 'combined':
        priors_dir = config['training'].get('priors_dir')
        if priors_dir:
            fold_priors = load_fold_priors(Path(priors_dir), fold_idx)
            if fold_priors.get('U_i') is not None and hasattr(criterion, 'set_coinfo_targets'):
                criterion.set_coinfo_targets(
                    U_i=torch.tensor(fold_priors['U_i']),
                    C_ij=torch.tensor(fold_priors['C_ij']) if fold_priors.get('C_ij') else None
                )

    # Optimizer & scheduler
    lr = config['training'].get('lr', 1e-3)
    weight_decay = config['training'].get('weight_decay', 0.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched_patience = config['training'].get('scheduler_patience', 10)
    sched_factor = config['training'].get('scheduler_factor', 0.5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=sched_patience, factor=sched_factor)

    # Training loop
    epochs = config['training'].get('epochs', 100)
    patience = config['training'].get('patience', 10)
    grad_clip = config['training'].get('grad_clip', 1.0)

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, model_name,
            grad_clip=grad_clip, modality_names=modality_names
        )

        val_metrics = evaluate(model, val_loader, device, model_name,
                               modality_names=modality_names)
        val_loss = val_metrics['nll_bits']
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
            marker = " ★ new best"
        else:
            patience_counter += 1
            marker = ""

        print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} ({train_acc*100:.1f}%) | "
              f"Val: {val_metrics['accuracy']*100:.1f}% F1={val_metrics['f1_macro']*100:.1f}% "
              f"NLL={val_loss:.4f}{marker}",
              flush=True)

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})", flush=True)
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Temperature scaling
    val_logits, val_labels = get_val_logits(
        model, val_loader, device, model_name, modality_names=modality_names
    )
    opt_temp, _ = find_optimal_temperature(val_logits, val_labels)
    print(f"Optimal temperature: {opt_temp:.3f}")

    # Final evaluation
    val_metrics = evaluate(model, val_loader, device, model_name,
                           temperature=opt_temp, modality_names=modality_names)
    test_metrics = evaluate(model, test_loader, device, model_name,
                            temperature=opt_temp, modality_names=modality_names)

    print(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Test F1 Macro: {test_metrics['f1_macro']*100:.2f}%")

    # Save
    fold_dir = results_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': best_model_state,
        'optimal_temperature': opt_temp,
        'config': config,
        'fold_idx': fold_idx,
    }, fold_dir / "model.pt")

    result = {
        'fold_idx': fold_idx,
        'test_subject': str(test_subj),
        'optimal_temperature': opt_temp,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'n_params': n_params,
    }

    with open(fold_dir / "metrics.json", 'w') as f:
        json.dump(result, f, indent=2)

    return result


def train_loso(config: Dict):
    """Train model with LOSO cross-validation."""
    model_name = config['model']['name']
    dataset_name = config['dataset']['name']

    print("=" * 60)
    print(f"MULTIMODAL STRESS DETECTION - LOSO Training")
    print(f"Model: {model_name} | Dataset: {dataset_name}")
    print("=" * 60)

    seed = config['training'].get('seed', 42)
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Results directory
    results_dir = Path(config['training']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(results_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Build data module
    data_module = build_data_module(config)
    n_folds = data_module.n_folds
    print(f"LOSO folds: {n_folds}")

    # Determine which folds to run
    fold_arg = config.get('fold')
    if fold_arg is not None:
        folds_to_run = [fold_arg]
    else:
        folds_to_run = list(range(n_folds))

    # Train all folds
    results = []
    for fold_idx in folds_to_run:
        result = train_fold(fold_idx, config, data_module, device, results_dir)
        results.append(result)

    # Aggregate
    if len(results) > 1:
        summary = aggregate_fold_results(results_dir)
        summary['config'] = config
        with open(results_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print_summary(summary)

    return results


def main():
    parser = argparse.ArgumentParser(description="Multimodal Stress Detection - LOSO Training")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--fold', type=int, default=None,
                        help='Specific fold to train (default: all)')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Override results directory from config')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override random seed')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max epochs (useful for verification)')
    args = parser.parse_args()

    config = load_config(args.config)

    # Apply overrides
    if args.fold is not None:
        config['fold'] = args.fold
    if args.results_dir:
        config['training']['results_dir'] = args.results_dir
    if args.seed is not None:
        config['training']['seed'] = args.seed
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs

    train_loso(config)


if __name__ == '__main__':
    main()
