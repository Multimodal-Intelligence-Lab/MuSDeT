"""
Evaluation Metrics for Multimodal Stress Detection

Includes accuracy, F1, confusion matrix, NLL, ECE, and calibration utilities.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import Dict, Optional


def compute_nll_bits(logits: torch.Tensor, targets: torch.Tensor, temperature: float = 1.0) -> float:
    """
    Compute mean NLL in bits (log base 2).

    Args:
        logits: (N, C) raw logits
        targets: (N,) class indices
        temperature: Temperature scaling factor

    Returns:
        Mean NLL in bits
    """
    scaled_logits = logits / temperature
    log_probs = F.log_softmax(scaled_logits, dim=1)
    nll_nats = F.nll_loss(log_probs, targets, reduction='mean').item()
    return nll_nats / np.log(2)


def compute_ece(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15,
                temperature: float = 1.0) -> float:
    """
    Compute Expected Calibration Error.

    Args:
        logits: (N, C) raw logits
        targets: (N,) class indices
        n_bins: Number of bins
        temperature: Temperature scaling

    Returns:
        ECE value
    """
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=1)
    confidences, predictions = probs.max(dim=1)

    correct = predictions.eq(targets).float()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            avg_confidence = confidences[in_bin].mean().item()
            avg_accuracy = correct[in_bin].mean().item()
            ece += in_bin.float().mean().item() * abs(avg_accuracy - avg_confidence)

    return ece


def compute_marginal_entropy_bits(labels: np.ndarray) -> float:
    """Compute H(Y) = -sum p(y) log2 p(y) in bits."""
    unique, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-10))


def evaluate_predictions(
    all_logits: torch.Tensor,
    all_labels: torch.Tensor,
    temperature: float = 1.0
) -> Dict:
    """
    Compute all evaluation metrics from logits and labels.

    Args:
        all_logits: (N, C) raw logits
        all_labels: (N,) class indices
        temperature: Temperature for scaling

    Returns:
        Dict with accuracy, f1_macro, f1_weighted, confusion_matrix, nll_bits, ece
    """
    scaled_logits = all_logits / temperature
    preds = scaled_logits.argmax(dim=1).numpy()
    labels_np = all_labels.numpy()

    return {
        'accuracy': float(accuracy_score(labels_np, preds)),
        'f1_macro': float(f1_score(labels_np, preds, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(labels_np, preds, average='weighted', zero_division=0)),
        'confusion_matrix': confusion_matrix(labels_np, preds).tolist(),
        'nll_bits': compute_nll_bits(all_logits, all_labels, temperature),
        'ece': compute_ece(all_logits, all_labels, temperature=temperature),
        'n_samples': len(labels_np),
    }


def find_optimal_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temps: np.ndarray = np.arange(0.5, 3.0, 0.05)
) -> tuple:
    """
    Find temperature that minimizes NLL on validation set.

    Returns:
        (best_temperature, best_nll_bits)
    """
    best_temp = 1.0
    best_nll = float('inf')

    for temp in temps:
        nll = compute_nll_bits(logits, labels, temp)
        if nll < best_nll:
            best_nll = nll
            best_temp = float(temp)

    return best_temp, best_nll
