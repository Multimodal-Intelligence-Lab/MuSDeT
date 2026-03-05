#!/usr/bin/env python3
"""Extract embeddings and generate t-SNE visualization.

Extracts pre-temporal (30D) and post-temporal (128D) embeddings from
trained MuSDeT checkpoints, then generates t-SNE plots for representative
LOSO folds.

Requires trained model checkpoints (model.pt) from MuSDeT training.

Usage:
  python scripts/extract_and_plot_tsne.py --extract --plot
  python scripts/extract_and_plot_tsne.py --plot-only  # if embeddings already cached
"""

import argparse
import pickle
import numpy as np
import torch
from pathlib import Path
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

DATASET_CONFIGS = {
    'wesad': {
        'modality_dims': [1, 1, 1, 1, 1, 1],
        'modality_seq_lens': [700, 64, 700, 700, 700, 4],
        'modality_names': ['GSR_chest', 'BVP_wrist', 'EMG_chest', 'ECG_chest', 'RESP_chest', 'GSR_wrist'],
        'n_classes': 3,
        'class_names': ['Baseline', 'Stress', 'Amusement'],
        'class_colors': ['#27AE60', '#E74C3C', '#F39C12'],
        'data_file': 'data/wesad/loso_folds.pkl',
        'results_dir': 'results/wesad/musdet',
    },
    'affectiveroad': {
        'modality_dims': [1, 1, 1, 1, 1, 1, 1, 1],
        'modality_seq_lens': [64, 4, 1, 4, 64, 4, 1, 4],
        'modality_names': ['Left_BVP', 'Left_EDA', 'Left_HR', 'Left_TEMP', 'Right_BVP', 'Right_EDA', 'Right_HR', 'Right_TEMP'],
        'n_classes': 2,
        'class_names': ['Non-stressed', 'Stressed'],
        'class_colors': ['#3498DB', '#E74C3C'],
        'data_file': 'data/affectiveroad/loso_folds.pkl',
        'results_dir': 'results/affectiveroad/musdet',
    }
}


def load_fold_data(data_path, fold_idx):
    """Load pre-computed LOSO fold data."""
    print(f"Loading {data_path} ...")
    with open(data_path, 'rb') as f:
        all_folds = pickle.load(f)
    fold_key = f'fold_{fold_idx}'
    return all_folds[fold_key]


def extract_embeddings(model, data_loader, device):
    """Extract pre-temporal and post-temporal embeddings."""
    model.eval()
    pre_embs, post_embs, labels = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            modalities = tuple(m.to(device) for m in batch['modalities'])
            y = batch['label'].numpy()

            # Pre-temporal: fused per-window embeddings
            z_fused = model._fuse(model._encode(modalities))
            pre_embs.append(z_fused[:, -1].cpu().numpy())

            # Post-temporal: GRU output
            out = model(modalities)
            if hasattr(model, 'temporal_head'):
                h = model.temporal_head(z_fused)
                post_embs.append(h[:, -1].cpu().numpy() if h.dim() == 3 else h.cpu().numpy())
            labels.append(y)

    return {
        'pre_temporal': np.concatenate(pre_embs),
        'post_temporal': np.concatenate(post_embs),
        'labels': np.concatenate(labels),
    }


def run_tsne(embeddings, perplexity=30, seed=42):
    """Run t-SNE dimensionality reduction."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, n_iter=1000)
    return tsne.fit_transform(embeddings)


def generate_figure(wesad_npz, ar_npz, output_path):
    """Generate the 2x2 t-SNE figure."""
    wesad = np.load(wesad_npz)
    ar = np.load(ar_npz)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for row, (data, cfg_key) in enumerate([
        (wesad, 'wesad'), (ar, 'affectiveroad')
    ]):
        cfg = DATASET_CONFIGS[cfg_key]
        for col, emb_key in enumerate(['pre_temporal', 'post_temporal']):
            ax = axes[row, col]
            coords = run_tsne(data[emb_key])
            labels = data['labels']
            is_test = data['is_test']

            for cls_idx, (cls_name, color) in enumerate(zip(cfg['class_names'], cfg['class_colors'])):
                mask_train = (labels == cls_idx) & (~is_test)
                mask_test = (labels == cls_idx) & is_test
                ax.scatter(coords[mask_train, 0], coords[mask_train, 1],
                           c=color, s=15, alpha=0.4, label=cls_name)
                ax.scatter(coords[mask_test, 0], coords[mask_test, 1],
                           c=color, s=40, alpha=0.9, edgecolors='black', linewidths=0.8)

            ax.set_xticks([])
            ax.set_yticks([])

    axes[0, 0].set_title("Pre-temporal (30D)", fontsize=10, fontweight='bold')
    axes[0, 1].set_title("Post-temporal (128D)", fontsize=10, fontweight='bold')
    axes[0, 0].set_ylabel("WESAD", fontsize=9)
    axes[1, 0].set_ylabel("AffectiveROAD", fontsize=9)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', action='store_true', help='Extract embeddings from checkpoints')
    parser.add_argument('--plot-only', action='store_true', help='Plot from cached embeddings')
    parser.add_argument('--wesad-fold', type=int, default=2, help='WESAD fold index')
    parser.add_argument('--ar-fold', type=int, default=9, help='AR fold index')
    parser.add_argument('--output', type=str, default='figures/tsne_embeddings.pdf')
    args = parser.parse_args()

    wesad_npz = Path(f'embeddings_tsne/wesad_fold{args.wesad_fold}.npz')
    ar_npz = Path(f'embeddings_tsne/affectiveroad_fold{args.ar_fold}.npz')

    if args.plot_only:
        if wesad_npz.exists() and ar_npz.exists():
            generate_figure(wesad_npz, ar_npz, args.output)
        else:
            print(f"Missing cached embeddings. Run with --extract first.")
    elif args.extract:
        print("Extraction requires trained model checkpoints.")
        print("Run training first, then use this script to extract embeddings.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
