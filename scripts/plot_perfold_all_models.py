#!/usr/bin/env python3
"""Per-fold LOSO accuracy comparison for all models.

Grouped bar chart showing all 5 models per LOSO fold
on both WESAD and AffectiveROAD.

Output: figures/perfold_all_models.pdf
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

# Per-fold accuracy data (balanced class weights, seed=42)
# Source: per-fold metrics from LOSO evaluation (seed=42, balanced class weights)

WESAD_SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
                  "S10", "S11", "S13", "S14", "S15", "S16", "S17"]

WESAD = {
    "Husformer":    [47.6, 38.3, 75.9, 43.8, 60.6, 29.2, 59.4, 75.6, 38.3, 63.7, 22.8, 18.7, 75.8, 81.9, 39.4],
    "H2":           [65.0, 42.4, 62.5, 47.5, 66.1, 29.1, 69.6, 96.9, 46.4, 72.9, 29.8, 30.2, 92.3, 82.3, 70.5],
    "PHemoNet":     [57.9, 52.2, 76.4, 31.3, 54.0, 29.1, 69.4, 77.6, 50.2, 58.9, 29.8, 22.7, 83.7, 89.1, 64.7],
    "HyperFuseNet": [70.9, 40.7, 68.2, 25.8, 71.1, 29.1, 66.8, 92.1, 41.7, 72.9, 30.3, 23.3, 89.3, 80.7, 63.5],
    "MuSDeT":       [48.4, 47.8, 68.5, 56.2, 90.2, 46.1, 84.0, 98.4, 69.6, 84.2, 22.1, 26.9, 88.8, 84.6, 88.5],
}

AR_SUBJECTS = ["Drv1", "Drv3", "Drv4", "Drv5", "Drv6", "Drv7",
               "Drv8", "Drv9", "Drv10", "Drv11", "Drv12", "Drv13"]

AR = {
    "Husformer":    [62.4, 59.1, 37.3, 34.3, 48.3, 45.5, 29.6, 40.3, 47.5, 41.4, 61.5, 18.6],
    "H2":           [64.5, 50.2, 41.7, 30.2, 62.3, 44.0, 35.1, 31.2, 47.8, 41.7, 59.1,  2.1],
    "PHemoNet":     [64.8, 54.4, 37.0, 33.1, 50.0, 38.2, 30.9, 33.9, 55.0, 51.1, 59.1,  2.7],
    "HyperFuseNet": [64.9, 48.3, 37.0, 30.3, 67.0, 57.7, 33.8, 29.1, 45.1, 37.9, 59.3,  7.5],
    "MuSDeT":       [64.8, 56.2, 43.0, 62.7, 71.1, 61.9, 46.0, 31.2, 61.2, 46.7, 34.9,  1.2],
}

MODEL_ORDER = ["Husformer", "H2", "PHemoNet", "HyperFuseNet", "MuSDeT"]

COLORS = {
    "Husformer":    "#A8A8A8",
    "H2":           "#C4956A",
    "PHemoNet":     "#8FBC8F",
    "HyperFuseNet": "#B0A0CC",
    "MuSDeT":       "#2E86AB",
}
EDGECOLORS = {m: "none" for m in MODEL_ORDER}
EDGECOLORS["MuSDeT"] = "#1B5E7B"

BAR_WIDTH = 0.15

fig, axes = plt.subplots(2, 1, figsize=(16, 7), gridspec_kw={'hspace': 0.35})

for ax, data_dict, subjects, title in [
    (axes[0], WESAD, WESAD_SUBJECTS,
     f"WESAD (3-class, 15 LOSO folds) — MuSDeT mean: {np.mean(WESAD['MuSDeT']):.1f}%"),
    (axes[1], AR, AR_SUBJECTS,
     f"AffectiveROAD (binary, 12 LOSO folds) — MuSDeT mean: {np.mean(AR['MuSDeT']):.1f}%"),
]:
    n = len(subjects)
    x = np.arange(n)

    for i, model in enumerate(MODEL_ORDER):
        offset = (i - len(MODEL_ORDER) / 2 + 0.5) * BAR_WIDTH
        ax.bar(x + offset, data_dict[model], BAR_WIDTH,
               color=COLORS[model], edgecolor=EDGECOLORS[model],
               linewidth=0.5, label=model, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(subjects, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, zorder=0)

axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
               ncol=5, fontsize=8, frameon=True)

out_dir = Path("figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "perfold_all_models.pdf"
fig.savefig(out_path, bbox_inches="tight", dpi=300)
print(f"Saved: {out_path}")
