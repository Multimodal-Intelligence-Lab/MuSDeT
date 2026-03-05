"""
Co-Information Regularized Fusion Module

Fuses modality embeddings using:
- Learnable modality gates g_i (importance weights)
- Learnable pairwise interaction strengths a_ij

Co-info priors (soft constraints):
- L_U: regularize gates toward targets derived from unique information U_i
- L_C: regularize interactions toward targets from co-information C_ij

Key insight: Co-info is a PRIOR, not a hard-coded architecture.
The model learns gate/interaction values; we just guide them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class CoInfoFusion(nn.Module):
    """
    Fusion module with learnable gates and pairwise interactions.

    Architecture:
    1. Modality gates g_i (softmax over modalities): weighted sum of embeddings
    2. Pairwise interactions a_ij (sigmoid): bilinear/MLP on pairs

    Fusion computation:
        z = Σ_i g_i * z_i + Σ_{i<j} a_ij * φ(z_i, z_j)

    where φ is an interaction function (bilinear or MLP).

    Args:
        n_modalities: Number of modalities
        embed_dim: Embedding dimension per modality
        interaction_type: 'bilinear', 'mlp', or 'hadamard'
        interaction_hidden: Hidden dimension for MLP interaction
        dropout: Dropout rate
    """

    def __init__(
        self,
        n_modalities: int = 6,
        embed_dim: int = 30,
        interaction_type: str = 'hadamard',
        interaction_hidden: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_modalities = n_modalities
        self.embed_dim = embed_dim
        self.interaction_type = interaction_type

        # Learnable gate logits (pre-softmax)
        self.gate_logits = nn.Parameter(torch.zeros(n_modalities))

        # Pairwise interaction logits (pre-sigmoid)
        n_pairs = n_modalities * (n_modalities - 1) // 2
        self.interaction_logits = nn.Parameter(torch.zeros(n_pairs))

        # Interaction functions for each pair
        self.interactions = nn.ModuleList()
        for _ in range(n_pairs):
            if interaction_type == 'bilinear':
                interaction = nn.Bilinear(embed_dim, embed_dim, embed_dim)
            elif interaction_type == 'mlp':
                interaction = nn.Sequential(
                    nn.Linear(embed_dim * 3, interaction_hidden),
                    nn.ReLU(),
                    nn.Linear(interaction_hidden, embed_dim)
                )
            else:  # hadamard (simplest, fastest)
                interaction = nn.Linear(embed_dim, embed_dim)
            self.interactions.append(interaction)

        # Output projection
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Store pair indices for co-info computation
        self._pair_indices = []
        for i in range(n_modalities):
            for j in range(i + 1, n_modalities):
                self._pair_indices.append((i, j))

    @property
    def gates(self) -> torch.Tensor:
        """Get softmax-normalized gates."""
        return F.softmax(self.gate_logits, dim=0)

    @property
    def interactions_weights(self) -> torch.Tensor:
        """Get sigmoid-activated interaction weights."""
        return torch.sigmoid(self.interaction_logits)

    def forward(
        self,
        embeddings: Tuple[torch.Tensor, ...],
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Args:
            embeddings: Tuple of (batch, embed_dim) tensors, one per modality
            return_components: If True, also return gate-weighted sum and interaction term

        Returns:
            (batch, embed_dim) fused embedding
            Optionally: (fused, gated_sum, interaction_sum)
        """
        batch_size = embeddings[0].shape[0]

        # Stack embeddings: (batch, n_modalities, embed_dim)
        stacked = torch.stack(embeddings, dim=1)

        # Get gates and interactions
        gates = self.gates  # (n_modalities,)
        int_weights = self.interactions_weights  # (n_pairs,)

        # Gated sum: Σ_i g_i * z_i
        gates_expanded = gates.view(1, -1, 1)  # (1, n_modalities, 1)
        gated_sum = (stacked * gates_expanded).sum(dim=1)  # (batch, embed_dim)

        # Pairwise interactions: Σ_{i<j} a_ij * φ(z_i, z_j)
        interaction_sum = torch.zeros_like(gated_sum)

        for idx, (i, j) in enumerate(self._pair_indices):
            z_i = embeddings[i]
            z_j = embeddings[j]
            w = int_weights[idx]

            if self.interaction_type == 'bilinear':
                phi_ij = self.interactions[idx](z_i, z_j)
            elif self.interaction_type == 'mlp':
                concat = torch.cat([z_i, z_j, z_i * z_j], dim=1)
                phi_ij = self.interactions[idx](concat)
            else:  # hadamard
                phi_ij = self.interactions[idx](z_i * z_j)

            interaction_sum = interaction_sum + w * phi_ij

        # Combine
        fused = gated_sum + interaction_sum
        fused = self.layer_norm(self.dropout(fused))

        if return_components:
            return fused, gated_sum, interaction_sum

        return fused

    def get_gate_targets(self, U_i: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Compute gate targets from unique information.

        Higher U_i → higher gate value (more unique contribution).
        Handle negative U_i (harmful modality) by clamping.

        Args:
            U_i: (n_modalities,) unique information values
            temperature: Softmax temperature for conversion

        Returns:
            (n_modalities,) target gate distribution
        """
        # Clamp to [0, inf) - negative U_i means harmful, should have low gate
        U_positive = torch.clamp(U_i, min=0.0)

        # Convert to probability via softmax
        return F.softmax(U_positive / temperature, dim=0)

    def get_interaction_targets(
        self,
        C_ij: torch.Tensor,
        synergy_boost: float = 0.7,
        redundancy_penalty: float = 0.3
    ) -> torch.Tensor:
        """
        Compute interaction targets from co-information.

        C_ij < 0 → synergy → higher interaction (want to combine)
        C_ij > 0 → redundancy → lower interaction (avoid duplication)

        Args:
            C_ij: (n_pairs,) co-information values
            synergy_boost: Target for highly synergistic pairs
            redundancy_penalty: Target for highly redundant pairs

        Returns:
            (n_pairs,) target interaction weights in [0, 1]
        """
        # Normalize C_ij to [-1, 1] range approximately
        C_normalized = torch.tanh(C_ij)

        # Map: -1 (synergy) → synergy_boost, +1 (redundancy) → redundancy_penalty
        # Linear interpolation
        targets = synergy_boost + (C_normalized + 1) / 2 * (redundancy_penalty - synergy_boost)

        return targets


class SimpleFusion(nn.Module):
    """
    Simple fusion baseline (gated sum only, no interactions).

    For ablation studies comparing against CoInfoFusion.
    """

    def __init__(
        self,
        n_modalities: int = 6,
        embed_dim: int = 30,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_modalities = n_modalities
        self.embed_dim = embed_dim

        # Learnable gate logits
        self.gate_logits = nn.Parameter(torch.zeros(n_modalities))

        # Output
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    @property
    def gates(self) -> torch.Tensor:
        return F.softmax(self.gate_logits, dim=0)

    def forward(self, embeddings: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Args:
            embeddings: Tuple of (batch, embed_dim) tensors

        Returns:
            (batch, embed_dim) fused embedding
        """
        stacked = torch.stack(embeddings, dim=1)  # (batch, n_mod, dim)
        gates = self.gates.view(1, -1, 1)
        fused = (stacked * gates).sum(dim=1)
        return self.layer_norm(self.dropout(fused))


class EqualFusion(nn.Module):
    """
    Equal-weight fusion (no learnable gates).

    For ablation: tests whether learnable modality gates contribute.
    Uses fixed 1/N weighting for all modalities.
    """

    def __init__(
        self,
        n_modalities: int = 6,
        embed_dim: int = 30,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_modalities = n_modalities
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, embeddings: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        stacked = torch.stack(embeddings, dim=1)  # (batch, n_mod, dim)
        fused = stacked.mean(dim=1)  # Equal weighting = mean
        return self.layer_norm(self.dropout(fused))


class ConcatFusion(nn.Module):
    """
    Concatenation fusion baseline.

    Simply concatenates embeddings and projects down.
    """

    def __init__(
        self,
        n_modalities: int = 6,
        embed_dim: int = 30,
        dropout: float = 0.1
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_modalities * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, embeddings: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Args:
            embeddings: Tuple of (batch, embed_dim) tensors

        Returns:
            (batch, embed_dim) fused embedding
        """
        concat = torch.cat(embeddings, dim=1)  # (batch, n_mod * dim)
        return self.proj(concat)


class AttentionFusion(nn.Module):
    """
    Attention-based fusion.

    Uses self-attention over modality embeddings.
    """

    def __init__(
        self,
        n_modalities: int = 6,
        embed_dim: int = 30,
        num_heads: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, embeddings: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Args:
            embeddings: Tuple of (batch, embed_dim) tensors

        Returns:
            (batch, embed_dim) fused embedding (mean over modalities)
        """
        # Stack: (batch, n_mod, dim)
        stacked = torch.stack(embeddings, dim=1)

        # Self-attention
        attn_out, _ = self.attention(stacked, stacked, stacked)

        # Mean pool
        fused = attn_out.mean(dim=1)
        return self.layer_norm(fused)


if __name__ == '__main__':
    # Test fusion modules
    batch_size = 16
    n_modalities = 6
    embed_dim = 30

    # Create dummy embeddings
    embeddings = tuple(torch.randn(batch_size, embed_dim) for _ in range(n_modalities))

    # Test CoInfoFusion
    print("CoInfoFusion:")
    fusion = CoInfoFusion(n_modalities, embed_dim)
    fused, gated, interact = fusion(embeddings, return_components=True)
    print(f"  Fused: {fused.shape}")
    print(f"  Gated sum: {gated.shape}")
    print(f"  Interaction: {interact.shape}")
    print(f"  Gates: {fusion.gates.data}")
    print(f"  Interactions: {fusion.interactions_weights.data}")

    # Test gate/interaction targets
    U_i = torch.tensor([0.38, 0.30, -0.05, 0.13, 0.12, 0.12])  # From XAI results
    gate_targets = fusion.get_gate_targets(U_i)
    print(f"\n  U_i targets: {U_i.data}")
    print(f"  Gate targets: {gate_targets.data}")

    C_ij = torch.tensor([-0.37, -0.11, -0.10, +0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    int_targets = fusion.get_interaction_targets(C_ij)
    print(f"  C_ij[0:4]: {C_ij[:4].data}")
    print(f"  Int targets[0:4]: {int_targets[:4].data}")

    # Test other fusions
    print("\nSimpleFusion:")
    simple = SimpleFusion(n_modalities, embed_dim)
    print(f"  Output: {simple(embeddings).shape}")

    print("\nConcatFusion:")
    concat = ConcatFusion(n_modalities, embed_dim)
    print(f"  Output: {concat(embeddings).shape}")

    print("\nAttentionFusion:")
    attn = AttentionFusion(n_modalities, embed_dim)
    print(f"  Output: {attn(embeddings).shape}")

    # Count parameters
    for name, module in [('CoInfoFusion', fusion), ('SimpleFusion', simple),
                         ('ConcatFusion', concat), ('AttentionFusion', attn)]:
        n_params = sum(p.numel() for p in module.parameters())
        print(f"\n{name}: {n_params:,} parameters")
