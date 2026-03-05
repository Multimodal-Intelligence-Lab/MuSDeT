"""
Subject Conditioning Module (FiLM-based)

Implements subject-specific calibration/conditioning to improve
cross-subject generalization without GRL (which tends to hurt).

Key insight: Instead of removing subject-specific information (GRL),
we explicitly model it as a conditioning signal (FiLM).

FiLM (Feature-wise Linear Modulation):
- Learn scale (γ) and shift (β) parameters from subject embedding
- Apply to intermediate representations: γ * x + β

This allows the model to adapt to subject-specific baselines while
preserving task-relevant physiological patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class FiLMConditioner(nn.Module):
    """
    FiLM conditioning module.

    Generates scale (γ) and shift (β) parameters from a conditioning
    embedding and applies them to feature vectors.

    Args:
        cond_dim: Dimension of conditioning embedding
        feature_dim: Dimension of features to modulate
        hidden_dim: Hidden dimension for MLP
    """

    def __init__(
        self,
        cond_dim: int,
        feature_dim: int,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # MLP to generate γ and β from conditioning embedding
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim * 2)  # γ and β
        )

        # Initialize to identity transform (γ=1, β=0)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply FiLM conditioning.

        Args:
            x: (batch, feature_dim) or (batch, seq_len, feature_dim) features
            cond: (batch, cond_dim) conditioning embedding

        Returns:
            Modulated features with same shape as x
        """
        # Generate scale and shift
        params = self.mlp(cond)  # (batch, feature_dim * 2)
        gamma = params[:, :self.feature_dim] + 1.0  # Center around 1
        beta = params[:, self.feature_dim:]

        # Handle sequence dimension
        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return gamma * x + beta


class SubjectEncoder(nn.Module):
    """
    Encoder for computing subject-specific embedding from baseline windows.

    Given a set of baseline windows, computes a subject embedding that
    captures subject-specific physiological characteristics.

    Args:
        modality_dims: List of input dimensions for each modality
        modality_seq_lens: List of sequence lengths for each modality
        embed_dim: Output embedding dimension
        hidden_dim: Hidden dimension for projection
    """

    def __init__(
        self,
        modality_dims: List[int],
        modality_seq_lens: List[int],
        embed_dim: int = 64,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.n_modalities = len(modality_dims)

        # Simple encoders per modality (mean + variance features)
        total_features = self.n_modalities * 2  # mean and std per modality

        self.proj = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(
        self,
        baseline_windows: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """
        Compute subject embedding from baseline windows.

        Args:
            baseline_windows: Tuple of (n_windows, channels, time) per modality

        Returns:
            (1, embed_dim) subject embedding
        """
        features = []

        for mod in baseline_windows:
            # Compute mean and std across all windows and time
            mean = mod.mean().unsqueeze(0)
            std = mod.std().unsqueeze(0)
            features.extend([mean, std])

        features = torch.cat(features, dim=0).unsqueeze(0)  # (1, n_features)
        return self.proj(features)


class SubjectConditionedModel(nn.Module):
    """
    Wrapper that adds subject conditioning to any base model.

    Applies FiLM conditioning at specified injection points:
    - 'embedding': After modality encoding, before fusion
    - 'fused': After fusion, before temporal head
    - 'temporal': After temporal head, before classifier

    Args:
        base_model: Base model to condition
        cond_dim: Subject embedding dimension
        injection_points: List of where to apply conditioning
    """

    def __init__(
        self,
        base_model: nn.Module,
        cond_dim: int = 64,
        injection_points: List[str] = ['fused']
    ):
        super().__init__()
        self.base_model = base_model
        self.injection_points = injection_points

        # Create FiLM modules for each injection point
        self.film_modules = nn.ModuleDict()

        if 'embedding' in injection_points:
            # One FiLM per modality embedding
            n_mod = getattr(base_model, 'n_modalities', 6)
            embed_dim = getattr(base_model, 'embed_dim', 30)
            for i in range(n_mod):
                self.film_modules[f'embedding_{i}'] = FiLMConditioner(
                    cond_dim=cond_dim,
                    feature_dim=embed_dim
                )

        if 'fused' in injection_points:
            embed_dim = getattr(base_model, 'embed_dim', 30)
            self.film_modules['fused'] = FiLMConditioner(
                cond_dim=cond_dim,
                feature_dim=embed_dim
            )

        if 'temporal' in injection_points:
            hidden_dim = getattr(base_model, 'temporal_hidden_dim', 128)
            self.film_modules['temporal'] = FiLMConditioner(
                cond_dim=cond_dim,
                feature_dim=hidden_dim
            )

    def forward(
        self,
        modalities: Tuple[torch.Tensor, ...],
        subject_embedding: Optional[torch.Tensor] = None
    ):
        """
        Forward pass with optional subject conditioning.

        Args:
            modalities: Tuple of (batch, channels, time) tensors
            subject_embedding: (batch, cond_dim) subject embedding or None

        Returns:
            Model output (depends on base model)
        """
        # If no subject embedding, use base model directly
        if subject_embedding is None:
            return self.base_model(modalities)

        # Otherwise, forward with conditioning
        # This requires the base model to expose intermediate representations
        # Implementation depends on base model architecture
        raise NotImplementedError("Full conditioning requires custom forward pass")


class LearnableSubjectEmbedding(nn.Module):
    """
    Learnable embedding table for known subjects.

    For training on known subjects, we can learn an embedding directly.
    For new subjects at test time, use SubjectEncoder with baseline windows.

    Args:
        n_subjects: Number of known subjects
        embed_dim: Embedding dimension
    """

    def __init__(
        self,
        n_subjects: int,
        embed_dim: int = 64
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_subjects, embed_dim)

    def forward(self, subject_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            subject_ids: (batch,) subject indices

        Returns:
            (batch, embed_dim) embeddings
        """
        return self.embedding(subject_ids)


class AdaptiveLayerNormConditioner(nn.Module):
    """
    Adaptive Layer Normalization conditioning.

    Alternative to FiLM that modulates layer norm parameters.

    Args:
        cond_dim: Dimension of conditioning embedding
        feature_dim: Dimension of features to normalize
        hidden_dim: Hidden dimension for parameter generation
    """

    def __init__(
        self,
        cond_dim: int,
        feature_dim: int,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # Generate γ and β for layer norm
        self.param_gen = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim * 2)
        )

        # Initialize to standard layer norm (γ=1, β=0)
        nn.init.zeros_(self.param_gen[-1].weight)
        nn.init.zeros_(self.param_gen[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply adaptive layer normalization.

        Args:
            x: (batch, ..., feature_dim) features
            cond: (batch, cond_dim) conditioning

        Returns:
            Normalized features
        """
        # Standard layer norm
        x_norm = F.layer_norm(x, (self.feature_dim,))

        # Generate parameters
        params = self.param_gen(cond)
        gamma = params[:, :self.feature_dim] + 1.0
        beta = params[:, self.feature_dim:]

        # Reshape for broadcasting
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return gamma * x_norm + beta


if __name__ == '__main__':
    # Test conditioning modules
    batch_size = 16
    feature_dim = 30
    cond_dim = 64
    seq_len = 10

    # Create dummy data
    x = torch.randn(batch_size, feature_dim)
    x_seq = torch.randn(batch_size, seq_len, feature_dim)
    cond = torch.randn(batch_size, cond_dim)

    # Test FiLM
    print("FiLM Conditioner:")
    film = FiLMConditioner(cond_dim, feature_dim)
    out = film(x, cond)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    out_seq = film(x_seq, cond)
    print(f"  Sequence Input: {x_seq.shape} -> Output: {out_seq.shape}")

    # Test Adaptive LayerNorm
    print("\nAdaptive LayerNorm:")
    aln = AdaptiveLayerNormConditioner(cond_dim, feature_dim)
    out = aln(x, cond)
    print(f"  Input: {x.shape} -> Output: {out.shape}")

    # Test Subject Encoder
    print("\nSubject Encoder:")
    encoder = SubjectEncoder(
        modality_dims=[1, 1, 1, 1, 1, 1],
        modality_seq_lens=[700, 64, 700, 700, 700, 4]
    )
    baseline = tuple(torch.randn(10, 1, seq_len) for seq_len in [700, 64, 700, 700, 700, 4])
    subj_emb = encoder(baseline)
    print(f"  Subject embedding: {subj_emb.shape}")

    # Test Learnable Embedding
    print("\nLearnable Subject Embedding:")
    n_subjects = 15
    emb_table = LearnableSubjectEmbedding(n_subjects, cond_dim)
    subj_ids = torch.randint(0, n_subjects, (batch_size,))
    embs = emb_table(subj_ids)
    print(f"  Subject IDs: {subj_ids.shape} -> Embeddings: {embs.shape}")

    # Parameter counts
    for name, module in [('FiLM', film), ('AdaptiveLN', aln),
                         ('SubjectEncoder', encoder), ('LearnableEmb', emb_table)]:
        n_params = sum(p.numel() for p in module.parameters())
        print(f"\n{name}: {n_params:,} parameters")
