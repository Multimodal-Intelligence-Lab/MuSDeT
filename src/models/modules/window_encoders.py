"""
Window Encoders for Per-Modality Feature Extraction

Multi-scale 1D CNN encoders that learn temporal patterns within each
1-second window. Replaces Husformer's simple linear projection.

Key features:
- Multi-scale kernels to capture patterns at different temporal scales
- Handles varying sampling rates per modality
- Global average pooling for fixed-size output
- LayerNorm + dropout for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class MultiScaleCNNEncoder(nn.Module):
    """
    Multi-scale 1D CNN encoder for a single modality.

    Architecture:
    - Parallel conv branches with different kernel sizes
    - Concatenate branch outputs
    - Global average pooling over time
    - LayerNorm + dropout

    Args:
        input_dim: Input channels (typically 1 for physiological signals)
        output_dim: Output embedding dimension
        kernel_sizes: List of kernel sizes for parallel branches
        hidden_channels: Number of channels per branch
        dropout: Dropout rate
        seq_len: Expected sequence length (for kernel size adjustment)
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 30,
        kernel_sizes: List[int] = [3, 5, 9, 15],
        hidden_channels: int = 32,
        dropout: float = 0.1,
        seq_len: int = 700
    ):
        super().__init__()
        self.output_dim = output_dim

        # Adjust kernel sizes for short sequences
        effective_kernels = [min(k, seq_len - 1) if seq_len > 1 else 1 for k in kernel_sizes]
        effective_kernels = [k if k % 2 == 1 else k - 1 for k in effective_kernels]  # Ensure odd
        effective_kernels = [max(k, 1) for k in effective_kernels]  # Ensure positive

        # Parallel conv branches
        self.branches = nn.ModuleList()
        for k in effective_kernels:
            branch = nn.Sequential(
                nn.Conv1d(input_dim, hidden_channels, kernel_size=k, padding=k // 2),
                nn.ReLU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=k, padding=k // 2),
                nn.ReLU()
            )
            self.branches.append(branch)

        # Combine branches
        total_channels = hidden_channels * len(effective_kernels)

        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(total_channels, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time) input tensor

        Returns:
            (batch, output_dim) embedding
        """
        # Apply each branch
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)  # (batch, hidden_channels, time)
            branch_outputs.append(out)

        # Concatenate along channel dimension
        combined = torch.cat(branch_outputs, dim=1)  # (batch, total_channels, time)

        # Global average pooling
        pooled = combined.mean(dim=2)  # (batch, total_channels)

        # Project to output dimension
        return self.proj(pooled)


class AdaptiveMultiScaleEncoder(nn.Module):
    """
    Adaptive multi-scale encoder that adjusts to modality characteristics.

    Different modalities have different sampling rates:
    - Chest signals: 700Hz (700 samples/sec)
    - Wrist BVP: 64Hz (64 samples/sec)
    - Wrist GSR: 4Hz (4 samples/sec)

    This encoder adapts kernel sizes based on the input sequence length.
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 30,
        base_kernel_sizes: List[int] = [3, 5, 9, 15],
        hidden_channels: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.base_kernel_sizes = base_kernel_sizes
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        # Will be initialized on first forward pass
        self._initialized = False
        self._encoder = None

    def _init_encoder(self, seq_len: int):
        """Initialize encoder based on actual sequence length."""
        self._encoder = MultiScaleCNNEncoder(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            kernel_sizes=self.base_kernel_sizes,
            hidden_channels=self.hidden_channels,
            dropout=self.dropout,
            seq_len=seq_len
        )
        # Move to same device
        if next(self.parameters(), None) is not None:
            self._encoder = self._encoder.to(next(self.parameters()).device)
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time)

        Returns:
            (batch, output_dim)
        """
        if not self._initialized:
            self._init_encoder(x.shape[2])

        return self._encoder(x)


class MultiModalityEncoder(nn.Module):
    """
    Encoder for all modalities.

    Creates separate encoders for each modality with appropriate
    architecture based on sampling rate.

    Args:
        modality_dims: List of input dimensions for each modality
        modality_seq_lens: List of sequence lengths for each modality
        output_dim: Output embedding dimension per modality
        hidden_channels: Hidden channels per branch
        dropout: Dropout rate
    """

    def __init__(
        self,
        modality_dims: List[int],
        modality_seq_lens: List[int],
        output_dim: int = 30,
        hidden_channels: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_modalities = len(modality_dims)
        self.output_dim = output_dim

        # Create encoder for each modality
        self.encoders = nn.ModuleList()
        for i, (dim, seq_len) in enumerate(zip(modality_dims, modality_seq_lens)):
            # Adjust kernel sizes based on sequence length
            if seq_len >= 100:
                kernel_sizes = [3, 5, 9, 15]
            elif seq_len >= 10:
                kernel_sizes = [3, 5, 7]
            else:
                kernel_sizes = [1, 3]

            encoder = MultiScaleCNNEncoder(
                input_dim=dim,
                output_dim=output_dim,
                kernel_sizes=kernel_sizes,
                hidden_channels=hidden_channels,
                dropout=dropout,
                seq_len=seq_len
            )
            self.encoders.append(encoder)

    def forward(
        self,
        modalities: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            modalities: Tuple of (batch, channels, time) tensors

        Returns:
            Tuple of (batch, output_dim) embeddings
        """
        embeddings = []
        for encoder, x in zip(self.encoders, modalities):
            emb = encoder(x)
            embeddings.append(emb)

        return tuple(embeddings)


class SimpleLinearEncoder(nn.Module):
    """
    Simple linear encoder (Husformer-style baseline).

    Uses 1x1 convolution to project input to output dimension,
    then mean pooling over time.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 30,
        dropout: float = 0.1
    ):
        super().__init__()
        self.proj = nn.Conv1d(input_dim, output_dim, kernel_size=1, padding=0, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time)

        Returns:
            (batch, output_dim)
        """
        out = self.proj(x)  # (batch, output_dim, time)
        out = out.mean(dim=2)  # (batch, output_dim)
        return self.dropout(out)


if __name__ == '__main__':
    # Test the encoders
    batch_size = 16

    # WESAD modality dimensions and sequence lengths
    dims = [1, 1, 1, 1, 1, 1]
    seq_lens = [700, 64, 700, 700, 700, 4]

    # Create encoder
    encoder = MultiModalityEncoder(
        modality_dims=dims,
        modality_seq_lens=seq_lens,
        output_dim=30
    )

    # Create dummy data
    modalities = tuple(
        torch.randn(batch_size, dim, seq_len)
        for dim, seq_len in zip(dims, seq_lens)
    )

    # Forward pass
    embeddings = encoder(modalities)

    print("Input shapes:")
    for i, m in enumerate(modalities):
        print(f"  Modality {i}: {m.shape}")

    print("\nOutput shapes:")
    for i, e in enumerate(embeddings):
        print(f"  Embedding {i}: {e.shape}")

    # Count parameters
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {n_params:,}")
