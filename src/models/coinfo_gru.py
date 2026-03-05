"""
Hierarchical Co-Information Model

Full model combining:
- M1: Per-modality window encoders (Multi-scale CNN)
- M2: Co-Info regularized fusion
- M3: Temporal head (GRU)
- M4: Subject conditioning (FiLM) - optional

This is the main novel architecture proposed in this project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict

# Handle both relative and absolute imports
try:
    from .modules.window_encoders import MultiModalityEncoder, SimpleLinearEncoder
    from .modules.fusion import CoInfoFusion, SimpleFusion, ConcatFusion, AttentionFusion, EqualFusion
    from .modules.temporal_heads import get_temporal_head
    from .modules.conditioning import FiLMConditioner
except ImportError:
    from src.models.modules.window_encoders import MultiModalityEncoder, SimpleLinearEncoder
    from src.models.modules.fusion import CoInfoFusion, SimpleFusion, ConcatFusion, AttentionFusion, EqualFusion
    from src.models.modules.temporal_heads import get_temporal_head
    from src.models.modules.conditioning import FiLMConditioner


class HierarchicalCoInfoModel(nn.Module):
    """
    Hierarchical Multimodal Model with Co-Information Priors.

    Architecture:
    1. Window Encoders: Per-modality Multi-scale CNN
    2. Fusion: Learnable gates + pairwise interactions (co-info regularized)
    3. Temporal Head: GRU over sequence of fused embeddings
    4. Classifier: Final linear layer

    Optional:
    - Subject conditioning via FiLM

    Args:
        modality_dims: Input dimensions for each modality
        modality_seq_lens: Sequence lengths for each modality
        embed_dim: Embedding dimension per modality
        hidden_channels: Hidden channels for CNN encoder
        fusion_type: 'coinfo', 'simple', 'concat', 'attention'
        temporal_type: 'gru', 'transformer', 'identity', 'mean_pool'
        temporal_hidden: Hidden dimension for temporal head
        temporal_layers: Number of temporal layers
        output_dim: Number of output classes
        dropout: Dropout rate
        use_conditioning: Whether to use subject conditioning
        cond_dim: Subject conditioning embedding dimension
    """

    def __init__(
        self,
        modality_dims: List[int],
        modality_seq_lens: List[int],
        embed_dim: int = 30,
        hidden_channels: int = 32,
        encoder_type: str = 'multiscale_cnn',
        fusion_type: str = 'coinfo',
        temporal_type: str = 'gru',
        temporal_hidden: int = 128,
        temporal_layers: int = 2,
        output_dim: int = 3,
        dropout: float = 0.1,
        use_conditioning: bool = False,
        cond_dim: int = 64
    ):
        super().__init__()
        self.n_modalities = len(modality_dims)
        self.embed_dim = embed_dim
        self.use_conditioning = use_conditioning
        self.fusion_type = fusion_type
        self.temporal_type = temporal_type
        self.encoder_type = encoder_type

        # M1: Window Encoders
        if encoder_type == 'linear':
            # Husformer-style: 1x1 conv + mean pool per modality
            self.encoder = nn.ModuleList([
                SimpleLinearEncoder(
                    input_dim=dim,
                    output_dim=embed_dim,
                    dropout=dropout
                )
                for dim in modality_dims
            ])
        else:
            self.encoder = MultiModalityEncoder(
                modality_dims=modality_dims,
                modality_seq_lens=modality_seq_lens,
                output_dim=embed_dim,
                hidden_channels=hidden_channels,
                dropout=dropout
            )

        # M2: Fusion
        if fusion_type == 'coinfo':
            self.fusion = CoInfoFusion(
                n_modalities=self.n_modalities,
                embed_dim=embed_dim,
                dropout=dropout
            )
        elif fusion_type == 'simple':
            self.fusion = SimpleFusion(
                n_modalities=self.n_modalities,
                embed_dim=embed_dim,
                dropout=dropout
            )
        elif fusion_type == 'concat':
            self.fusion = ConcatFusion(
                n_modalities=self.n_modalities,
                embed_dim=embed_dim,
                dropout=dropout
            )
        elif fusion_type == 'attention':
            self.fusion = AttentionFusion(
                n_modalities=self.n_modalities,
                embed_dim=embed_dim,
                dropout=dropout
            )
        elif fusion_type == 'equal':
            self.fusion = EqualFusion(
                n_modalities=self.n_modalities,
                embed_dim=embed_dim,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # M3: Temporal Head
        self.temporal_head = get_temporal_head(
            head_type=temporal_type,
            input_dim=embed_dim,
            hidden_dim=temporal_hidden,
            num_layers=temporal_layers,
            output_dim=output_dim,
            dropout=dropout
        )

        # M4: Subject Conditioning (optional)
        if use_conditioning:
            self.film = FiLMConditioner(
                cond_dim=cond_dim,
                feature_dim=embed_dim
            )
        else:
            self.film = None

    def _encode(self, modalities: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Encode modalities using the configured encoder type."""
        if self.encoder_type == 'linear':
            return tuple(enc(m) for enc, m in zip(self.encoder, modalities))
        else:
            return self.encoder(modalities)

    def forward(
        self,
        modalities: Tuple[torch.Tensor, ...],
        subject_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with automatic shape detection.

        Handles both single-window and sequence inputs:
        - Single window: modalities as (batch, channels, time)
        - Sequence: modalities as (batch, context_len, channels, time)

        The LOSOSequenceDataset returns (context_len, channels, time) per modality,
        which becomes (batch, context_len, channels, time) after DataLoader batching.

        Args:
            modalities: Tuple of tensors with shape:
                - Single window: (batch, channels, time)
                - Sequence: (batch, context_len, channels, time)
            subject_embedding: Optional (batch, cond_dim) for conditioning

        Returns:
            logits: (batch, output_dim)
        """
        # Detect if input is sequence (4D) or single window (3D)
        first_mod = modalities[0]
        is_sequence = first_mod.dim() == 4

        if is_sequence:
            # Sequence mode: (batch, context_len, channels, time)
            return self._forward_sequence(modalities, subject_embedding)
        else:
            # Single-window mode: (batch, channels, time)
            return self._forward_single(modalities, subject_embedding)

    def _forward_single(
        self,
        modalities: Tuple[torch.Tensor, ...],
        subject_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for single window classification.

        Args:
            modalities: Tuple of (batch, channels, time) tensors
            subject_embedding: Optional (batch, cond_dim) for conditioning

        Returns:
            logits: (batch, output_dim)
        """
        # Encode each modality
        embeddings = self._encode(modalities)  # Tuple of (batch, embed_dim)

        # Fuse
        fused = self.fusion(embeddings)  # (batch, embed_dim)

        # Apply conditioning if enabled
        if self.film is not None and subject_embedding is not None:
            fused = self.film(fused, subject_embedding)

        # Add sequence dimension for temporal head
        fused = fused.unsqueeze(1)  # (batch, 1, embed_dim)

        # Temporal head (degenerates to classifier for seq_len=1)
        logits = self.temporal_head(fused)

        return logits

    def _forward_sequence(
        self,
        modality_sequences: Tuple[torch.Tensor, ...],
        subject_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for sequence of windows (temporal modeling).

        CRITICAL: This is the proper temporal context mode where the GRU
        sees multiple windows. Stage C should use this with context_len > 1.

        Args:
            modality_sequences: Tuple of (batch, context_len, channels, time) tensors
            subject_embedding: Optional (batch, cond_dim) for conditioning

        Returns:
            logits: (batch, output_dim)
        """
        batch_size, seq_len = modality_sequences[0].shape[:2]

        # Process each window in the sequence
        fused_sequence = []

        for t in range(seq_len):
            # Get modalities for time step t: (batch, channels, time)
            mods_t = tuple(m[:, t] for m in modality_sequences)

            # Encode and fuse
            embeddings = self._encode(mods_t)
            fused = self.fusion(embeddings)

            # Apply conditioning
            if self.film is not None and subject_embedding is not None:
                fused = self.film(fused, subject_embedding)

            fused_sequence.append(fused)

        # Stack into sequence: (batch, seq_len, embed_dim)
        fused_sequence = torch.stack(fused_sequence, dim=1)

        # Temporal head processes the full sequence
        logits = self.temporal_head(fused_sequence)

        return logits

    def forward_sequence(
        self,
        modality_sequences: Tuple[torch.Tensor, ...],
        subject_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Explicit sequence forward pass (backwards compatibility).

        NOTE: The forward() method now auto-detects 4D inputs and routes
        to _forward_sequence(), so this method is usually not needed.

        Args:
            modality_sequences: Tuple of (batch, seq_len, channels, time) tensors
            subject_embedding: Optional (batch, cond_dim) for conditioning

        Returns:
            logits: (batch, output_dim)
        """
        return self._forward_sequence(modality_sequences, subject_embedding)

    def get_gate_and_interaction_params(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get current gate and interaction parameters for co-info regularization.

        Returns:
            gates: (n_modalities,) if fusion has gates, else None
            interactions: (n_pairs,) if fusion has interactions, else None
        """
        gates = None
        interactions = None

        if hasattr(self.fusion, 'gates'):
            gates = self.fusion.gates

        if hasattr(self.fusion, 'interactions_weights'):
            interactions = self.fusion.interactions_weights

        return gates, interactions

    def get_embeddings(
        self,
        modalities: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """
        Get fused embeddings without classification.

        Useful for embedding extraction and XAI.

        Args:
            modalities: Tuple of (batch, channels, time) tensors

        Returns:
            (batch, embed_dim) fused embeddings
        """
        embeddings = self._encode(modalities)
        fused = self.fusion(embeddings)
        return fused

    def get_sequence_embeddings(
        self,
        modality_sequences: Tuple[torch.Tensor, ...],
        return_window_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Get post-temporal hidden state for sequence inputs.

        Extracts the temporal head's internal representation (GRU h_n or
        Transformer last-token embedding) which is diagnostic for understanding
        temporal modeling quality.

        Args:
            modality_sequences: Tuple of (batch, seq_len, channels, time) tensors
            return_window_embeddings: If True, also return pre-temporal fused embeddings

        Returns:
            If return_window_embeddings=False:
                hidden: (batch, temporal_hidden_dim) post-temporal embedding
            If return_window_embeddings=True:
                (hidden, window_embeddings) where window_embeddings is (batch, seq_len, embed_dim)
        """
        batch_size, seq_len = modality_sequences[0].shape[:2]

        # Process each window in the sequence
        fused_sequence = []

        for t in range(seq_len):
            # Get modalities for time step t: (batch, channels, time)
            mods_t = tuple(m[:, t] for m in modality_sequences)

            # Encode and fuse
            embeddings = self._encode(mods_t)
            fused = self.fusion(embeddings)
            fused_sequence.append(fused)

        # Stack into sequence: (batch, seq_len, embed_dim)
        fused_sequence = torch.stack(fused_sequence, dim=1)

        # Get post-temporal hidden state
        if hasattr(self.temporal_head, 'forward_with_hidden'):
            _, hidden = self.temporal_head.forward_with_hidden(fused_sequence)
        else:
            # Fallback for heads without forward_with_hidden (e.g., identity)
            # Use the last fused embedding
            hidden = fused_sequence[:, -1, :]

        if return_window_embeddings:
            return hidden, fused_sequence
        return hidden


class WindowOnlyModel(nn.Module):
    """
    Model variant without temporal head (single-window classification).

    For Stage B ablation: test window encoder improvement in isolation.

    Args:
        modality_dims: Input dimensions for each modality
        modality_seq_lens: Sequence lengths for each modality
        embed_dim: Embedding dimension per modality
        hidden_channels: Hidden channels for CNN encoder
        fusion_type: 'coinfo', 'simple', 'concat', 'attention'
        output_dim: Number of output classes
        dropout: Dropout rate
    """

    def __init__(
        self,
        modality_dims: List[int],
        modality_seq_lens: List[int],
        embed_dim: int = 30,
        hidden_channels: int = 32,
        fusion_type: str = 'simple',
        output_dim: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_modalities = len(modality_dims)
        self.embed_dim = embed_dim

        # Window Encoders
        self.encoder = MultiModalityEncoder(
            modality_dims=modality_dims,
            modality_seq_lens=modality_seq_lens,
            output_dim=embed_dim,
            hidden_channels=hidden_channels,
            dropout=dropout
        )

        # Fusion
        if fusion_type == 'coinfo':
            self.fusion = CoInfoFusion(
                n_modalities=self.n_modalities,
                embed_dim=embed_dim,
                dropout=dropout
            )
        elif fusion_type == 'simple':
            self.fusion = SimpleFusion(
                n_modalities=self.n_modalities,
                embed_dim=embed_dim,
                dropout=dropout
            )
        else:
            self.fusion = ConcatFusion(
                n_modalities=self.n_modalities,
                embed_dim=embed_dim,
                dropout=dropout
            )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, modalities: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Args:
            modalities: Tuple of (batch, channels, time) tensors

        Returns:
            logits: (batch, output_dim)
        """
        embeddings = self.encoder(modalities)
        fused = self.fusion(embeddings)
        return self.classifier(fused)

    def get_gate_and_interaction_params(self):
        """Get gate and interaction parameters."""
        gates = None
        interactions = None

        if hasattr(self.fusion, 'gates'):
            gates = self.fusion.gates

        if hasattr(self.fusion, 'interactions_weights'):
            interactions = self.fusion.interactions_weights

        return gates, interactions


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the full model
    batch_size = 16
    seq_len = 30

    # WESAD modality specs
    dims = [1, 1, 1, 1, 1, 1]
    seq_lens = [700, 64, 700, 700, 700, 4]

    # Create model
    model = HierarchicalCoInfoModel(
        modality_dims=dims,
        modality_seq_lens=seq_lens,
        embed_dim=30,
        fusion_type='coinfo',
        temporal_type='gru',
        temporal_hidden=128,
        output_dim=3
    )

    print(f"Model parameters: {count_parameters(model):,}")

    # Test single-window forward
    print("\nSingle-window forward:")
    modalities = tuple(torch.randn(batch_size, d, s) for d, s in zip(dims, seq_lens))
    logits = model(modalities)
    print(f"  Input shapes: {[m.shape for m in modalities]}")
    print(f"  Output: {logits.shape}")

    # Test sequence forward (via explicit method)
    print("\nSequence forward (explicit forward_sequence):")
    mod_seq = tuple(torch.randn(batch_size, seq_len, d, s) for d, s in zip(dims, seq_lens))
    logits_seq = model.forward_sequence(mod_seq)
    print(f"  Input shapes: {[m.shape for m in mod_seq]}")
    print(f"  Output: {logits_seq.shape}")

    # Test sequence forward (via auto-detection in forward())
    print("\nSequence forward (auto-detection):")
    logits_seq_auto = model(mod_seq)  # Should auto-detect 4D and use _forward_sequence
    print(f"  Input shapes: {[m.shape for m in mod_seq]}")
    print(f"  Output: {logits_seq_auto.shape}")
    assert logits_seq_auto.shape == logits_seq.shape, "Auto-detection shape mismatch!"
    print("  ✓ Auto-detection works correctly!")

    # Get gate and interaction params
    gates, interactions = model.get_gate_and_interaction_params()
    print(f"\nGates: {gates.data if gates is not None else 'N/A'}")
    print(f"Interactions shape: {interactions.shape if interactions is not None else 'N/A'}")

    # Test WindowOnlyModel
    print("\n" + "="*50)
    print("WindowOnlyModel:")
    window_model = WindowOnlyModel(dims, seq_lens)
    print(f"Parameters: {count_parameters(window_model):,}")
    logits_w = window_model(modalities)
    print(f"Output: {logits_w.shape}")
