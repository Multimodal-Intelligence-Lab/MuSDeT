"""
Temporal Heads for Across-Window State Estimation

Models temporal dependencies across multiple 1-second windows.
Key insight: Stress states persist over time - modeling temporal
context significantly improves LOSO generalization.

Features:
- GRU head (fast, effective, proven in Stage 3)
- Transformer head (ablation)
- Support for context length sweep
- Causal vs bidirectional modes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GRUTemporalHead(nn.Module):
    """
    GRU-based temporal head for sequence classification.

    Takes a sequence of window embeddings, outputs class logits.
    Uses the last hidden state for classification (causal).

    Args:
        input_dim: Input embedding dimension
        hidden_dim: GRU hidden dimension
        num_layers: Number of GRU layers
        output_dim: Number of output classes
        dropout: Dropout rate
        bidirectional: If True, use bidirectional GRU
    """

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output dimension depends on bidirectionality
        gru_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(gru_out_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) sequence of embeddings

        Returns:
            logits: (batch, output_dim)
        """
        # GRU forward
        output, h_n = self.gru(x)
        # h_n: (num_layers * num_directions, batch, hidden)

        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            # h_n[-2] is forward, h_n[-1] is backward
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            # Use last layer's hidden state
            last_hidden = h_n[-1]

        # Classify
        out = self.dropout(last_hidden)
        logits = self.classifier(out)

        return logits

    def forward_with_hidden(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both logits and hidden representation.

        Returns:
            logits: (batch, output_dim)
            hidden: (batch, hidden_dim) or (batch, 2*hidden_dim) if bidirectional
        """
        output, h_n = self.gru(x)

        if self.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]

        out = self.dropout(last_hidden)
        logits = self.classifier(out)

        return logits, last_hidden


class TransformerTemporalHead(nn.Module):
    """
    Transformer-based temporal head for sequence classification.

    Uses causal attention (each position only attends to past).

    Args:
        input_dim: Input embedding dimension
        hidden_dim: Transformer hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        output_dim: Number of output classes
        dropout: Dropout rate
        max_seq_len: Maximum sequence length (for positional embedding)
        causal: If True, use causal attention mask
    """

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        output_dim: int = 3,
        dropout: float = 0.3,
        max_seq_len: int = 128,
        causal: bool = True
    ):
        super().__init__()
        self.causal = causal
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) sequence of embeddings

        Returns:
            logits: (batch, output_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)

        # Add positional embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(positions)

        # Apply transformer with causal mask if needed
        if self.causal:
            mask = self._get_causal_mask(seq_len, x.device)
            x = self.transformer(x, mask=mask)
        else:
            x = self.transformer(x)

        # Use last position for classification (causal prediction)
        last_hidden = x[:, -1, :]

        # Classify
        out = self.dropout(last_hidden)
        logits = self.classifier(out)

        return logits

    def forward_with_hidden(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both logits and hidden representation.

        Returns:
            logits: (batch, output_dim)
            hidden: (batch, hidden_dim) - last position embedding
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)

        # Add positional embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(positions)

        # Apply transformer with causal mask if needed
        if self.causal:
            mask = self._get_causal_mask(seq_len, x.device)
            x = self.transformer(x, mask=mask)
        else:
            x = self.transformer(x)

        # Use last position for classification (causal prediction)
        last_hidden = x[:, -1, :]

        # Classify
        out = self.dropout(last_hidden)
        logits = self.classifier(out)

        return logits, last_hidden


class IdentityTemporalHead(nn.Module):
    """
    Identity temporal head (no temporal modeling).

    For ablation: what if we classify each window independently?
    Takes only the last embedding in the sequence.

    Args:
        input_dim: Input embedding dimension
        output_dim: Number of output classes
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int = 30,
        output_dim: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) sequence of embeddings

        Returns:
            logits: (batch, output_dim)
        """
        # Use only the last embedding
        last_emb = x[:, -1, :]
        out = self.dropout(last_emb)
        return self.classifier(out)


class PoolingTemporalHead(nn.Module):
    """
    Pooling-based temporal head.

    Aggregates sequence by mean/max pooling, no learned temporal modeling.
    For ablation: does simple aggregation suffice?

    Args:
        input_dim: Input embedding dimension
        output_dim: Number of output classes
        pooling: 'mean', 'max', or 'both'
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int = 30,
        output_dim: int = 3,
        pooling: str = 'mean',
        dropout: float = 0.3
    ):
        super().__init__()
        self.pooling = pooling

        if pooling == 'both':
            classifier_input = input_dim * 2
        else:
            classifier_input = input_dim

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(classifier_input, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) sequence of embeddings

        Returns:
            logits: (batch, output_dim)
        """
        if self.pooling == 'mean':
            pooled = x.mean(dim=1)
        elif self.pooling == 'max':
            pooled = x.max(dim=1)[0]
        else:  # both
            mean_pooled = x.mean(dim=1)
            max_pooled = x.max(dim=1)[0]
            pooled = torch.cat([mean_pooled, max_pooled], dim=1)

        out = self.dropout(pooled)
        return self.classifier(out)


def get_temporal_head(
    head_type: str,
    input_dim: int = 30,
    hidden_dim: int = 128,
    num_layers: int = 2,
    output_dim: int = 3,
    dropout: float = 0.3,
    **kwargs
) -> nn.Module:
    """
    Factory function for temporal heads.

    Args:
        head_type: 'gru', 'transformer', 'identity', 'mean_pool', 'max_pool'
        input_dim: Input embedding dimension
        hidden_dim: Hidden dimension (for GRU/Transformer)
        num_layers: Number of layers
        output_dim: Number of classes
        dropout: Dropout rate
        **kwargs: Additional arguments for specific heads

    Returns:
        Temporal head module
    """
    if head_type == 'gru':
        return GRUTemporalHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
            bidirectional=kwargs.get('bidirectional', False)
        )
    elif head_type == 'transformer':
        return TransformerTemporalHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=kwargs.get('num_heads', 4),
            output_dim=output_dim,
            dropout=dropout,
            max_seq_len=kwargs.get('max_seq_len', 128),
            causal=kwargs.get('causal', True)
        )
    elif head_type == 'identity':
        return IdentityTemporalHead(
            input_dim=input_dim,
            output_dim=output_dim,
            dropout=dropout
        )
    elif head_type in ['mean_pool', 'max_pool', 'both_pool']:
        pooling = head_type.replace('_pool', '')
        if pooling == 'both':
            pooling = 'both'
        return PoolingTemporalHead(
            input_dim=input_dim,
            output_dim=output_dim,
            pooling=pooling,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown temporal head type: {head_type}")


if __name__ == '__main__':
    # Test temporal heads
    batch_size = 16
    seq_len = 30
    input_dim = 30
    output_dim = 3

    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_dim)

    heads = ['gru', 'transformer', 'identity', 'mean_pool']

    for head_type in heads:
        head = get_temporal_head(head_type, input_dim=input_dim, output_dim=output_dim)
        out = head(x)
        n_params = sum(p.numel() for p in head.parameters())
        print(f"{head_type}: output {out.shape}, params {n_params:,}")
