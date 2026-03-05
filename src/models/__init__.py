"""
Model Registry for Multimodal Stress Detection

All models accept a unified config dict and return {"logits": (B, C)} from forward().
"""

from src.models.coinfo_gru import HierarchicalCoInfoModel, WindowOnlyModel
from src.models.husformer import Husformer
from src.models.h2 import H2
from src.models.phemonet import PHemoNet
from src.models.hyperfusenet import HyperFuseNet


MODELS = {
    'coinfo_gru': HierarchicalCoInfoModel,
    'window_only': WindowOnlyModel,
    'husformer': Husformer,
    'h2': H2,
    'phemonet': PHemoNet,
    'hyperfusenet': HyperFuseNet,
}


def build_model(config, device='cpu'):
    """
    Build a model from config dict.

    For CoInfo-GRU / WindowOnly (novel models):
        Uses modality_dims, modality_seq_lens, embed_dim, etc. from config.
        Input: tuple of (B, C, T) tensors
        Output: (B, n_classes) logits

    For baselines (Husformer, H2, PHemoNet, HyperFuseNet):
        Uses model_cfg, seq_dims, data_dims from config.
        Input: dict {modality_name: (B, T, D)} tensors
        Output: {"logits": (B, n_classes)}

    Args:
        config: Dict with model configuration
        device: Target device

    Returns:
        model: nn.Module on device
    """
    import torch

    model_name = config['model_name']
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    if model_name in ('coinfo_gru', 'window_only'):
        # Novel model: parameterized by modality specs
        kwargs = {
            'modality_dims': config['modality_dims'],
            'modality_seq_lens': config['modality_seq_lens'],
            'embed_dim': config.get('embed_dim', 30),
            'hidden_channels': config.get('hidden_channels', 32),
            'encoder_type': config.get('encoder_type', 'multiscale_cnn'),
            'fusion_type': config.get('fusion_type', 'coinfo'),
            'output_dim': config['n_classes'],
            'dropout': config.get('dropout', 0.1),
        }
        if model_name == 'coinfo_gru':
            kwargs.update({
                'temporal_type': config.get('temporal_type', 'gru'),
                'temporal_hidden': config.get('temporal_hidden', 128),
                'temporal_layers': config.get('temporal_layers', 2),
            })
        model = MODELS[model_name](**kwargs)

    else:
        # Baseline model: uses v1 interface
        model_cfg = config.get('model_cfg', _default_baseline_cfg(model_name))
        seq_dims = config['seq_dims']
        data_dims = config['data_dims']
        n_classes = config['n_classes']
        model = MODELS[model_name](model_cfg, seq_dims, data_dims, n_classes)

    return model.to(device) if isinstance(device, torch.device) else model.to(torch.device(device))


def _default_baseline_cfg(model_name):
    """Default config for baseline models."""
    if model_name == 'husformer':
        return {
            'modality_specific_encoder': {
                'num_heads': 5,
                'layers': 5,
                'attn_dropout': 0.05,
                'relu_dropout': 0.1,
                'res_dropout': 0.1,
                'out_dropout': 0.1,
                'embed_dropout': 0.1,
                'attn_mask': True,
            },
            'trans_final': {
                'fusion_type': 'concat',
                'num_heads': 5,
                'layers': 5,
                'attn_dropout': 0.05,
                'relu_dropout': 0.1,
                'res_dropout': 0.1,
                'embed_dropout': 0.1,
                'attn_mask': True,
            },
        }
    else:
        # H2, PHemoNet, HyperFuseNet don't use model_cfg internally
        return {}
