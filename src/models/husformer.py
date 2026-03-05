from src.models.modules.transformer import TransformerEncoder
from torch import nn
import torch
from torch.distributions import Normal
import torch.nn.functional as F

class MODEL(nn.Module):
    def __init__(self, cfg, seq_dims, data_dims, n_classes):
        super(MODEL, self).__init__()

        self.encoder= Encoder(cfg, data_dims)
        
        trans_final_cfg = cfg['trans_final']
        self.fusion_type = trans_final_cfg['fusion_type']

        embed_dim = 30
        self.modality_keys = list(data_dims.keys())

            
        if self.fusion_type =="mean":
            self.channels = next(iter(seq_dims.values())) 
        if self.fusion_type =="concat":
            self.channels = sum(seq_dims.values())

        self.fusion_head = FusionHead(
            cfg=cfg,
            embed_dim=embed_dim,
            n_classes=n_classes,
            channels=self.channels  
        )


        if cfg['auxilary_classification']:
            self.aux_head = nn.ModuleDict({
                m: ClassificationHead(embed_dim=embed_dim, n_classes=n_classes, channels=seq_dims[m])
                for m in self.modality_keys
            })

        self._init_weights()

    def _init_weights(self):
        def init_fn(m):
            # Linear
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_fn)
    
    def get_multimodal_parameters(self):
        params = (
            list(self.fusion_head.parameters())
        )
        return params
    
    def get_unimodal_parameters(self):
        params = (
            list(self.encoder.proj.parameters()) +
            list(self.encoder.trans_self.parameters()) +
            list(self.aux_head.parameters())
        )
        return params
    
    def uncertainty(self, embed, embed_aug):
        with torch.no_grad():
            var = torch.norm(embed - embed_aug,dim=1)
            u = 1-torch.exp(-var)
        return u

    def forward(self, x_dict, x_aug_dict=None):
        proj_tokens = self.encoder(x_dict)

        p_flat_dict = {
            m: proj_tokens[m].permute(1, 0, 2).reshape(proj_tokens[m].size(1), -1)
            for m in proj_tokens
        }

        if x_aug_dict is not None:
            
            proj_aug_tokens = self.encoder(x_aug_dict)
            
            p_aug_flat_dict = {
                m: proj_aug_tokens[m].permute(1, 0, 2).reshape(proj_aug_tokens[m].size(1), -1)
                for m in proj_aug_tokens
            }

            uncertainty = {m: self.uncertainty(p_aug_flat_dict[m], p_flat_dict[m]) for m in p_aug_flat_dict}
            u = torch.stack([uncertainty[m] for m in uncertainty], dim=1)  # (B,M)
            alpha = torch.softmax(-u / 0.5, dim=1)   

        if x_aug_dict is not None:
            if self.fusion_type == "mean":
                Z = torch.stack([proj_tokens[m] * alpha[:,i].reshape(1,-1,1) for i, m in enumerate(self.modality_keys)], dim=0).mean(dim=0)   # (T,B,D)
            elif self.fusion_type == "concat":
                Z = torch.cat([proj_tokens[m] * alpha[:,i].reshape(1,-1,1) for i, m in enumerate(self.modality_keys)], dim=0)                 # (M*T,B,D)

        else:
            if self.fusion_type == "mean":
                Z = torch.stack([proj_tokens[m] for m in self.modality_keys], dim=0).mean(dim=0)   # (T,B,D)
            elif self.fusion_type == "concat":
                Z = torch.cat([proj_tokens[m] for m in self.modality_keys], dim=0)                 # (M*T,B,D)

        logits, feature = self.fusion_head(Z)

        aux_logits_dict = {}
        if hasattr(self, "aux_head") and self.aux_head is not None:
            for m in self.modality_keys:
                if m not in proj_tokens:
                    continue
                aux_logits_dict[m] = self.aux_head[m](proj_tokens[m])

        output = {}
        output["logits"] = logits
        output["feature"] = feature
        output["p_flat"] = p_flat_dict
        output["aux_logits"] = aux_logits_dict

        return output
    
class Husformer(nn.Module):
    def __init__(self, cfg, seq_dims, data_dims, n_classes):
        super(Husformer, self).__init__()

        self.encoder= Husformer_Encoder(cfg, data_dims)
        
        trans_final_cfg = cfg['trans_final']
        self.fusion_type = trans_final_cfg['fusion_type']

        embed_dim = 30
        self.modality_keys = list(data_dims.keys())
            
        if self.fusion_type =="mean":
            self.channels = next(iter(seq_dims.values())) 
        if self.fusion_type =="concat":
            self.channels = sum(seq_dims.values())

        self.fusion_head = FusionHead(
            cfg=cfg,
            embed_dim=embed_dim,
            n_classes=n_classes,
            channels=self.channels  
        )

    def forward(self, x_dict):
        z_dict = self.encoder(x_dict)

        if self.fusion_type == "mean":
            Z = torch.stack([z_dict[m] for m in self.modality_keys], dim=0).mean(dim=0)   # (T,B,D)
        elif self.fusion_type == "concat":
            Z = torch.cat([z_dict[m] for m in self.modality_keys], dim=0)                 # (M*T,B,D)

        fusion_out = self.fusion_head(Z)
        # FusionHead returns (logits, feature) tuple
        logits = fusion_out[0] if isinstance(fusion_out, tuple) else fusion_out

        output = {}
        output["logits"] = logits

        return output

class FusionHead(nn.Module):
    def __init__(self, cfg, embed_dim, n_classes, channels):
        super(FusionHead, self).__init__()

        trans_cfg = cfg['trans_final']
        self.fusion_type = trans_cfg['fusion_type']

        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=trans_cfg['num_heads'],
            layers=trans_cfg['layers'],
            attn_dropout=trans_cfg['attn_dropout'],
            relu_dropout=trans_cfg['relu_dropout'],
            res_dropout=trans_cfg['res_dropout'],
            embed_dropout=trans_cfg['embed_dropout'],
            attn_mask=trans_cfg['attn_mask'],
        )

        self.final_conv = nn.Conv1d(
            channels, 1, kernel_size=1, padding=0, bias=False
        )

        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = self.transformer(x).permute(1, 0, 2)
        x = self.final_conv(x).squeeze(1) 
        logits = self.classifier(x)
        feature = x

        return logits, feature

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, n_classes, channels):
        super(ClassificationHead, self).__init__()

        self.final_conv = nn.Conv1d(
            channels, 1, kernel_size=1, padding=0, bias=False
        )

        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.final_conv(x).squeeze(1)  # (B, T)
        logits = self.classifier(x)

        return logits

class Encoder(nn.Module):
    def __init__(self, cfg, data_dims):

        super(Encoder, self).__init__()

        self.data_dims = data_dims
        mse_cfg = cfg['modality_specific_encoder']

        self.modality_keys = list(self.data_dims.keys())
        self.d_m = 30
        self.num_heads = mse_cfg['num_heads']
        self.layers = mse_cfg['layers']
        self.attn_dropout = mse_cfg['attn_dropout']
        self.relu_dropout = mse_cfg['relu_dropout']
        self.res_dropout = mse_cfg['res_dropout']
        self.out_dropout = mse_cfg['out_dropout']
        self.embed_dropout = mse_cfg['embed_dropout']
        self.attn_mask = mse_cfg['attn_mask']

        self.proj = nn.ModuleDict({
            m: nn.Conv1d(self.data_dims[m], self.d_m, kernel_size=1, padding=0, bias=False)
            for m in self.modality_keys
        })
        
        self.trans_self = nn.ModuleDict({
            m: self.get_network(layers=3)
            for m in self.modality_keys
        })

        # self.trans_all = nn.ModuleDict({
        #     m: self.get_network(layers=3)
        #     for m in self.modality_keys
        # })

        
    def get_network(self, layers=-1):
        embed_dim, attn_dropout = self.d_m, self.attn_dropout

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=self.layers,
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
    def uncertainty(self, embed, embed_aug):
        var = torch.norm(embed - embed_aug,dim=1)
        return 1-torch.exp(-var)

    def forward(self, x_dict): 
        proj_tokens = {} 

        for m in self.modality_keys:
            x = x_dict[m].transpose(1, 2)          # (B, D, T)
            x = self.proj[m](x)                    # (B, d_m, T)
            x = x.permute(2, 0, 1)                 # (T, B, d_m)

            x = self.trans_self[m](x)
            proj_tokens[m] = x

        return proj_tokens


class Husformer_Encoder(nn.Module):
    def __init__(self, cfg, data_dims):

        super(Husformer_Encoder, self).__init__()

        self.data_dims = data_dims
        mse_cfg = cfg['modality_specific_encoder']

        self.modality_keys = list(self.data_dims.keys())
        self.d_m = 30
        self.num_heads = mse_cfg['num_heads']
        self.layers = mse_cfg['layers']
        self.attn_dropout = mse_cfg['attn_dropout']
        self.relu_dropout = mse_cfg['relu_dropout']
        self.res_dropout = mse_cfg['res_dropout']
        self.out_dropout = mse_cfg['out_dropout']
        self.embed_dropout = mse_cfg['embed_dropout']
        self.attn_mask = mse_cfg['attn_mask']

        self.proj = nn.ModuleDict({
            m: nn.Conv1d(self.data_dims[m], self.d_m, kernel_size=1, padding=0, bias=False)
            for m in self.modality_keys
        })
        
        self.trans_all = nn.ModuleDict({
            m: self.get_network(layers=3)
            for m in self.modality_keys
        })

        
    def get_network(self, layers=-1):
        embed_dim, attn_dropout = self.d_m, self.attn_dropout

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=self.layers,
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
            
    def forward(self, x_dict): 
        proj_tokens = {} 

        for m in self.modality_keys:
            x = x_dict[m].transpose(1, 2)          # (B, D, T)
            x = self.proj[m](x)                    # (B, d_m, T)
            x = x.permute(2, 0, 1)                 # (T, B, d_m)
            proj_tokens[m] = x

        proj_all = torch.cat([proj_tokens[m] for m in self.modality_keys], dim=0)
        
        z_dict = {}
        for m in self.modality_keys:

            out_m = self.trans_all[m](proj_tokens[m], proj_all, proj_all)
            z_dict[m] = out_m
        
        return z_dict

