# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Multi-task deep learning architectures for streamflow prediction.
# Houses STL, HPS, MMoE, and CGC with asymmetric specific expert compatibility.
# ==============================================================================

import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional, Any

from mtl_cgc.core.cgc_models.cgc_layer import CGCLayer
from mtl_cgc.core.cgc_models.heads import get_head

class HydroBaseEncoder(nn.Module):
    """LSTM sequence processor for meteorological dynamic forcings."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(x)
        return torch.nan_to_num(h_n[-1, :, :], nan=0.0)


class StaticEmbedder(nn.Module):
    """Integrates both categorical features and numeric catchment attributes."""
    def __init__(self, 
                 static_dim: int, 
                 categorical_features: List[str], 
                 num_classes_dict: Dict[str, int], 
                 embed_dims_dict: Dict[str, int]):
        super().__init__()
        self.cat_features = categorical_features
        self.embs = nn.ModuleList()
        total_emb_dim = 0
        
        if self.cat_features:
            for cat_name in self.cat_features:
                num_classes = num_classes_dict.get(cat_name, 20)
                dim_e = embed_dims_dict.get(cat_name, 8)
                self.embs.append(nn.Embedding(num_embeddings=num_classes + 1, embedding_dim=dim_e, padding_idx=0))
                total_emb_dim += dim_e
                
        self.mlp = nn.Sequential(
            nn.Linear(static_dim + total_emb_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, stat_num: torch.Tensor, stat_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.cat_features and stat_cat is not None:
            cat_list = [emb(stat_cat[:, i].long()) for i, emb in enumerate(self.embs)]
            cat_repr = torch.cat(cat_list, dim=-1)
            inp = torch.cat([stat_num, cat_repr], dim=-1)
        else:
            inp = stat_num
        return self.mlp(inp)


class HydroMTL_STL(nn.Module):
    """Single Task Learning network with completely disjoint model parameters."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.task_names = [str(t['name']).lower() for t in config['data']['targets']]
        dyn_dim = len(config['data'].get('dynamic_features', []))
        stat_dim = len(config['data'].get('static_features', []))
        enc_dim = config['model'].get('encoder', {}).get('hidden_dim', 256)
        
        self.encoders = nn.ModuleDict({t: HydroBaseEncoder(dyn_dim, enc_dim) for t in self.task_names})
        self.embedders = nn.ModuleDict({
            t: StaticEmbedder(stat_dim, config['data'].get('categorical_static_features', []),
                             config['data'].get('categorical_num_classes', {}),
                             config['data'].get('categorical_embed_dims', {})) for t in self.task_names
        })
        self.heads = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(enc_dim + 128, 128), nn.ReLU(), nn.Linear(128, 1)) for t in self.task_names
        })

    def forward(self, dyn_x: torch.Tensor, stat_num: torch.Tensor, stat_cat: Optional[torch.Tensor] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        preds = {}
        for t in self.task_names:
            rep_d = self.encoders[t](dyn_x)
            rep_s = self.embedders[t](stat_num, stat_cat)
            preds[t] = self.heads[t](torch.cat([rep_d, rep_s], dim=-1))
        return preds, {}


class HydroMTL_HPS(nn.Module):
    """Hard Parameter Sharing framework."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.task_names = [str(t['name']).lower() for t in config['data']['targets']]
        dyn_dim = len(config['data'].get('dynamic_features', []))
        stat_dim = len(config['data'].get('static_features', []))
        enc_dim = config['model'].get('encoder', {}).get('hidden_dim', 256)
        
        self.encoder = HydroBaseEncoder(dyn_dim, enc_dim)
        self.embedder = StaticEmbedder(stat_dim, config['data'].get('categorical_static_features', []),
                                      config['data'].get('categorical_num_classes', {}),
                                      config['data'].get('categorical_embed_dims', {}))
        self.heads = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(enc_dim + 128, 128), nn.ReLU(), nn.Linear(128, 1)) for t in self.task_names
        })

    def forward(self, dyn_x: torch.Tensor, stat_num: torch.Tensor, stat_cat: Optional[torch.Tensor] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        rep_d = self.encoder(dyn_x)
        rep_s = self.embedder(stat_num, stat_cat)
        feat = torch.cat([rep_d, rep_s], dim=-1)
        preds = {t: self.heads[t](feat) for t in self.task_names}
        return preds, {}


class HydroMTL_MMoE(nn.Module):
    """Multi-gate Mixture-of-Experts architecture."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.task_names = [str(t['name']).lower() for t in config['data']['targets']]
        dyn_dim = len(config['data'].get('dynamic_features', []))
        stat_dim = len(config['data'].get('static_features', []))
        enc_dim = config['model'].get('encoder', {}).get('hidden_dim', 256)
        
        m_cfg = config['model'].get('mmoe', {})
        self.num_experts = m_cfg.get('num_experts', 4)
        
        self.expert_encoders = nn.ModuleList([HydroBaseEncoder(dyn_dim, enc_dim) for _ in range(self.num_experts)])
        self.expert_embedders = nn.ModuleList([
            StaticEmbedder(stat_dim, config['data'].get('categorical_static_features', []),
                           config['data'].get('categorical_num_classes', {}),
                           config['data'].get('categorical_embed_dims', {})) for _ in range(self.num_experts)
        ])
        
        gate_dim = enc_dim + 128
        self.gates = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(gate_dim, self.num_experts), nn.Softmax(dim=-1)) for t in self.task_names
        })
        self.heads = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(gate_dim, 128), nn.ReLU(), nn.Linear(128, 1)) for t in self.task_names
        })

    def forward(self, dyn_x: torch.Tensor, stat_num: torch.Tensor, stat_cat: Optional[torch.Tensor] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        expert_outputs = []
        for i in range(self.num_experts):
            rep_d = self.expert_encoders[i](dyn_x)
            rep_s = self.expert_embedders[i](stat_num, stat_cat)
            expert_outputs.append(torch.cat([rep_d, rep_s], dim=-1).unsqueeze(1))
            
        expert_tensor = torch.cat(expert_outputs, dim=1)
        query_rep = expert_tensor.mean(dim=1)
        
        preds = {}
        gate_weights = {}
        for t in self.task_names:
            g = self.gates[t](query_rep)
            gate_weights[f"gate_{t}"] = g
            mixed_feat = torch.sum(expert_tensor * g.unsqueeze(-1), dim=1)
            preds[t] = self.heads[t](mixed_feat)
            
        return preds, gate_weights


class HydroMTL_CGC(nn.Module):
    """Customized Gate Control architecture preserving private task expert parameters."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.task_names = [str(t['name']).lower() for t in config['data']['targets']]
        self.num_tasks = len(self.task_names)
        
        dyn_dim = len(config['data'].get('dynamic_features', []))
        stat_dim = len(config['data'].get('static_features', []))
        enc_dim = config['model'].get('encoder', {}).get('hidden_dim', 256)
        
        cgc_cfg = config['model'].get('cgc', {})
        self.num_shared = cgc_cfg.get('shared_experts', 4)
        self.task_experts = cgc_cfg.get('task_experts', [4, 2])
        self.temperature = cgc_cfg.get('temperature', 1.0)
        
        self.shared_encoder = HydroBaseEncoder(dyn_dim, enc_dim)
        self.shared_embedder = StaticEmbedder(
            stat_dim, 
            config['data'].get('categorical_static_features', []),
            config['data'].get('categorical_num_classes', {}),
            config['data'].get('categorical_embed_dims', {})
        )
        
        # Build the exact customized gate control block mapping
        self.cgc_layer = CGCLayer(
            in_dim=enc_dim + 128, 
            out_dim=enc_dim + 128, 
            n_shared=self.num_shared, 
            n_task_list=self.task_experts,
            drop=cgc_cfg.get('dropout_rate', 0.3),
            temperature=self.temperature
        )
        
        self.heads = nn.ModuleDict({
            t: get_head('regression', enc_dim + 128, 1) for t in self.task_names
        })

    def forward(self, dyn_x: torch.Tensor, stat_num: torch.Tensor, stat_cat: Optional[torch.Tensor] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        rep_d = self.shared_encoder(dyn_x)
        rep_s = self.shared_embedder(stat_num, stat_cat)
        shared_feat = torch.cat([rep_d, rep_s], dim=-1)
        
        task_feats, gate_weights = self.cgc_layer(shared_feat)
        
        preds = {}
        for i, t in enumerate(self.task_names):
            preds[t] = self.heads[t](task_feats[i])
            
        return preds, gate_weights


_ARCH_REGISTRY = {
    "stl": HydroMTL_STL,
    "hps": HydroMTL_HPS,
    "mmoe": HydroMTL_MMoE,
    "cgc": HydroMTL_CGC
}

def build_model(config: Dict[str, Any]) -> nn.Module:
    """Builds specific multi-task architectures corresponding to the config."""
    # Standard dict-style get with default fallback
    model_cfg = config.get('model', {})
    arch = str(model_cfg.get('architecture', 'cgc')).lower()
    if arch not in _ARCH_REGISTRY:
        raise KeyError(f"Selected architecture '{arch}' is missing from the build registry. "
                       f"Available: {list(_ARCH_REGISTRY.keys())}")
    return _ARCH_REGISTRY[arch](config)