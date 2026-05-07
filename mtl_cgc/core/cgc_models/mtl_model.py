# ==============================================================================
# Copyright (c) 2024-2025. All Rights Reserved.
# Author: Mochen & Project Contributors
# Description: Main Multi-Task Learning Architecture (HydroMTL_CGC).
# Implements physical data safeguards and handles multi-layer representations.
# ==============================================================================

import torch
import torch.nn as nn
from typing import Dict, Tuple
from .cgc_layer import CGCLayer
from .heads import get_head

class HydroMTL_CGC(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        data_cfg = config.get('data', {})
        model_cfg = config.get('model', {})
        
        self.targets = data_cfg.get('targets', [])
        self.task_names =[str(t.get('name', '')).lower() for t in self.targets]
        self.num_tasks = len(self.task_names)
        
        dyn_features = data_cfg.get('dynamic_features',[])
        stat_features = data_cfg.get('static_features',[])
        
        enc_cfg = model_cfg.get('encoder', {})
        self.lstm = nn.LSTM(
            input_size=len(dyn_features), 
            hidden_size=enc_cfg.get('hidden_dim', 256), 
            num_layers=enc_cfg.get('num_layers', 2),
            bidirectional=enc_cfg.get('bidirectional', False),
            batch_first=True
        )
        lstm_out_dim = enc_cfg.get('hidden_dim', 256) * (2 if enc_cfg.get('bidirectional', False) else 1)
        
        self.cat_features = data_cfg.get('categorical_static_features',[])
        self.embs = nn.ModuleList()
        total_emb_dim = 0
        
        if self.cat_features:
            num_classes_dict = data_cfg.get('categorical_num_classes', {})
            emb_dims_dict = data_cfg.get('categorical_embed_dims', {})
            for cat_name in self.cat_features:
                num_c = num_classes_dict.get(cat_name, 20)
                dim_e = emb_dims_dict.get(cat_name, 8)
                self.embs.append(nn.Embedding(num_embeddings=num_c + 1, embedding_dim=dim_e, padding_idx=0))
                total_emb_dim += dim_e
                
        s_dim = model_cfg.get('cgc', {}).get('static_dim', len(stat_features))
        self.s_mlp = nn.Sequential(
            nn.Linear(s_dim + total_emb_dim, 128), 
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(model_cfg.get('cgc', {}).get('dropout_rate', 0.3))
        )
        
        cgc_cfg = model_cfg.get('cgc', {})
        n_shared = cgc_cfg.get('shared_experts', 4)
        expert_dim = cgc_cfg.get('expert_hidden_dim', 256)
        
        raw_n_task_list = cgc_cfg.get('task_experts',[4, 2])
        if len(raw_n_task_list) < self.num_tasks:
            self.n_task_list = list(raw_n_task_list) + [1] * (self.num_tasks - len(raw_n_task_list))
        elif len(raw_n_task_list) > self.num_tasks:
            self.n_task_list = list(raw_n_task_list)[:self.num_tasks]
        else:
            self.n_task_list = list(raw_n_task_list)
            
        self.cgc = CGCLayer(
            in_dim=lstm_out_dim + 128, 
            out_dim=expert_dim, 
            n_shared=n_shared, 
            n_task_list=self.n_task_list, 
            drop=cgc_cfg.get('dropout_rate', 0.3)
        )
        
        tower_cfgs = model_cfg.get('task_towers',[])
        self.towers = nn.ModuleList()
        
        for i in range(self.num_tasks):
            t_cfg = tower_cfgs[i] if i < len(tower_cfgs) else {'hidden_dim': 128, 'output_head': 'regression'}
            self.towers.append(nn.Sequential(
                nn.Linear(expert_dim, t_cfg.get('hidden_dim', 128)), 
                nn.LayerNorm(t_cfg.get('hidden_dim', 128)),
                nn.ReLU(), 
                get_head(t_cfg.get('output_head', 'regression'), t_cfg.get('hidden_dim', 128))
            ))

    def forward(self, dyn_x: torch.Tensor, stat_num: torch.Tensor, stat_cat: torch.Tensor = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        dyn_x = torch.clamp(dyn_x, -20.0, 20.0)
        stat_num = torch.clamp(stat_num, -20.0, 20.0)

        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(dyn_x)
        
        if self.lstm.bidirectional:
            d_repr = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else:
            d_repr = h_n[-1,:,:]
            
        d_repr = torch.nan_to_num(d_repr, nan=0.0)
        
        if self.cat_features and stat_cat is not None:
            c_repr = torch.cat([emb(stat_cat[:, i].long()) for i, emb in enumerate(self.embs)], dim=-1)
            s_input = torch.cat([stat_num, c_repr], dim=-1)
        else:
            s_input = stat_num
            
        s_repr = self.s_mlp(s_input)
        
        cgc_input = torch.cat([d_repr, s_repr], dim=-1)
        cgc_input = torch.nan_to_num(cgc_input, nan=0.0)
        
        cgc_outs, gate_weights = self.cgc(cgc_input)
        
        preds_dict = {name: tower(torch.nan_to_num(out, nan=0.0)) for name, tower, out in zip(self.task_names, self.towers, cgc_outs)}
        
        return preds_dict, gate_weights