import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

from .mtl_model import FeatureEncoder, TaskTower

logger = logging.getLogger(__name__)

class Hard_MTL_Model(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        data_cfg = config.get('data', {})
        model_cfg = config.get('model', config)
        
        self.task_names = [t['name'] for t in data_cfg.get('targets', [])]
        
        self.categorical_features = data_cfg.get('categorical_static_features', [])
        self.categorical_embeddings = nn.ModuleDict()
        total_cat_embed_dim = 0
        
        for feat in self.categorical_features:
            num_classes = data_cfg.get('categorical_num_classes', {}).get(feat, 150)
            embed_dim = data_cfg.get('categorical_embed_dims', {}).get(feat, 8)
            self.categorical_embeddings[feat] = nn.Embedding(num_classes + 1, embed_dim, padding_idx=0)
            nn.init.normal_(self.categorical_embeddings[feat].weight, mean=0.0, std=0.1)
            total_cat_embed_dim += embed_dim
            
        num_numerical_features = len(data_cfg.get('static_features', [])) + len(data_cfg.get('dynamic_features', []))
        
        encoder_config = model_cfg['encoder']
        self.encoder = FeatureEncoder(
            input_dim=num_numerical_features,
            hidden_dim=encoder_config['hidden_dim'],
            num_layers=encoder_config['num_layers'],
            bidirectional=encoder_config.get('bidirectional', False),
            encoder_type=encoder_config['type'],
            dropout_rate=model_cfg.get('cgc', {}).get('dropout_rate', 0.5)
        )
        
        combined_dim = self.encoder.output_dim + total_cat_embed_dim
        
        self.task_towers = nn.ModuleList()
        for tower_config in model_cfg['task_towers']:
            tower = TaskTower(
                input_dim=combined_dim,
                hidden_dim=tower_config['hidden_dim'],
                num_layers=tower_config['num_layers'],
                tower_type=tower_config.get('type', 'mlp'),
                output_head_type=tower_config.get('output_head', 'regression'),
                output_dim=data_cfg.get('prediction_horizon', 1),
                dropout=model_cfg.get('cgc', {}).get('dropout_rate', 0.5)
            )
            self.task_towers.append(tower)

    def forward(self, x: torch.Tensor, categorical_features: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        encoded_time_features = self.encoder(x)
        
        if categorical_features is not None and len(self.categorical_features) > 0:
            cat_embeds = []
            for i, feat in enumerate(self.categorical_features):
                idx = categorical_features[:, i].long() if categorical_features.dim() == 2 else categorical_features[:, 0, i].long()
                cat_embeds.append(self.categorical_embeddings[feat](idx))
            
            cat_concat = torch.cat(cat_embeds, dim=-1)
            cat_concat = F.dropout(cat_concat, p=0.3, training=self.training)
            combined_features = torch.cat([encoded_time_features, cat_concat], dim=-1)
        else:
            combined_features = encoded_time_features
            
        predictions = {}
        for i, task_name in enumerate(self.task_names):
            task_pred = self.task_towers[i](combined_features)
            predictions[task_name] = task_pred['y_hat'] if isinstance(task_pred, dict) else task_pred
            
        return predictions