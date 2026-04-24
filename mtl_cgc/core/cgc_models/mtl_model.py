import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

from .cgc_layer import CGCLayer
from .heads import RegressionHead

logger = logging.getLogger(__name__)

class FeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 num_layers: int = 1, bidirectional: bool = False,
                 encoder_type: str = "lstm", dropout_rate: float = 0.5):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        if self.encoder_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=256,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True
            )
            self.output_dim = hidden_dim * (2 if bidirectional else 1)
        elif self.encoder_type == "gru":
            self.encoder = nn.GRU(
                input_size=256,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True
            )
            self.output_dim = hidden_dim * (2 if bidirectional else 1)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        
        if self.encoder_type in ["lstm", "gru"]:
            encoded, _ = self.encoder(x)
            if self.bidirectional:
                forward_last = encoded[:, -1, :self.hidden_dim]
                backward_first = encoded[:, 0, self.hidden_dim:]
                encoded_features = torch.cat([forward_last, backward_first], dim=-1)
            else:
                encoded_features = encoded[:, -1, :]
        
        encoded_features = self.layer_norm(encoded_features)
        return encoded_features

class TaskTower(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 1, tower_type: str = "mlp",
                 output_head_type: str = "regression",
                 output_dim: int = 1, dropout: float = 0.5):
        super().__init__()
        self.tower_type = tower_type.lower()
        self.output_head_type = output_head_type
        
        if self.tower_type != "mlp":
            logger.warning("Overriding tower_type to 'mlp' for task consistency.")
            self.tower_type = "mlp"
            
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        self.tower_layers = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_head = RegressionHead(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.tower_layers(x)
        predictions = self.output_head(features)
        return predictions

class HydroMTL_CGC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        data_cfg = config.get('data', {})
        model_cfg = config.get('model', config)
        
        self.num_tasks = len(model_cfg.get('task_towers', []))
        if self.num_tasks == 0:
            raise ValueError("No task_towers found in configuration.")
            
        self.categorical_features = data_cfg.get('categorical_static_features', [])
        self.categorical_embeddings = nn.ModuleDict()
        total_cat_embed_dim = 0
        
        for feat in self.categorical_features:
            num_classes = data_cfg.get('categorical_num_classes', {}).get(feat, 150)
            embed_dim = data_cfg.get('categorical_embed_dims', {}).get(feat, 8)
            self.categorical_embeddings[feat] = nn.Embedding(num_classes + 1, embed_dim, padding_idx=0)
            nn.init.normal_(self.categorical_embeddings[feat].weight, mean=0.0, std=0.1)
            total_cat_embed_dim += embed_dim
        
        num_numerical_features = len(data_cfg.get('static_features', [])) + \
                                 len(data_cfg.get('dynamic_features', []))
        
        encoder_config = model_cfg['encoder']
        cgc_config = model_cfg['cgc']
        
        self.encoder = FeatureEncoder(
            input_dim=num_numerical_features,
            hidden_dim=encoder_config['hidden_dim'],
            num_layers=encoder_config['num_layers'],
            bidirectional=encoder_config.get('bidirectional', False),
            encoder_type=encoder_config['type'],
            dropout_rate=cgc_config.get('dropout_rate', 0.5)
        )
        
        cgc_input_dim = self.encoder.output_dim + total_cat_embed_dim
        
        self.cgc_layer = CGCLayer(
            input_dim=cgc_input_dim,
            output_dim=cgc_config['expert_hidden_dim'],
            num_shared_experts=cgc_config['shared_experts'],
            num_task_experts=cgc_config['task_experts'],
            use_attention_gate=cgc_config.get('use_attention_gate', True),
            dropout_rate=cgc_config.get('dropout_rate', 0.5)
        )
        
        self.task_towers = nn.ModuleList()
        for tower_config in model_cfg['task_towers']:
            tower = TaskTower(
                input_dim=cgc_config['expert_hidden_dim'],
                hidden_dim=tower_config['hidden_dim'],
                num_layers=tower_config['num_layers'],
                tower_type=tower_config.get('type', 'mlp'),
                output_head_type=tower_config.get('output_head', 'regression'),
                output_dim=data_cfg.get('prediction_horizon', 1),
                dropout=cgc_config.get('dropout_rate', 0.5)
            )
            self.task_towers.append(tower)
        
        self.physics_config = model_cfg.get('physics_constraints', {})
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'categorical_embeddings' in name:
                continue 
            if 'weight' in name and param.dim() > 1:
                if 'lstm' in name or 'gru' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, categorical_features: Optional[torch.Tensor] = None,
                return_gate_analysis: bool = False) -> Dict[str, torch.Tensor]:
        
        encoded_time_features = self.encoder(x)
        
        if categorical_features is not None and len(self.categorical_features) > 0:
            cat_embeds = []
            for i, feat in enumerate(self.categorical_features):
                idx = categorical_features[:, i] if categorical_features.dim() == 2 else categorical_features[:, 0, i]
                cat_embeds.append(self.categorical_embeddings[feat](idx))
            cat_concat = torch.cat(cat_embeds, dim=-1)
            cat_concat = F.dropout(cat_concat, p=0.3, training=self.training)
            combined_features = torch.cat([encoded_time_features, cat_concat], dim=-1)
        else:
            combined_features = encoded_time_features
        
        cgc_outputs = self.cgc_layer(combined_features)
        
        predictions = {}
        for i, (cgc_out, tower) in enumerate(zip(cgc_outputs, self.task_towers)):
            task_pred = tower(cgc_out)
            task_name = self.config['data']['targets'][i]['name']
            predictions[task_name] = task_pred['y_hat'] if isinstance(task_pred, dict) else task_pred
        
        if return_gate_analysis:
            predictions['gate_analysis'] = self.cgc_layer.get_gate_analysis(combined_features)
            
        return predictions
    
    def apply_physics_constraints(self, predictions: Dict[str, torch.Tensor],
                                  precipitation: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if not self.physics_config.get('enabled', False):
            return predictions
        
        constrained_predictions = predictions.copy()
        for task_name, pred in predictions.items():
            if task_name in ['streamflow', 'et', 'usgsFlow', 'ET']:
                constrained_predictions[task_name] = torch.relu(pred)
        
        water_balance_config = self.physics_config.get('water_balance', {})
        if (water_balance_config.get('enabled', False) and precipitation is not None and
            'streamflow' in predictions and 'evapotranspiration' in predictions):
            streamflow_pred = predictions['streamflow']
            et_pred = predictions['evapotranspiration']
            water_imbalance = precipitation - streamflow_pred - et_pred
            constrained_predictions['water_imbalance'] = water_imbalance
        
        return constrained_predictions