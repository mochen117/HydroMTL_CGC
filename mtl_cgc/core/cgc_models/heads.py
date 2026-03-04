import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional


class RegressionHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_dim: Optional[int] = None, 
                 dropout: float = 0.0, activation: str = "relu", **kwargs):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        if hidden_dim is None:
            hidden_dim = in_features
            
        self.activation_fn = self._get_activation(activation)
        
        layers = []
        
        if hidden_dim > 0 and hidden_dim != in_features:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(self.activation_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim
        
        layers.append(nn.Linear(in_features, out_features))
        
        self.network = nn.Sequential(*layers)
        
    def _get_activation(self, activation: str):
        activation = activation.lower()
        if activation == "relu":
            return nn.ReLU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU(0.1)
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GMMHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_components: int = 3,
                 hidden_dim: Optional[int] = None, dropout: float = 0.0, 
                 activation: str = "relu", **kwargs):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_components = n_components
        
        if hidden_dim is None:
            hidden_dim = in_features
            
        self.activation_fn = self._get_activation(activation)
        
        self.shared_layers = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            self.activation_fn,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            self.activation_fn
        )
        
        self.mean_layer = nn.Linear(hidden_dim, out_features * n_components)
        self.var_layer = nn.Linear(hidden_dim, out_features * n_components)
        self.weight_layer = nn.Linear(hidden_dim, n_components)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.mean_layer.weight)
        nn.init.xavier_uniform_(self.var_layer.weight)
        nn.init.xavier_uniform_(self.weight_layer.weight)
        nn.init.constant_(self.mean_layer.bias, 0.0)
        nn.init.constant_(self.var_layer.bias, 0.0)
        nn.init.constant_(self.weight_layer.bias, 0.0)
    
    def _get_activation(self, activation: str):
        activation = activation.lower()
        if activation == "relu":
            return nn.ReLU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU(0.1)
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.shared_layers(x)
        
        means = self.mean_layer(features)
        means = means.view(-1, self.n_components, self.out_features)
        
        variances = F.softplus(self.var_layer(features))
        variances = variances.view(-1, self.n_components, self.out_features)
        
        weights = F.softmax(self.weight_layer(features), dim=-1)
        
        return {
            'means': means,
            'variances': variances,
            'weights': weights
        }


class CMALHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                 hidden_dim: Optional[int] = None, num_heads: int = 4,
                 dropout: float = 0.1, activation: str = "relu", **kwargs):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        
        if hidden_dim is None:
            hidden_dim = in_features * 2
            
        self.activation_fn = self._get_activation(activation)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            self.activation_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.input_projection = nn.Linear(in_features, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, out_features)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.constant_(self.input_projection.bias, 0.0)
        nn.init.constant_(self.output_projection.bias, 0.0)
        nn.init.xavier_uniform_(self.query)
    
    def _get_activation(self, activation: str):
        activation = activation.lower()
        if activation == "relu":
            return nn.ReLU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU(0.1)
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        x_proj = self.input_projection(x)
        x_proj = self.activation_fn(x_proj)
        x_proj = self.dropout(x_proj)
        
        query = self.query.expand(batch_size, -1, -1)
        
        attn_output, _ = self.attention(query, x_proj, x_proj)
        attn_output = self.dropout(attn_output)
        
        output = self.norm1(attn_output + query)
        
        ffn_output = self.ffn(output)
        
        output = self.norm2(ffn_output + output)
        
        output = self.output_projection(output)
        
        output = output.squeeze(1)
        
        return output


def get_head(head_type: str, in_features: int, out_features: int, **kwargs) -> nn.Module:
    head_type = head_type.lower()
    
    if head_type == 'regression':
        return RegressionHead(in_features, out_features, **kwargs)
    elif head_type == 'gmm':
        return GMMHead(in_features, out_features, **kwargs)
    elif head_type == 'cmal':
        return CMALHead(in_features, out_features, **kwargs)
    else:
        raise ValueError(f"Unknown head type: {head_type}. Available types: 'regression', 'gmm', 'cmal'")