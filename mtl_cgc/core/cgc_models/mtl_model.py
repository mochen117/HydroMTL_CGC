"""
Main Multi-Task Learning model with CGC architecture
Combines encoder, CGC layer, and task-specific towers
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

from .cgc_layer import CGCLayer
from .heads import get_head, RegressionHead, GMMHead, CMALHead

logger = logging.getLogger(__name__)

class FeatureEncoder(nn.Module):
    """Feature encoder for time series data"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 num_layers: int = 2, bidirectional: bool = True,
                 encoder_type: str = "lstm"):
        """
        Initialize feature encoder
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            bidirectional: Whether to use bidirectional RNN
            encoder_type: Type of encoder ('lstm', 'gru', 'transformer', 'cnn')
        """
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        if encoder_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0
            )
            self.output_dim = hidden_dim * (2 if bidirectional else 1)
            
        elif encoder_type == "gru":
            self.encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0
            )
            self.output_dim = hidden_dim * (2 if bidirectional else 1)
            
        elif encoder_type == "cnn":
            # Temporal CNN encoder
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, 
                         kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim*2,
                         kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim*2),
                nn.AdaptiveAvgPool1d(1)
            )
            self.output_dim = hidden_dim * 2
            
        elif encoder_type == "transformer":
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_projection = nn.Linear(input_dim, hidden_dim)
            self.output_dim = hidden_dim
            
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input features
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            
        Returns:
            Encoded features [batch_size, output_dim]
        """
        batch_size = x.shape[0]
        
        if self.encoder_type in ["lstm", "gru"]:
            # RNN-based encoders
            encoded, _ = self.encoder(x)
            # Take the last time step
            if self.bidirectional:
                # For bidirectional, concatenate last forward and first backward
                forward_last = encoded[:, -1, :self.hidden_dim]
                backward_first = encoded[:, 0, self.hidden_dim:]
                encoded_features = torch.cat([forward_last, backward_first], dim=-1)
            else:
                encoded_features = encoded[:, -1, :]
                
        elif self.encoder_type == "cnn":
            # CNN encoder expects [batch, channels, seq_len]
            x_transposed = x.transpose(1, 2)
            encoded = self.encoder(x_transposed)
            encoded_features = encoded.squeeze(-1)
            
        elif self.encoder_type == "transformer":
            # Transformer encoder
            encoded = self.encoder(x)
            # Global average pooling over sequence dimension
            encoded_features = encoded.mean(dim=1)
            encoded_features = self.output_projection(encoded_features)
        
        # Apply layer normalization
        encoded_features = self.layer_norm(encoded_features)
        
        return encoded_features

class TaskTower(nn.Module):
    """Task-specific tower for final prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 1, output_head_type: str = "regression",
                 output_dim: int = 1, dropout: float = 0.2):
        """
        Initialize task tower
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            output_head_type: Type of output head
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.output_head_type = output_head_type
        
        # Build tower layers
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        self.tower_layers = nn.Sequential(*layers) if layers else nn.Identity()
        
        # Create output head
        if output_head_type == "regression":
            self.output_head = RegressionHead(hidden_dim, output_dim)
        elif output_head_type == "gmm":
            self.output_head = GMMHead(hidden_dim, output_dim)
        elif output_head_type == "cmal":
            self.output_head = CMALHead(hidden_dim, output_dim)
        else:
            raise ValueError(f"Unsupported output head type: {output_head_type}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through task tower
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary with predictions
        """
        # Process through tower layers
        features = self.tower_layers(x)
        
        # Get predictions from output head
        predictions = self.output_head(features)
        
        return predictions

class HydroMTL_CGC(nn.Module):
    """
    Complete HydroMTL_CGC model for multi-task hydrological prediction
    
    Architecture:
    1. Feature Encoder: Extracts temporal features from input sequences
    2. CGC Layer: Customized Gate Control for multi-task routing
    3. Task Towers: Task-specific networks for final predictions
    """
    
    def __init__(self, config):
        """
        Initialize HydroMTL_CGC model
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        
        # Store configuration
        self.config = config
        self.num_tasks = len(config['task_towers'])
        
        # 1. Feature Encoder
        encoder_config = config['encoder']
        self.encoder = FeatureEncoder(
            input_dim=len(config['data']['static_features']) + 
                      len(config['data']['dynamic_features']),
            hidden_dim=encoder_config['hidden_dim'],
            num_layers=encoder_config['num_layers'],
            bidirectional=encoder_config.get('bidirectional', True),
            encoder_type=encoder_config['type']
        )
        
        # 2. CGC Layer
        cgc_config = config['cgc']
        self.cgc_layer = CGCLayer(
            input_dim=self.encoder.output_dim,
            output_dim=cgc_config['expert_hidden_dim'],
            num_shared_experts=cgc_config['shared_experts'],
            num_task_experts=cgc_config['task_experts'],
            use_attention_gate=cgc_config['use_attention_gate'],
            dropout_rate=cgc_config['dropout_rate']
        )
        
        # 3. Task Towers
        self.task_towers = nn.ModuleList()
        for i, tower_config in enumerate(config['task_towers']):
            tower = TaskTower(
                input_dim=cgc_config['expert_hidden_dim'],
                hidden_dim=tower_config['hidden_dim'],
                num_layers=tower_config['num_layers'],
                output_head_type=tower_config['output_head'],
                output_dim=config['data']['prediction_horizon'],
                dropout=cgc_config['dropout_rate']
            )
            self.task_towers.append(tower)
        
        # Physics constraint parameters
        self.physics_config = config.get('physics_constraints', {})
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized HydroMTL_CGC model with {self.num_tasks} tasks")
        logger.info(f"Encoder type: {encoder_config['type']}")
        logger.info(f"CGC: {cgc_config['shared_experts']} shared experts, "
                   f"{cgc_config['task_experts']} task experts")
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                if 'lstm' in name or 'gru' in name:
                    # Orthogonal initialization for RNNs
                    nn.init.orthogonal_(param)
                else:
                    # Xavier initialization for other layers
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, 
                return_gate_analysis: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model
        
        Args:
            x: Input features [batch_size, seq_len, features]
            return_gate_analysis: Whether to return gate analysis
            
        Returns:
            Dictionary with predictions and optional analysis
        """
        # Encode input features
        encoded_features = self.encoder(x)
        
        # Process through CGC layer
        cgc_outputs = self.cgc_layer(encoded_features)
        
        # Get task-specific predictions
        predictions = {}
        gate_analysis = {}
        
        for i, (cgc_out, tower) in enumerate(zip(cgc_outputs, self.task_towers)):
            task_pred = tower(cgc_out)
            
            # Store predictions with task-specific keys
            task_name = self.config['data']['targets'][i]['name']
            predictions[task_name] = task_pred
        
        # Get gate analysis if requested
        if return_gate_analysis:
            gate_analysis = self.cgc_layer.get_gate_analysis(encoded_features)
            predictions['gate_analysis'] = gate_analysis
        
        return predictions
    
    def apply_physics_constraints(self, predictions: Dict[str, torch.Tensor],
                                 precipitation: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Apply physics constraints to predictions
        
        Args:
            predictions: Model predictions dictionary
            precipitation: Precipitation data for water balance constraint
            
        Returns:
            Constrained predictions
        """
        if not self.physics_config.get('enabled', False):
            return predictions
        
        constrained_predictions = predictions.copy()
        
        # 1. Non-negativity constraint for streamflow and ET
        for task_name, pred_dict in predictions.items():
            if task_name in ['streamflow', 'et', 'usgsFlow', 'ET']:
                if 'y_hat' in pred_dict:
                    # Apply ReLU for non-negativity
                    pred_dict['y_hat'] = torch.relu(pred_dict['y_hat'])
                elif 'mu' in pred_dict:
                    # For probabilistic outputs, ensure positive mean
                    pred_dict['mu'] = torch.relu(pred_dict['mu'])
        
        # 2. Water balance constraint (if precipitation available)
        water_balance_config = self.physics_config.get('water_balance', {})
        if (water_balance_config.get('enabled', False) and 
            precipitation is not None and
            'streamflow' in predictions and 'et' in predictions):
            
            # Get streamflow and ET predictions
            streamflow_pred = predictions['streamflow'].get('y_hat', 
                                                          predictions['streamflow'].get('mu'))
            et_pred = predictions['et'].get('y_hat', predictions['et'].get('mu'))
            
            if streamflow_pred is not None and et_pred is not None:
                # Simplified water balance: P - Q - ET ≈ 0
                water_imbalance = precipitation - streamflow_pred - et_pred
                
                # Store imbalance for analysis
                constrained_predictions['water_imbalance'] = water_imbalance
        
        return constrained_predictions
    
    def get_model_summary(self) -> Dict[str, any]:
        """
        Get model summary statistics
        
        Returns:
            Dictionary with model summary
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_type': self.config['encoder']['type'],
            'num_tasks': self.num_tasks,
            'cgc_shared_experts': self.config['cgc']['shared_experts'],
            'cgc_task_experts': self.config['cgc']['task_experts'],
            'physics_constraints_enabled': self.physics_config.get('enabled', False)
        }
        
        return summary