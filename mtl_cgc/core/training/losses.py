"""
Loss functions for hydrological multi-task learning
Includes base regression losses, multi-task balancing losses, and physics-constrained losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import torch.distributions as tdist


class BaseLoss(nn.Module):
    """Base class for all loss functions with gradient tracking"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MSELoss(BaseLoss):
    """Mean Squared Error loss"""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return loss


class NSELoss(BaseLoss):
    """Nash-Sutcliffe Efficiency loss (1 - NSE to minimize)"""
    
    def __init__(self, epsilon: float = 1e-6, reduction: str = 'mean'):
        super().__init__(reduction)
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure predictions and targets are same shape
        if pred.shape != target.shape:
            if pred.dim() == 3 and target.dim() == 2:
                target = target.unsqueeze(-1)
        
        # Calculate mean of targets
        target_mean = torch.mean(target, dim=0, keepdim=True)
        
        # Calculate numerator and denominator
        numerator = torch.sum((target - pred) ** 2, dim=0)
        denominator = torch.sum((target - target_mean) ** 2, dim=0)
        
        # Calculate NSE (1 - NSE for minimization)
        nse = 1 - numerator / (denominator + self.epsilon)
        
        if self.reduction == 'mean':
            return torch.mean(1 - nse)
        elif self.reduction == 'sum':
            return torch.sum(1 - nse)
        else:
            return 1 - nse


class KGELoss(BaseLoss):
    """Kling-Gupta Efficiency loss (1 - KGE to minimize)"""
    
    def __init__(self, epsilon: float = 1e-6, reduction: str = 'mean'):
        super().__init__(reduction)
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calculate components
        r = self._pearson_correlation(pred, target)
        alpha = torch.std(pred, unbiased=False) / (torch.std(target, unbiased=False) + self.epsilon)
        beta = torch.mean(pred) / (torch.mean(target) + self.epsilon)
        
        # Calculate KGE (1 - KGE for minimization)
        kge = 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        
        if self.reduction == 'mean':
            return torch.mean(1 - kge)
        elif self.reduction == 'sum':
            return torch.sum(1 - kge)
        else:
            return 1 - kge
    
    def _pearson_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate Pearson correlation coefficient"""
        x_centered = x - torch.mean(x)
        y_centered = y - torch.mean(y)
        
        numerator = torch.sum(x_centered * y_centered)
        denominator = torch.sqrt(torch.sum(x_centered ** 2) * torch.sum(y_centered ** 2))
        
        return numerator / (denominator + self.epsilon)


class GaussianNLLLoss(BaseLoss):
    """Gaussian Negative Log-Likelihood loss for probabilistic predictions"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction)
    
    def forward(self, pred_dict: Dict[str, torch.Tensor], 
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute NLL for Gaussian predictions
        
        Args:
            pred_dict: Dictionary with 'mu' and 'sigma' keys
            target: Ground truth values
            
        Returns:
            Negative log-likelihood loss
        """
        mu = pred_dict['mu']
        sigma = pred_dict['sigma']
        
        # Create Gaussian distribution
        dist = tdist.Normal(mu, sigma)
        
        # Compute negative log-likelihood
        nll = -dist.log_prob(target)
        
        if self.reduction == 'mean':
            return torch.mean(nll)
        elif self.reduction == 'sum':
            return torch.sum(nll)
        else:
            return nll


class MultiTaskLoss(nn.Module):
    """Multi-task loss with automatic balancing"""
    
    def __init__(self, task_losses: List[nn.Module], 
                 balancing_method: str = 'uncertainty',
                 initial_weights: Optional[List[float]] = None):
        """
        Initialize multi-task loss
        
        Args:
            task_losses: List of loss functions for each task
            balancing_method: 'uncertainty', 'dynamic', or 'fixed'
            initial_weights: Initial loss weights (if fixed)
        """
        super().__init__()
        self.task_losses = nn.ModuleList(task_losses)
        self.num_tasks = len(task_losses)
        self.balancing_method = balancing_method
        
        if balancing_method == 'uncertainty':
            # Learnable log variances for uncertainty weighting
            self.log_vars = nn.Parameter(torch.zeros(self.num_tasks))
        elif balancing_method == 'fixed':
            if initial_weights is None:
                initial_weights = [1.0] * self.num_tasks
            self.register_buffer('weights', torch.tensor(initial_weights))
        elif balancing_method == 'dynamic':
            # Dynamic task prioritization
            self.gamma = 2.0  # Focusing parameter
            self.alpha = 0.5  # Moving average parameter
            self.register_buffer('kpi_history', torch.zeros(self.num_tasks))
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor],
                task_names: List[str]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Dictionary of predictions for each task
            targets: Dictionary of targets for each task
            task_names: List of task names in order
            
        Returns:
            Total loss and dictionary of individual task losses
        """
        task_losses = {}
        total_loss = 0.0
        
        for i, task_name in enumerate(task_names):
            if task_name not in predictions or task_name not in targets:
                continue
            
            # Compute base loss for this task
            pred = predictions[task_name]
            target = targets[task_name]
            
            # Handle different prediction formats
            if isinstance(pred, dict):
                # Probabilistic output
                loss = self.task_losses[i](pred, target)
            else:
                # Deterministic output
                loss = self.task_losses[i](pred, target)
            
            task_losses[task_name] = loss
            
            # Apply balancing method
            if self.balancing_method == 'uncertainty':
                # Uncertainty weighting
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = precision * loss + 0.5 * self.log_vars[i]
                total_loss += weighted_loss
                
            elif self.balancing_method == 'fixed':
                # Fixed weights
                total_loss += self.weights[i] * loss
                
            elif self.balancing_method == 'dynamic':
                # Dynamic task prioritization
                # Compute KPI (e.g., correlation between pred and target)
                if isinstance(pred, dict):
                    pred_values = pred.get('mu', pred.get('y_hat', None))
                else:
                    pred_values = pred
                
                if pred_values is not None:
                    # Simplified KPI calculation (can be improved)
                    kpi = self._compute_kpi(pred_values, target)
                    # Update KPI history
                    self.kpi_history[i] = (self.alpha * kpi + 
                                          (1 - self.alpha) * self.kpi_history[i])
                    # Focal loss weighting
                    weight = (1 - self.kpi_history[i]) ** self.gamma
                    total_loss += weight * loss
        
        return total_loss, task_losses
    
    def _compute_kpi(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Key Performance Indicator for dynamic weighting"""
        # Use correlation as KPI
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        # Handle NaN values
        mask = ~torch.isnan(target_flat) & ~torch.isnan(pred_flat)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        pred_valid = pred_flat[mask]
        target_valid = target_flat[mask]
        
        # Compute correlation
        pred_mean = torch.mean(pred_valid)
        target_mean = torch.mean(target_valid)
        
        numerator = torch.sum((pred_valid - pred_mean) * (target_valid - target_mean))
        denominator = torch.sqrt(
            torch.sum((pred_valid - pred_mean) ** 2) * 
            torch.sum((target_valid - target_mean) ** 2)
        )
        
        correlation = numerator / (denominator + 1e-8)
        
        # Scale to [0, 1] range
        kpi = (correlation + 1) / 2
        
        return kpi


class PhysicsConstrainedLoss(nn.Module):
    """Loss function with physics constraints"""
    
    def __init__(self, base_loss: nn.Module, 
                 water_balance_weight: float = 0.1,
                 positivity_weight: float = 0.05):
        """
        Initialize physics-constrained loss
        
        Args:
            base_loss: Base loss function
            water_balance_weight: Weight for water balance constraint
            positivity_weight: Weight for positivity constraint
        """
        super().__init__()
        self.base_loss = base_loss
        self.water_balance_weight = water_balance_weight
        self.positivity_weight = positivity_weight
    
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                precipitation: Optional[torch.Tensor] = None,
                task_names: List[str] = None) -> torch.Tensor:
        """
        Compute loss with physics constraints
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            precipitation: Precipitation data for water balance
            task_names: List of task names
            
        Returns:
            Total loss with physics constraints
        """
        # Base multi-task loss
        total_loss, task_losses = self.base_loss(predictions, targets, task_names)
        
        # Add physics constraints
        physics_loss = 0.0
        
        # 1. Non-negativity constraint for streamflow and ET
        if 'streamflow' in predictions or 'usgsFlow' in predictions:
            flow_key = 'streamflow' if 'streamflow' in predictions else 'usgsFlow'
            flow_pred = predictions[flow_key]
            
            if isinstance(flow_pred, dict):
                flow_values = flow_pred.get('mu', flow_pred.get('y_hat', None))
            else:
                flow_values = flow_pred
            
            if flow_values is not None:
                # Penalize negative predictions
                negativity_penalty = torch.sum(F.relu(-flow_values))
                physics_loss += self.positivity_weight * negativity_penalty
        
        if 'et' in predictions or 'ET' in predictions:
            et_key = 'et' if 'et' in predictions else 'ET'
            et_pred = predictions[et_key]
            
            if isinstance(et_pred, dict):
                et_values = et_pred.get('mu', et_pred.get('y_hat', None))
            else:
                et_values = et_pred
            
            if et_values is not None:
                # Penalize negative ET predictions
                negativity_penalty = torch.sum(F.relu(-et_values))
                physics_loss += self.positivity_weight * negativity_penalty
        
        # 2. Water balance constraint
        if (self.water_balance_weight > 0 and precipitation is not None and
            ('streamflow' in predictions or 'usgsFlow' in predictions) and
            ('et' in predictions or 'ET' in predictions)):
            
            # Get streamflow and ET predictions
            flow_key = 'streamflow' if 'streamflow' in predictions else 'usgsFlow'
            et_key = 'et' if 'et' in predictions else 'ET'
            
            flow_pred = predictions[flow_key]
            et_pred = predictions[et_key]
            
            if isinstance(flow_pred, dict):
                flow_values = flow_pred.get('mu', flow_pred.get('y_hat', None))
            else:
                flow_values = flow_pred
            
            if isinstance(et_pred, dict):
                et_values = et_pred.get('mu', et_pred.get('y_hat', None))
            else:
                et_values = et_pred
            
            if flow_values is not None and et_values is not None:
                # Simplified water balance: P - Q - ET ≈ 0
                water_imbalance = precipitation - flow_values - et_values
                water_balance_penalty = torch.mean(water_imbalance ** 2)
                physics_loss += self.water_balance_weight * water_balance_penalty
        
        # Combine base loss with physics constraints
        total_loss = total_loss + physics_loss
        
        return total_loss


def get_loss_function(loss_config: Dict, task_names: List[str]) -> nn.Module:
    """
    Factory function to create loss function based on configuration
    
    Args:
        loss_config: Loss configuration dictionary
        task_names: List of task names
        
    Returns:
        Loss function module
    """
    base_loss_name = loss_config.get('base_loss', 'mse')
    balancing_method = loss_config.get('multi_task_balancing', 'fixed')
    
    # Create base loss functions for each task
    task_losses = []
    for task_name in task_names:
        if base_loss_name == 'nse':
            task_losses.append(NSELoss())
        elif base_loss_name == 'kge':
            task_losses.append(KGELoss())
        elif base_loss_name == 'mse':
            task_losses.append(MSELoss())
        else:
            raise ValueError(f"Unknown base loss: {base_loss_name}")
    
    # Create multi-task loss
    if balancing_method == 'fixed':
        initial_weights = loss_config.get('loss_weights', None)
        multi_task_loss = MultiTaskLoss(
            task_losses, 
            balancing_method='fixed',
            initial_weights=initial_weights
        )
    elif balancing_method == 'uncertainty':
        multi_task_loss = MultiTaskLoss(
            task_losses,
            balancing_method='uncertainty'
        )
    elif balancing_method == 'dynamic':
        multi_task_loss = MultiTaskLoss(
            task_losses,
            balancing_method='dynamic'
        )
    else:
        raise ValueError(f"Unknown balancing method: {balancing_method}")
    
    # Add physics constraints if enabled
    if loss_config.get('water_balance_weight', 0) > 0:
        physics_loss = PhysicsConstrainedLoss(
            base_loss=multi_task_loss,
            water_balance_weight=loss_config.get('water_balance_weight', 0.1),
            positivity_weight=loss_config.get('positivity_weight', 0.05)
        )
        return physics_loss
    
    return multi_task_loss