# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Custom water balance physical criterion.
# Enforces strictly non-negative observations and ignores missing timesteps.
# Evaluates losses strictly over flattened aligned 1D tensors to prevent
# silent broadcasting or shape-matching errors.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class DynamicMultiTaskLoss(nn.Module):
    """
    Computes masked mathematical MSE dynamically based on config weights.
    Applies physical Water Balance Loss if configured and both Q & ET exist.
    """
    def __init__(self, config: Dict, stat_dict: Dict):
        super().__init__()
        self.stat = stat_dict
        
        data_cfg = config.get('data', {})
        self.weights = {t['name']: float(t.get('loss_weight', 1.0)) for t in data_cfg.get('targets', [])}
        self.q_name = next((t['name'] for t in data_cfg.get('targets', []) if 'streamflow' in t['name'].lower()), 'streamflow')
        self.et_name = next((t['name'] for t in data_cfg.get('targets', []) if 'evapo' in t['name'].lower()), 'evapotranspiration')
        
        self.prcp_idx = data_cfg.get('static_features', []).index('p_mean') if 'p_mean' in data_cfg.get('static_features', []) else -1
        
        phys_cfg = config.get('model', {}).get('physics_constraints', {}).get('water_balance', {})
        self.use_physics = phys_cfg.get('enabled', False)
        self.alpha = phys_cfg.get('alpha', 0.1)

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Helper to compute MSE excluding missing observations."""
        mask = torch.isfinite(target) & torch.isfinite(pred)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return F.mse_loss(pred[mask], target[mask])

    def forward(self, preds_dict: Dict[str, torch.Tensor], targets_dict: Dict[str, torch.Tensor], s_num: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Base Math Loss (Latent Space)
        total_math_loss = torch.tensor(0.0, device=next(iter(preds_dict.values())).device)
        for task, weight in self.weights.items():
            if weight > 0 and task in preds_dict and task in targets_dict:
                p = preds_dict[task].reshape(-1)
                t = targets_dict[task].reshape(-1)
                
                # Enforce strict length assertion before loss calculation
                if p.shape[0] != t.shape[0]:
                    raise ValueError(
                        f"Shape mismatch in DynamicMultiTaskLoss: pred length {p.shape[0]} "
                        f"does not match target length {t.shape[0]}."
                    )
                total_math_loss += weight * self._masked_mse(p, t)
                
        # 2. Physics Constraints (Physical Space conversion)
        if self.use_physics and self.stat and self.q_name in preds_dict and self.et_name in preds_dict and s_num is not None and self.prcp_idx != -1:
            q_std = self.stat.get(f'{self.q_name}_std', 1.0)
            q_mean = self.stat.get(f'{self.q_name}_mean', 0.0)
            et_std = self.stat.get(f'{self.et_name}_std', 1.0)
            et_mean = self.stat.get(f'{self.et_name}_mean', 0.0)

            # Revert to physical domain with strict shape alignments
            q_log = preds_dict[self.q_name].reshape(-1) * q_std + q_mean
            q_log = torch.clamp(q_log, -5.0, 10.0)
            q_phys = (torch.pow(10, q_log) - 0.1)**2
            
            et_phys = preds_dict[self.et_name].reshape(-1) * et_std + et_mean
            p_phys = s_num[:, self.prcp_idx].reshape(-1)
            
            # Enforce strict alignment assertions for physics calculations
            if q_phys.shape[0] != p_phys.shape[0] or et_phys.shape[0] != p_phys.shape[0]:
                raise ValueError(
                    f"Physical shape mismatch: Q ({q_phys.shape[0]}), "
                    f"ET ({et_phys.shape[0]}), P ({p_phys.shape[0]}) must have identical batch dimensions."
                )
            
            valid_mask = torch.isfinite(p_phys) & torch.isfinite(q_phys) & torch.isfinite(et_phys)
            
            if valid_mask.any():
                phys_loss = F.mse_loss(q_phys[valid_mask] + et_phys[valid_mask], p_phys[valid_mask])
                return (1 - self.alpha) * total_math_loss + self.alpha * phys_loss
                
        return total_math_loss