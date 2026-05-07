import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

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
        
        # Locate precipitation index
        self.prcp_idx = data_cfg.get('static_features', []).index('p_mean')
        
        phys_cfg = config.get('model', {}).get('physics_constraints', {}).get('water_balance', {})
        self.use_physics = phys_cfg.get('enabled', False)
        self.alpha = phys_cfg.get('alpha', 0.1)

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return F.mse_loss(pred[mask], target[mask])

    def forward(self, preds_dict: Dict[str, torch.Tensor], targets_dict: Dict[str, torch.Tensor], s_num: torch.Tensor = None):
        # 1. Base Math Loss (Latent Space)
        total_math_loss = 0.0
        for task, weight in self.weights.items():
            if weight > 0 and task in preds_dict and task in targets_dict:
                total_math_loss += weight * self._masked_mse(preds_dict[task], targets_dict[task])
                
        # 2. Physics Constraints (Physical Space)
        if self.use_physics and self.stat and self.q_name in preds_dict and self.et_name in preds_dict:
            # Revert to physical domain
            q_log = preds_dict[self.q_name].squeeze() * self.stat[f'{self.q_name}_std'] + self.stat[f'{self.q_name}_mean']
            q_phys = (torch.pow(10, q_log) - 0.1)**2
            et_phys = preds_dict[self.et_name].squeeze() * self.stat[f'{self.et_name}_std'] + self.stat[f'{self.et_name}_mean']
            
            p_phys = s_num[:, self.prcp_idx]
            valid_mask = ~torch.isnan(p_phys)
            
            if valid_mask.any():
                phys_loss = F.mse_loss(q_phys[valid_mask] + et_phys[valid_mask], p_phys[valid_mask])
                return (1 - self.alpha) * total_math_loss + self.alpha * phys_loss
                
        return total_math_loss