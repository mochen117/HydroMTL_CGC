# ==============================================================================
# Copyright (c) 2024-2025. All Rights Reserved.
# Author: Mochen & Project Contributors
# Description: Customized Gate Control (CGC) Layer for robust MTL.
# Equipped with extraction mechanisms for interpretability mapping.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

class ExpertNetwork(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, drop: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CGCLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_shared: int, n_task_list: List[int], drop: float = 0.5):
        super().__init__()
        self.num_tasks = len(n_task_list)
        self.n_shared = n_shared
        self.n_task_list = n_task_list

        self.shared_experts = nn.ModuleList([ExpertNetwork(in_dim, out_dim, drop) for _ in range(n_shared)])
        
        self.task_experts = nn.ModuleList()
        self.task_gates = nn.ModuleList()

        for num_experts in n_task_list:
            self.task_experts.append(nn.ModuleList([ExpertNetwork(in_dim, out_dim, drop) for _ in range(num_experts)]))
            self.task_gates.append(nn.Linear(in_dim, n_shared + num_experts))

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        shared_outs =[expert(x).unsqueeze(1) for expert in self.shared_experts]
        
        task_reps =[]
        gate_weights_dict = {}

        for i in range(self.num_tasks):
            t_experts = self.task_experts[i]
            t_gate = self.task_gates[i]

            t_outs =[expert(x).unsqueeze(1) for expert in t_experts]
            all_outs = torch.cat(shared_outs + t_outs, dim=1) 

            gate_scores = t_gate(x) 
            gate_weights = F.softmax(gate_scores, dim=-1) 
            
            # Extract routing probabilities
            gate_weights_dict[f'task_{i}_gate'] = gate_weights.detach()

            gate_weights_expanded = gate_weights.unsqueeze(-1) 
            task_rep = torch.sum(all_outs * gate_weights_expanded, dim=1) 
            
            task_reps.append(task_rep)

        return task_reps, gate_weights_dict