"""
Customized Gate Control (CGC) Layer for Multi-Task Learning
Core component of the HydroMTL_CGC architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Tuple, Any

class AttentionGate(nn.Module):
    """Attention-based gating mechanism for expert selection"""
    
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 64):
        """
        Initialize attention gate
        
        Args:
            input_dim: Dimension of input features
            num_experts: Number of experts to route to
            hidden_dim: Hidden dimension for attention network
        """
        super().__init__()
        self.num_experts = num_experts
        
        # Attention network
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights for experts
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Attention weights [batch_size, num_experts]
        """
        return self.attention_net(x)

class ExpertNetwork(nn.Module):
    """Individual expert network in CGC layer"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.2):
        """
        Initialize expert network
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert network
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Expert output [batch_size, output_dim]
        """
        return self.net(x)

class CGCLayer(nn.Module):
    """
    Customized Gate Control Layer for Multi-Task Learning
    
    Implements the CGC architecture with shared experts and task-specific experts,
    using attention-based gating to dynamically route information.
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 num_shared_experts: int, num_task_experts: List[int],
                 use_attention_gate: bool = True, dropout_rate: float = 0.2):
        """
        Initialize CGC layer
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output features for each task
            num_shared_experts: Number of shared experts
            num_task_experts: List of task-specific expert counts for each task
            use_attention_gate: Whether to use attention-based gating
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts
        self.num_tasks = len(num_task_experts)
        self.use_attention_gate = use_attention_gate
        
        # Calculate total experts
        self.total_experts = num_shared_experts + sum(num_task_experts)
        
        # Create expert networks
        self.shared_experts = nn.ModuleList([
            ExpertNetwork(input_dim, output_dim, hidden_dim=128, dropout=dropout_rate)
            for _ in range(num_shared_experts)
        ])
        
        # Create task-specific experts
        self.task_experts = nn.ModuleList()
        self.task_expert_offsets = [num_shared_experts]  # Start index for each task's experts
        
        for i, num_experts in enumerate(num_task_experts):
            task_expert_list = nn.ModuleList([
                ExpertNetwork(input_dim, output_dim, hidden_dim=128, dropout=dropout_rate)
                for _ in range(num_experts)
            ])
            self.task_experts.append(task_expert_list)
            
            # Update offset for next task
            if i < self.num_tasks - 1:
                self.task_expert_offsets.append(
                    self.task_expert_offsets[-1] + num_experts
                )
        
        # Create gating networks
        if use_attention_gate:
            self.gates = nn.ModuleList([
                AttentionGate(input_dim, self.total_experts, hidden_dim=64)
                for _ in range(self.num_tasks)
            ])
        else:
            # Simple linear gates
            self.gates = nn.ModuleList([
                nn.Linear(input_dim, self.total_experts)
                for _ in range(self.num_tasks)
            ])
        
        # Create gate masks for each task
        self.register_buffer('gate_masks', self._create_gate_masks())
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
    def _create_gate_masks(self) -> torch.Tensor:
        """
        Create binary masks for each task's gate
        
        Returns:
            Gate masks tensor [num_tasks, total_experts]
        """
        masks = torch.zeros(self.num_tasks, self.total_experts)
        
        # Shared experts are accessible to all tasks
        masks[:, :self.num_shared_experts] = 1
        
        # Task-specific experts are only accessible to their respective tasks
        for i in range(self.num_tasks):
            start_idx = self.task_expert_offsets[i]
            end_idx = start_idx + self.num_task_experts[i]
            masks[i, start_idx:end_idx] = 1
        
        return masks
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through CGC layer
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            List of task-specific outputs, each [batch_size, output_dim]
        """
        batch_size = x.shape[0]
        
        # Compute outputs from all experts
        all_expert_outputs = []
        
        # Shared experts
        for expert in self.shared_experts:
            expert_out = expert(x)
            all_expert_outputs.append(expert_out)
        
        # Task-specific experts
        for task_idx in range(self.num_tasks):
            for expert in self.task_experts[task_idx]:
                expert_out = expert(x)
                all_expert_outputs.append(expert_out)
        
        # Stack expert outputs [batch_size, total_experts, output_dim]
        expert_stack = torch.stack(all_expert_outputs, dim=1)
        
        # Compute task-specific outputs
        task_outputs = []
        
        for task_idx in range(self.num_tasks):
            # Compute gate weights
            if self.use_attention_gate:
                gate_logits = self.gates[task_idx](x)
            else:
                gate_logits = self.gates[task_idx](x)
            
            # Apply mask to restrict expert access
            gate_logits = gate_logits * self.gate_masks[task_idx]
            
            # Apply masked softmax
            # Set masked-out experts to very negative value for softmax
            gate_logits = gate_logits - 1e9 * (1 - self.gate_masks[task_idx])
            gate_weights = F.softmax(gate_logits, dim=-1)
            
            # Expand weights for broadcasting [batch_size, total_experts, 1]
            gate_weights_expanded = gate_weights.unsqueeze(-1)
            
            # Weighted combination of experts
            weighted_experts = expert_stack * gate_weights_expanded
            task_output = weighted_experts.sum(dim=1)
            
            # Apply dropout
            task_output = self.dropout(task_output)
            
            task_outputs.append(task_output)
        
        return task_outputs
    
    def get_gate_analysis(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze gate behavior for interpretability
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary with gate analysis metrics
        """
        batch_size = x.shape[0]
        analysis = {}
        
        # Compute gate weights for all tasks
        all_gate_weights = []
        
        for task_idx in range(self.num_tasks):
            if self.use_attention_gate:
                gate_weights = self.gates[task_idx](x)
            else:
                gate_weights = F.softmax(self.gates[task_idx](x), dim=-1)
            
            # Apply mask
            gate_weights = gate_weights * self.gate_masks[task_idx]
            gate_weights = gate_weights / (gate_weights.sum(dim=-1, keepdim=True) + 1e-8)
            
            all_gate_weights.append(gate_weights)
        
        gate_weights_stack = torch.stack(all_gate_weights, dim=1)
        
        # Store analysis metrics
        analysis['gate_weights'] = gate_weights_stack  # [batch, tasks, experts]
        
        # Compute expert utilization
        expert_utilization = gate_weights_stack.mean(dim=(0, 1))
        analysis['expert_utilization'] = expert_utilization
        
        # Compute task-specific expert importance
        task_expert_importance = []
        for task_idx in range(self.num_tasks):
            task_weights = gate_weights_stack[:, task_idx, :]
            task_importance = task_weights.mean(dim=0)
            task_expert_importance.append(task_importance)
        
        analysis['task_expert_importance'] = torch.stack(task_expert_importance)
        
        return analysis