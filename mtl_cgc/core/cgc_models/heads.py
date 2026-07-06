# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Output heads for regression tasks.
# ==============================================================================

import torch
import torch.nn as nn

class RegressionHead(nn.Module):
    """Linear output head for deterministic target estimation."""
    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def get_head(head_type: str, in_features: int, out_features: int = 1) -> nn.Module:
    """Factory helper to fetch output modules."""
    if head_type.lower() == 'regression': 
        return RegressionHead(in_features, out_features)
    return RegressionHead(in_features, out_features)