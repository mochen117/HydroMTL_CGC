import torch
import torch.nn as nn
import torch.nn.functional as F

class RegressionHead(nn.Module):
    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
    def forward(self, x): return self.fc(x)

def get_head(head_type: str, in_features: int, out_features: int = 1) -> nn.Module:
    if head_type.lower() == 'regression': return RegressionHead(in_features, out_features)
    # Add GMM or CMAL here
    return RegressionHead(in_features, out_features)