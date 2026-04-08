import torch
import torch.nn as nn
from typing import Dict

class HardSharingLSTM(nn.Module):
    """
    Hard parameter sharing LSTM for multi-task hydrological modeling.
    Shared LSTM encoder + task-specific linear heads.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.streamflow_head = nn.Linear(hidden_dim, 1)
        self.et_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        last_hidden = self.dropout(last_hidden)

        streamflow = self.streamflow_head(last_hidden)
        evapotranspiration = self.et_head(last_hidden)

        return {
            'streamflow': streamflow,
            'evapotranspiration': evapotranspiration
        }