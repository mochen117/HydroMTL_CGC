# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Torch Dataset wrapper for Hydrological time-series data.
# Yields robust numpy dictionaries to prevent multiprocessing memory leaks.
# ==============================================================================

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, Any

from .data_scalers import HydroScaler

class HydroDataset(Dataset):
    def __init__(self, raw_data: Dict[str, Any], data_params: Dict[str, Any], mode: str = 'train', scaler: HydroScaler = None):
        self.mode = mode
        self.rho = data_params.get('sequence_length', 365)
        self.task_names = [str(t['name']).lower() for t in data_params.get('targets',[])]
        
        self.s_cat = raw_data.get('s_cat', None)
        
        if mode == 'train':
            self.scaler = HydroScaler(data_params)
            self.dyn, self.s_num, _, self.y_dict = self.scaler.fit_transform(
                raw_data['dyn'], raw_data['s_num'], raw_data['s_cat'], raw_data['y_dict']
            )
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided for non-training sets to prevent data leakage.")
            self.scaler = scaler
            self.dyn, self.s_num, _, self.y_dict = self.scaler.transform(
                raw_data['dyn'], raw_data['s_num'], raw_data['s_cat'], raw_data['y_dict']
            )

        self._process_targets()
        self.samples =[(b, t) for b in range(self.dyn.shape[0]) for t in range(self.dyn.shape[1] - self.rho + 1)]

    def _process_targets(self):
        """
        Interpolate short gaps (sensor glitches <= 3 days).
        Long gaps remain NaN to be ignored by Masked Loss.
        """
        for task in self.task_names:
            if task not in self.y_dict: 
                continue
            for b in range(self.y_dict[task].shape[0]):
                df = pd.DataFrame(self.y_dict[task][b])
                interpolated = df.interpolate(method='linear', limit=3, limit_direction='both')
                self.y_dict[task][b] = interpolated.values.squeeze()

    def __len__(self) -> int: 
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        b, t = self.samples[idx]
        
        out_dict = {
            'features': self.dyn[b, t:t+self.rho, :].astype(np.float32),
            'static_num': self.s_num[b].astype(np.float32),
            'basin_idx': np.array(b, dtype=np.int64)
        }
        
        if self.s_cat is not None and len(self.s_cat) > 0:
            out_dict['categorical_features'] = self.s_cat[b].astype(np.int64)
            
        for task in self.task_names:
            if task in self.y_dict:
                out_dict[task] = np.array([self.y_dict[task][b, t+self.rho-1]], dtype=np.float32)
                
        return out_dict