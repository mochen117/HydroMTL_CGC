# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Model-agnostic feature engineering scaler.
# Enforces strictly positive physical domains and dimensionless runoff ratios.
# ==============================================================================

import numpy as np
from typing import Tuple, Dict

class HydroScaler:
    def __init__(self, config: Dict):
        self.config = config
        self.stat_dict = {}
        
        data_cfg = config.get('data', config)
        self.static_features = data_cfg.get('static_features',[])
        
        try:
            self.area_idx = self.static_features.index('area_gages2')
        except ValueError:
            self.area_idx = -1
        
        try:
            self.prcp_idx = self.static_features.index('p_mean')
        except ValueError:
            self.prcp_idx = -1

        self.targets_cfg = data_cfg.get('targets', [])
        self.task_names = [str(t['name']).lower() for t in self.targets_cfg]
        self.q_name = next((t for t in self.task_names if 'streamflow' in t), 'streamflow')
        
        # CFS to mm/day conversion base
        self.conv_factor = (0.0283168 * 86400) / 1000.0

    def fit_transform(self, dyn_x: np.ndarray, stat_num: np.ndarray, stat_cat: np.ndarray, target_dict: Dict[str, np.ndarray]) -> Tuple:
        self.stat_dict['s_num_mean'] = np.nanmean(stat_num, axis=0)
        self.stat_dict['s_num_std'] = np.nanstd(stat_num, axis=0)
        self.stat_dict['s_num_std'][self.stat_dict['s_num_std'] < 1e-6] = 1.0
        
        s_num_t = (stat_num - self.stat_dict['s_num_mean']) / self.stat_dict['s_num_std']
        s_num_t = np.nan_to_num(s_num_t, nan=0.0, posinf=0.0, neginf=0.0)

        self.stat_dict['dyn_mean'] = np.nanmean(dyn_x, axis=(0, 1))
        self.stat_dict['dyn_std'] = np.nanstd(dyn_x, axis=(0, 1))
        self.stat_dict['dyn_std'][self.stat_dict['dyn_std'] < 1e-6] = 1.0
        
        dyn_t = (dyn_x - self.stat_dict['dyn_mean']) / self.stat_dict['dyn_std']
        dyn_t = np.nan_to_num(dyn_t, nan=0.0, posinf=0.0, neginf=0.0)

        target_t = {}
        
        if self.area_idx != -1 and stat_num.shape[1] > self.area_idx:
            area = np.nan_to_num(stat_num[:, self.area_idx], nan=1.0)
        else:
            area = np.ones(stat_num.shape[0])
            
        if self.prcp_idx != -1 and stat_num.shape[1] > self.prcp_idx:
            prcp = np.nan_to_num(stat_num[:, self.prcp_idx], nan=1.0)
        else:
            prcp = np.ones(stat_num.shape[0])
            
        area = np.maximum(area, 1e-2)
        prcp = np.maximum(prcp, 1e-2)
        
        for task in self.task_names:
            raw_y = target_dict[task]
            if task == self.q_name:
                raw_y_safe = np.maximum(raw_y, 0.0)
                q_norm = self._physical_basin_norm(raw_y_safe, area, prcp, to_norm=True)
                q_norm = np.maximum(q_norm, 0.0)
                q_log = np.log10(np.sqrt(q_norm) + 0.1)
                
                mean_val = np.nanmean(q_log)
                std_val = np.nanstd(q_log)
                
                if np.isnan(std_val) or std_val < 1e-6: 
                    std_val = 1.0
                if np.isnan(mean_val): 
                    mean_val = 0.0
                
                self.stat_dict[f'{task}_mean'] = mean_val
                self.stat_dict[f'{task}_std'] = std_val
                target_t[task] = (q_log - mean_val) / std_val
            else:
                mean_val = np.nanmean(raw_y)
                std_val = np.nanstd(raw_y)
                
                if np.isnan(std_val) or std_val < 1e-6: 
                    std_val = 1.0
                if np.isnan(mean_val): 
                    mean_val = 0.0
                
                self.stat_dict[f'{task}_mean'] = mean_val
                self.stat_dict[f'{task}_std'] = std_val
                target_t[task] = (raw_y - mean_val) / std_val

        return dyn_t, s_num_t, np.copy(stat_cat) if stat_cat is not None else None, target_t

    def transform(self, dyn_x: np.ndarray, stat_num: np.ndarray, stat_cat: np.ndarray, target_dict: Dict[str, np.ndarray]) -> Tuple:
        if not self.stat_dict:
            raise RuntimeError("Scaler must be fitted before calling transform.")

        s_num_t = (stat_num - self.stat_dict['s_num_mean']) / self.stat_dict['s_num_std']
        s_num_t = np.nan_to_num(s_num_t, nan=0.0, posinf=0.0, neginf=0.0)
        
        dyn_t = (dyn_x - self.stat_dict['dyn_mean']) / self.stat_dict['dyn_std']
        dyn_t = np.nan_to_num(dyn_t, nan=0.0, posinf=0.0, neginf=0.0)
        
        target_t = {}
        
        if self.area_idx != -1 and stat_num.shape[1] > self.area_idx:
            area = np.nan_to_num(stat_num[:, self.area_idx], nan=1.0)
        else:
            area = np.ones(stat_num.shape[0])
            
        if self.prcp_idx != -1 and stat_num.shape[1] > self.prcp_idx:
            prcp = np.nan_to_num(stat_num[:, self.prcp_idx], nan=1.0)
        else:
            prcp = np.ones(stat_num.shape[0])
            
        area = np.maximum(area, 1e-2)
        prcp = np.maximum(prcp, 1e-2)

        for task in self.task_names:
            raw_y = target_dict.get(task)
            if raw_y is None: 
                continue
            
            if task == self.q_name:
                raw_y_safe = np.maximum(raw_y, 0.0)
                q_norm = self._physical_basin_norm(raw_y_safe, area, prcp, to_norm=True)
                q_norm = np.maximum(q_norm, 0.0)
                q_log = np.log10(np.sqrt(q_norm) + 0.1)
                target_t[task] = (q_log - self.stat_dict[f'{task}_mean']) / self.stat_dict[f'{task}_std']
            else:
                target_t[task] = (raw_y - self.stat_dict[f'{task}_mean']) / self.stat_dict[f'{task}_std']

        return dyn_t, s_num_t, np.copy(stat_cat) if stat_cat is not None else None, target_t

    def inverse_transform_target(self, target_latent_dict: Dict[str, np.ndarray], stat_num_scaled: np.ndarray) -> Dict[str, np.ndarray]:
        stat_num_raw = stat_num_scaled * self.stat_dict['s_num_std'] + self.stat_dict['s_num_mean']
        
        if self.prcp_idx != -1 and stat_num_raw.shape[1] > self.prcp_idx:
            prcp_raw = np.nan_to_num(stat_num_raw[:, self.prcp_idx], nan=1.0)
        else:
            prcp_raw = np.ones(stat_num_raw.shape[0])
            
        prcp_raw = np.maximum(prcp_raw, 1e-2)
        
        target_phys = {}
        for task, latent_arr in target_latent_dict.items():
            if task == self.q_name:
                q_log = latent_arr * self.stat_dict[f'{task}_std'] + self.stat_dict[f'{task}_mean']
                q_log = np.clip(q_log, -5.0, 10.0)
                
                q_norm = (np.power(10, q_log) - 0.1) ** 2
                phys_val_mm_day = q_norm * prcp_raw
                target_phys[task] = np.maximum(phys_val_mm_day, 0.0)
            else:
                target_phys[task] = latent_arr * self.stat_dict[f'{task}_std'] + self.stat_dict[f'{task}_mean']
                
        return target_phys

    def _physical_basin_norm(self, flow: np.ndarray, area: np.ndarray, prcp: np.ndarray, to_norm: bool) -> np.ndarray:
        area_ex = np.expand_dims(area, 1) if flow.ndim == 2 else area
        prcp_ex = np.expand_dims(prcp, 1) if flow.ndim == 2 else prcp
        
        if to_norm:
            return (flow * self.conv_factor) / (area_ex * prcp_ex)
        else:
            return (flow * (area_ex * prcp_ex)) / self.conv_factor