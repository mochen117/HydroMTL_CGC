# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: Hydrological metrics library.
# Computes core indicators (NSE, KGE, RMSE, MAE, Bias, Corr) securely.
# ==============================================================================

import torch
import numpy as np

def compute_metrics(preds: dict, targets: dict, metrics_list: list) -> dict:
    """Computes requested metrics. Expects flattened, valid (no-NaN) tensors."""
    results = {}
    for task in preds.keys():
        if task not in targets: continue
        p = preds[task].numpy()
        t = targets[task].numpy()
        
        if len(p) == 0: continue
        
        mean_t = np.mean(t)
        mean_p = np.mean(p)
        std_t = np.std(t)
        std_p = np.std(p)
        
        for m in metrics_list:
            m_lower = m.lower()
            val = np.nan
            if m_lower == 'rmse':
                val = np.sqrt(np.mean((p - t)**2))
            elif m_lower == 'mae':
                val = np.mean(np.abs(p - t))
            elif m_lower == 'bias':
                val = np.mean(p - t)
            elif m_lower == 'corr':
                if std_p > 0 and std_t > 0:
                    val = np.corrcoef(p, t)[0, 1]
            elif m_lower == 'nse':
                denominator = np.sum((t - mean_t)**2)
                if denominator > 0:
                    val = 1 - (np.sum((p - t)**2) / denominator)
            elif m_lower == 'kge':
                if std_t > 0 and mean_t != 0:
                    r = np.corrcoef(p, t)[0, 1] if std_p > 0 else 0
                    alpha = std_p / std_t
                    beta = mean_p / mean_t
                    val = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
                    
            results[f"{task}_{m_lower}"] = float(val)
    return results