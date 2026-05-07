import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import torch.nn.functional as F

# Minimum number of valid samples required to compute metrics
MIN_SAMPLES = 10


def compute_metrics(predictions: Dict[str, Union[torch.Tensor, Dict]],
                    targets: Dict[str, torch.Tensor],
                    metric_list: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Compute evaluation metrics for multi-task hydrological predictions.
    Supports tasks: 'streamflow', 'evapotranspiration'
    Supports metrics: 'nse', 'kge', 'rmse', 'mae', 'corr', 'bias'
    """
    if metric_list is None:
        metric_list = ['nse', 'kge', 'rmse', 'mae', 'bias', 'corr']

    all_metrics = {}

    def _extract_prediction(pred):
        if isinstance(pred, dict):
            if 'mu' in pred:
                return pred['mu']
            elif 'means' in pred:
                means = pred['means']
                weights = pred['weights']
                if len(weights.shape) == 2:
                    weights = weights.unsqueeze(-1)
                return torch.sum(means * weights, dim=1)
            else:
                for key, value in pred.items():
                    if torch.is_tensor(value):
                        return value
                return None
        return pred

    # Process streamflow task
    streamflow_keys = ['streamflow', 'usgsFlow', 'Q', 'flow']
    streamflow_pred_key = next((k for k in streamflow_keys if k in predictions), None)
    streamflow_target_key = next((k for k in streamflow_keys if k in targets), None)

    if streamflow_pred_key and streamflow_target_key:
        streamflow_pred = _extract_prediction(predictions[streamflow_pred_key])
        streamflow_target = targets[streamflow_target_key]

        if streamflow_pred is not None and streamflow_target is not None:
            streamflow_metrics = _calculate_task_metrics(
                streamflow_pred, streamflow_target, metric_list, 'streamflow'
            )
            all_metrics.update(streamflow_metrics)

    # Process evapotranspiration task
    et_keys = ['et', 'evapotranspiration', 'ET', 'evap']
    et_pred_key = next((k for k in et_keys if k in predictions), None)
    et_target_key = next((k for k in et_keys if k in targets), None)

    if et_pred_key and et_target_key:
        et_pred = _extract_prediction(predictions[et_pred_key])
        et_target = targets[et_target_key]

        if et_pred is not None and et_target is not None:
            et_metrics = _calculate_task_metrics(et_pred, et_target, metric_list, 'evapotranspiration')
            all_metrics.update(et_metrics)

    # Calculate combined metrics across all tasks
    combined_preds = []
    combined_targets = []

    for task_name in predictions.keys():
        if task_name in targets:
            pred = _extract_prediction(predictions[task_name])
            target = targets[task_name]
            if pred is not None:
                combined_preds.append(pred.flatten())
                combined_targets.append(target.flatten())

    if len(combined_preds) > 1:
        combined_preds_tensor = torch.cat(combined_preds)
        combined_targets_tensor = torch.cat(combined_targets)

        global_metrics = _calculate_global_metrics(
            combined_preds_tensor, combined_targets_tensor, metric_list
        )
        all_metrics.update(global_metrics)

    return all_metrics


def _calculate_task_metrics(predictions: torch.Tensor,
                            targets: torch.Tensor,
                            metric_list: List[str],
                            task_name: str) -> Dict[str, float]:
    metrics = {}

    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    min_len = min(len(pred_flat), len(target_flat))
    if min_len == 0:
        for metric_name in metric_list:
            metrics[f'{task_name}_{metric_name}'] = float('nan')
        return metrics

    pred_flat = pred_flat[:min_len]
    target_flat = target_flat[:min_len]

    # STRENGTHENED MASK: Filter out NaNs and Infs from BOTH targets and predictions
    mask = ~torch.isnan(target_flat) & ~torch.isnan(pred_flat) & ~torch.isinf(pred_flat)
    
    if mask.sum() < MIN_SAMPLES:
        for metric_name in metric_list:
            metrics[f'{task_name}_{metric_name}'] = float('nan')
        return metrics

    pred_valid = pred_flat[mask]
    target_valid = target_flat[mask]

    for metric_name in metric_list:
        metric_func = METRIC_FUNCTIONS.get(metric_name.lower())
        if metric_func is None:
            continue

        metric_value = metric_func(pred_valid, target_valid)

        if torch.is_tensor(metric_value):
            metric_value = metric_value.item()

        # CRITICAL FIX: Clip extremely negative scores to a hard floor (-10.0) instead of NaN
        if metric_name.lower() in ['nse', 'r2', 'kge']:
            if metric_value < -10.0:
                metric_value = -10.0

        metrics[f'{task_name}_{metric_name}'] = metric_value

    return metrics


def _calculate_global_metrics(predictions: torch.Tensor,
                              targets: torch.Tensor,
                              metric_list: List[str]) -> Dict[str, float]:
    metrics = {}

    mask = ~torch.isnan(targets) & ~torch.isnan(predictions) & ~torch.isinf(predictions)
    
    if mask.sum() < MIN_SAMPLES:
        for metric_name in metric_list:
            metrics[f'global_{metric_name}'] = float('nan')
        return metrics

    pred_valid = predictions[mask]
    target_valid = targets[mask]

    for metric_name in metric_list:
        metric_func = METRIC_FUNCTIONS.get(metric_name.lower())
        if metric_func is None:
            continue

        metric_value = metric_func(pred_valid, target_valid)

        if torch.is_tensor(metric_value):
            metric_value = metric_value.item()

        if metric_name.lower() in ['nse', 'r2', 'kge']:
            if metric_value < -10.0:
                metric_value = -10.0

        metrics[f'global_{metric_name}'] = metric_value

    return metrics


def _calculate_nse(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    target_mean = torch.mean(targets)
    numerator = torch.sum((targets - predictions) ** 2)
    denominator = torch.sum((targets - target_mean) ** 2)
    
    # Safely handle zero variance targets
    if denominator < epsilon:
        return torch.tensor(-10.0, device=predictions.device)
        
    nse = 1 - numerator / denominator
    return nse


def _calculate_kge(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    pred_mean = torch.mean(predictions)
    target_mean = torch.mean(targets)

    pred_centered = predictions - pred_mean
    target_centered = targets - target_mean

    pred_std = torch.std(predictions, unbiased=False)
    target_std = torch.std(targets, unbiased=False)
    
    if target_std < epsilon or pred_std < epsilon:
         return torch.tensor(-10.0, device=predictions.device)

    covariance = torch.mean(pred_centered * target_centered)
    r = covariance / (pred_std * target_std)
    
    alpha = pred_std / target_std
    beta = pred_mean / (target_mean + epsilon)

    kge = 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge


def _calculate_rmse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((predictions - targets) ** 2)
    return torch.sqrt(mse)


def _calculate_mae(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(predictions - targets))


def _calculate_correlation(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    pred_mean = torch.mean(predictions)
    target_mean = torch.mean(targets)

    numerator = torch.sum((predictions - pred_mean) * (targets - target_mean))
    denominator = torch.sqrt(
        torch.sum((predictions - pred_mean) ** 2) *
        torch.sum((targets - target_mean) ** 2)
    )

    if denominator < epsilon:
        return torch.tensor(0.0, device=predictions.device)

    return numerator / denominator


def _calculate_bias(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.mean(predictions - targets)


METRIC_FUNCTIONS = {
    'nse': _calculate_nse,
    'kge': _calculate_kge,
    'rmse': _calculate_rmse,
    'mae': _calculate_mae,
    'correlation': _calculate_correlation,
    'corr': _calculate_correlation,
    'bias': _calculate_bias,
}