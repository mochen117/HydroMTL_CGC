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
    Supports metrics: 'nse', 'kge', 'rmse', 'mae', 'correlation', 'bias'

    Args:
        predictions: Dictionary of model predictions for each task.
                     Can be tensor or dict with 'mu' (mean) for probabilistic outputs.
        targets: Dictionary of ground truth values for each task.
        metric_list: List of metric names to compute. If None, compute all default metrics.

    Returns:
        Dictionary containing computed metrics with keys '{task_name}_{metric_name}'.
    """

    if metric_list is None:
        metric_list = ['nse', 'kge', 'rmse', 'mae', 'bias']

    all_metrics = {}

    # Helper function to extract tensor from prediction (handles probabilistic outputs)
    def _extract_prediction(pred):
        if isinstance(pred, dict):
            # For probabilistic outputs (e.g., GMM), use the mean
            if 'mu' in pred:
                return pred['mu']
            elif 'means' in pred:
                # For GMM with multiple components, take weighted mean
                means = pred['means']
                weights = pred['weights']
                if len(weights.shape) == 2:  # [batch, n_components]
                    weights = weights.unsqueeze(-1)  # [batch, n_components, 1]
                weighted_mean = torch.sum(means * weights, dim=1)
                return weighted_mean
            else:
                # Fallback to first tensor value
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
            et_metrics = _calculate_task_metrics(et_pred, et_target, metric_list, 'et')
            all_metrics.update(et_metrics)

    # Calculate combined metrics across all tasks
    combined_preds = []
    combined_targets = []
    task_names = []

    for task_name in predictions.keys():
        if task_name in targets:
            pred = _extract_prediction(predictions[task_name])
            target = targets[task_name]
            if pred is not None:
                combined_preds.append(pred.flatten())
                combined_targets.append(target.flatten())
                task_names.append(task_name)

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
    """
    Calculate metrics for a single hydrological task.

    Args:
        predictions: Predicted values tensor.
        targets: Ground truth values tensor.
        metric_list: List of metric names to compute.
        task_name: Name of the task (for metric naming).

    Returns:
        Dictionary of metrics for the task.
    """
    metrics = {}

    # Flatten both tensors to 1D
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    # Ensure same length (truncate to shorter if necessary)
    min_len = min(len(pred_flat), len(target_flat))
    if min_len == 0:
        # No data, return NaN
        for metric_name in metric_list:
            metrics[f'{task_name}_{metric_name}'] = float('nan')
        return metrics

    pred_flat = pred_flat[:min_len]
    target_flat = target_flat[:min_len]

    # Handle NaN values
    mask = ~torch.isnan(target_flat)
    if mask.sum() < MIN_SAMPLES:
        # Too few valid samples to compute reliable metrics
        for metric_name in metric_list:
            metrics[f'{task_name}_{metric_name}'] = float('nan')
        return metrics

    pred_valid = pred_flat[mask]
    target_valid = target_flat[mask]

    # Calculate each requested metric
    for metric_name in metric_list:
        metric_func = METRIC_FUNCTIONS.get(metric_name.lower())
        if metric_func is None:
            continue

        metric_value = metric_func(pred_valid, target_valid)

        # Convert to Python float if it's a tensor
        if torch.is_tensor(metric_value):
            metric_value = metric_value.item()

        # Clip extremely negative NSE/R2 to a reasonable lower bound (optional)
        if metric_name.lower() in ['nse', 'r2'] and metric_value < -1000:
            metric_value = float('nan')

        metrics[f'{task_name}_{metric_name}'] = metric_value

    return metrics


def _calculate_global_metrics(predictions: torch.Tensor,
                              targets: torch.Tensor,
                              metric_list: List[str]) -> Dict[str, float]:
    """
    Calculate global metrics across all tasks.

    Args:
        predictions: Concatenated predictions from all tasks.
        targets: Concatenated targets from all tasks.
        metric_list: List of metric names to compute.

    Returns:
        Dictionary of global metrics.
    """
    metrics = {}

    # Handle NaN values
    mask = ~torch.isnan(targets)
    if mask.sum() < MIN_SAMPLES:
        # Return NaN-filled dictionary when insufficient samples
        for metric_name in metric_list:
            metrics[f'global_{metric_name}'] = float('nan')
        return metrics

    pred_valid = predictions[mask]
    target_valid = targets[mask]

    # Calculate global versions of metrics
    for metric_name in metric_list:
        metric_func = METRIC_FUNCTIONS.get(metric_name.lower())
        if metric_func is None:
            continue

        metric_value = metric_func(pred_valid, target_valid)

        if torch.is_tensor(metric_value):
            metric_value = metric_value.item()

        # Clip extremely negative NSE/R2 to NaN
        if metric_name.lower() in ['nse', 'r2'] and metric_value < -1000:
            metric_value = float('nan')

        metrics[f'global_{metric_name}'] = metric_value

    return metrics


def _calculate_nse(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Calculate Nash-Sutcliffe Efficiency."""
    target_mean = torch.mean(targets)
    numerator = torch.sum((targets - predictions) ** 2)
    denominator = torch.sum((targets - target_mean) ** 2)
    nse = 1 - numerator / (denominator + epsilon)
    return nse


def _calculate_kge(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Calculate Kling-Gupta Efficiency."""
    pred_mean = torch.mean(predictions)
    target_mean = torch.mean(targets)

    pred_centered = predictions - pred_mean
    target_centered = targets - target_mean

    covariance = torch.mean(pred_centered * target_centered)
    pred_std = torch.std(predictions, unbiased=False)
    target_std = torch.std(targets, unbiased=False)

    r = covariance / (pred_std * target_std + epsilon)

    alpha = pred_std / (target_std + epsilon)
    beta = pred_mean / (target_mean + epsilon)

    kge = 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge


def _calculate_rmse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculate Root Mean Square Error."""
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)
    return rmse


def _calculate_mae(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculate Mean Absolute Error."""
    mae = torch.mean(torch.abs(predictions - targets))
    return mae


def _calculate_correlation(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Calculate Pearson correlation coefficient."""
    pred_mean = torch.mean(predictions)
    target_mean = torch.mean(targets)

    numerator = torch.sum((predictions - pred_mean) * (targets - target_mean))
    denominator = torch.sqrt(
        torch.sum((predictions - pred_mean) ** 2) *
        torch.sum((targets - target_mean) ** 2)
    )

    correlation = numerator / (denominator + epsilon)
    return correlation


def _calculate_bias(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculate bias (mean error)."""
    bias = torch.mean(predictions - targets)
    return bias


def _calculate_r2(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Calculate R-squared (coefficient of determination)."""
    target_mean = torch.mean(targets)
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - target_mean) ** 2)
    r2 = 1 - ss_res / (ss_tot + epsilon)
    return r2


# Dictionary mapping metric names to their calculation functions
METRIC_FUNCTIONS = {
    'nse': _calculate_nse,
    'kge': _calculate_kge,
    'rmse': _calculate_rmse,
    'mae': _calculate_mae,
    'correlation': _calculate_correlation,
    'bias': _calculate_bias,
    'r2': _calculate_r2,
}