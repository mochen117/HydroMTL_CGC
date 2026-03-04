"""
Evaluation module for model performance assessment
Computes hydrological metrics and provides detailed analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
import torch
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)


class HydroEvaluator:
    """Evaluator for hydrological model performance"""
    
    def __init__(self, config: Dict[str, Any], save_dir: str = None):
        """
        Initialize evaluator
        
        Args:
            config: Evaluation configuration
            save_dir: Directory to save evaluation results
        """
        self.config = config
        self.metrics = config.get('metrics', ['nse', 'kge', 'rmse', 'mae', 'pbias', 'r2'])
        
        if save_dir:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None
        
        # Cache for evaluation results
        self.results_cache = {}
    
    def compute_all_metrics(self, predictions: Dict[str, np.ndarray],
                           targets: Dict[str, np.ndarray],
                           basin_ids: Optional[List[str]] = None,
                           save_results: bool = True) -> Dict[str, Any]:
        """
        Compute all configured metrics
        
        Args:
            predictions: Dictionary of predictions for each task
            targets: Dictionary of targets for each task
            basin_ids: List of basin IDs for per-basin evaluation
            save_results: Whether to save results to file
            
        Returns:
            Dictionary containing all computed metrics
        """
        results = {}
        
        for task_name in predictions.keys():
            # Skip non-task predictions (e.g., gate_analysis)
            if task_name in ['gate_analysis', 'water_imbalance']:
                continue
            
            # Get predictions and targets for this task
            pred_key = task_name
            target_key = task_name
            
            # Handle probabilistic predictions
            if f"{task_name}_mu" in predictions:
                pred_key = f"{task_name}_mu"
            
            if pred_key not in predictions or target_key not in targets:
                logger.warning(f"Missing data for task {task_name}")
                continue
            
            pred = predictions[pred_key]
            target = targets[target_key]
            
            # Reshape if needed
            if pred.ndim == 3:
                pred = pred.reshape(-1, pred.shape[-1])
                target = target.reshape(-1, target.shape[-1])
            
            # Remove NaN values
            mask = ~np.isnan(target) & ~np.isnan(pred)
            pred_valid = pred[mask]
            target_valid = target[mask]
            
            if len(pred_valid) == 0:
                logger.warning(f"No valid data for task {task_name}")
                continue
            
            # Compute metrics
            task_results = {}
            for metric_name in self.metrics:
                metric_func = getattr(self, f"compute_{metric_name}")
                metric_value = metric_func(pred_valid, target_valid)
                task_results[metric_name] = metric_value
            
            # Compute per-basin metrics if basin IDs provided
            if basin_ids is not None and len(basin_ids) == len(pred):
                basin_metrics = self._compute_per_basin_metrics(
                    pred, target, basin_ids, task_name
                )
                task_results['per_basin'] = basin_metrics
            
            results[task_name] = task_results
        
        # Save results if requested
        if save_results and self.save_dir:
            self.save_results(results)
        
        # Cache results
        self.results_cache = results
        
        return results
    
    def compute_nse(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute Nash-Sutcliffe Efficiency"""
        numerator = np.sum((target - pred) ** 2)
        denominator = np.sum((target - np.mean(target)) ** 2)
        
        if denominator == 0:
            return -np.inf
        
        nse = 1 - numerator / denominator
        return float(nse)
    
    def compute_kge(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute Kling-Gupta Efficiency"""
        # Correlation
        r = np.corrcoef(pred, target)[0, 1]
        if np.isnan(r):
            r = 0
        
        # Mean ratio (bias)
        beta = np.mean(pred) / np.mean(target)
        
        # Variability ratio
        alpha = np.std(pred) / np.std(target)
        
        # KGE
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        return float(kge)
    
    def compute_rmse(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute Root Mean Squared Error"""
        rmse = np.sqrt(np.mean((pred - target) ** 2))
        return float(rmse)
    
    def compute_mae(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute Mean Absolute Error"""
        mae = np.mean(np.abs(pred - target))
        return float(mae)
    
    def compute_pbias(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute Percent Bias"""
        pbias = 100 * np.sum(pred - target) / np.sum(target)
        return float(pbias)
    
    def compute_r2(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute R-squared"""
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        
        if ss_tot == 0:
            return -np.inf
        
        r2 = 1 - ss_res / ss_tot
        return float(r2)
    
    def compute_nselog(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute Log-NSE for low-flow performance"""
        pred_log = np.log(pred + 1e-6)
        target_log = np.log(target + 1e-6)
        
        numerator = np.sum((target_log - pred_log) ** 2)
        denominator = np.sum((target_log - np.mean(target_log)) ** 2)
        
        if denominator == 0:
            return -np.inf
        
        nselog = 1 - numerator / denominator
        return float(nselog)
    
    def _compute_per_basin_metrics(self, pred: np.ndarray, target: np.ndarray,
                                  basin_ids: List[str], task_name: str) -> pd.DataFrame:
        """Compute metrics for each basin individually"""
        unique_basins = np.unique(basin_ids)
        basin_results = []
        
        for basin_id in unique_basins:
            basin_mask = basin_ids == basin_id
            
            pred_basin = pred[basin_mask]
            target_basin = target[basin_mask]
            
            # Remove NaN values
            mask = ~np.isnan(target_basin) & ~np.isnan(pred_basin)
            pred_valid = pred_basin[mask]
            target_valid = target_basin[mask]
            
            if len(pred_valid) == 0:
                continue
            
            # Compute metrics for this basin
            basin_metrics = {'basin_id': basin_id}
            for metric_name in self.metrics:
                metric_func = getattr(self, f"compute_{metric_name}")
                try:
                    metric_value = metric_func(pred_valid, target_valid)
                    basin_metrics[metric_name] = metric_value
                except:
                    basin_metrics[metric_name] = np.nan
            
            basin_results.append(basin_metrics)
        
        return pd.DataFrame(basin_results)
    
    def analyze_predictions(self, predictions: Dict[str, np.ndarray],
                           targets: Dict[str, np.ndarray],
                           task_names: List[str]) -> Dict[str, Any]:
        """
        Perform detailed analysis of predictions
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            task_names: List of task names
            
        Returns:
            Dictionary with detailed analysis
        """
        analysis = {}
        
        for task_name in task_names:
            if task_name not in predictions or task_name not in targets:
                continue
            
            pred = predictions[task_name]
            target = targets[task_name]
            
            # Basic statistics
            task_analysis = {
                'prediction_stats': {
                    'mean': float(np.nanmean(pred)),
                    'std': float(np.nanstd(pred)),
                    'min': float(np.nanmin(pred)),
                    'max': float(np.nanmax(pred))
                },
                'target_stats': {
                    'mean': float(np.nanmean(target)),
                    'std': float(np.nanstd(target)),
                    'min': float(np.nanmin(target)),
                    'max': float(np.nanmax(target))
                }
            }
            
            # Error analysis
            errors = pred - target
            valid_mask = ~np.isnan(errors)
            
            if np.any(valid_mask):
                errors_valid = errors[valid_mask]
                task_analysis['error_stats'] = {
                    'mean_error': float(np.mean(errors_valid)),
                    'std_error': float(np.std(errors_valid)),
                    'mae': float(np.mean(np.abs(errors_valid))),
                    'rmse': float(np.sqrt(np.mean(errors_valid ** 2)))
                }
                
                # Error distribution percentiles
                percentiles = [10, 25, 50, 75, 90]
                error_percentiles = np.percentile(errors_valid, percentiles)
                task_analysis['error_percentiles'] = {
                    f'p{p}': float(val) for p, val in zip(percentiles, error_percentiles)
                }
            
            # Performance by flow regime
            if task_name in ['streamflow', 'usgsFlow']:
                flow_regime_analysis = self._analyze_flow_regimes(pred, target)
                task_analysis['flow_regime'] = flow_regime_analysis
            
            analysis[task_name] = task_analysis
        
        return analysis
    
    def _analyze_flow_regimes(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Analyze model performance across different flow regimes"""
        # Define flow percentiles
        flow_percentiles = np.percentile(target, [33, 66])
        
        # Low flow (<33%)
        low_flow_mask = target < flow_percentiles[0]
        # Medium flow (33-66%)
        medium_flow_mask = (target >= flow_percentiles[0]) & (target < flow_percentiles[1])
        # High flow (>=66%)
        high_flow_mask = target >= flow_percentiles[1]
        
        regime_analysis = {}
        
        for regime_name, mask in [('low_flow', low_flow_mask),
                                 ('medium_flow', medium_flow_mask),
                                 ('high_flow', high_flow_mask)]:
            if np.any(mask):
                pred_regime = pred[mask]
                target_regime = target[mask]
                
                # Compute NSE for this regime
                nse_regime = self.compute_nse(pred_regime, target_regime)
                regime_analysis[f'{regime_name}_nse'] = nse_regime
        
        return regime_analysis
    
    def generate_report(self, results: Dict[str, Any], 
                       analysis: Dict[str, Any]) -> str:
        """
        Generate a formatted evaluation report
        
        Args:
            results: Evaluation results from compute_all_metrics
            analysis: Detailed analysis from analyze_predictions
            
        Returns:
            Formatted report string
        """
        report_lines = ["=" * 60]
        report_lines.append("HYDROLOGICAL MODEL EVALUATION REPORT")
        report_lines.append("=" * 60)
        
        # Overall metrics
        report_lines.append("\nOVERALL METRICS:")
        report_lines.append("-" * 40)
        
        for task_name, task_results in results.items():
            if task_name == 'per_basin':
                continue
            
            report_lines.append(f"\n{task_name.upper()}:")
            for metric_name, metric_value in task_results.items():
                if metric_name == 'per_basin':
                    continue
                report_lines.append(f"  {metric_name.upper()}: {metric_value:.4f}")
        
        # Per-basin statistics if available
        for task_name, task_results in results.items():
            if 'per_basin' in task_results and isinstance(task_results['per_basin'], pd.DataFrame):
                df = task_results['per_basin']
                report_lines.append(f"\n{task_name.upper()} PER-BASIN STATISTICS:")
                report_lines.append("-" * 40)
                
                for metric_name in self.metrics:
                    if metric_name in df.columns:
                        report_lines.append(f"  {metric_name.upper()}:")
                        report_lines.append(f"    Mean: {df[metric_name].mean():.4f}")
                        report_lines.append(f"    Std: {df[metric_name].std():.4f}")
                        report_lines.append(f"    Min: {df[metric_name].min():.4f}")
                        report_lines.append(f"    Max: {df[metric_name].max():.4f}")
        
        # Detailed analysis
        report_lines.append("\nDETAILED ANALYSIS:")
        report_lines.append("-" * 40)
        
        for task_name, task_analysis in analysis.items():
            report_lines.append(f"\n{task_name.upper()}:")
            
            if 'prediction_stats' in task_analysis:
                report_lines.append("  Prediction Statistics:")
                for stat_name, stat_value in task_analysis['prediction_stats'].items():
                    report_lines.append(f"    {stat_name}: {stat_value:.4f}")
            
            if 'error_stats' in task_analysis:
                report_lines.append("  Error Statistics:")
                for stat_name, stat_value in task_analysis['error_stats'].items():
                    report_lines.append(f"    {stat_name}: {stat_value:.4f}")
        
        report_lines.append("\n" + "=" * 60)
        
        return "\n".join(report_lines)
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to files"""
        if not self.save_dir:
            return
        
        # Save overall metrics
        metrics_file = self.save_dir / 'metrics.json'
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            return obj
        
        serializable_results = {}
        for key, value in results.items():
            if key == 'per_basin' and isinstance(value, pd.DataFrame):
                serializable_results[key] = value.to_dict('records')
            else:
                serializable_results[key] = convert_to_serializable(value)
        
        with open(metrics_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save per-basin metrics as CSV
        for task_name, task_results in results.items():
            if 'per_basin' in task_results and isinstance(task_results['per_basin'], pd.DataFrame):
                csv_file = self.save_dir / f'{task_name}_per_basin_metrics.csv'
                task_results['per_basin'].to_csv(csv_file, index=False)
        
        logger.info(f"Evaluation results saved to {self.save_dir}")