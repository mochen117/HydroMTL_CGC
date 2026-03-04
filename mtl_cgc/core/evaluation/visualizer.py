import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Any
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class HydroVisualizer:
    """Visualization tools for hydrological model evaluation"""
    
    def __init__(self, save_dir: str = './visualizations', 
                 figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save visualizations
            figsize: Default figure size
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def plot_time_series(self, 
                        predictions: Dict[str, np.ndarray],
                        targets: Dict[str, np.ndarray],
                        timestamps: Optional[np.ndarray] = None,
                        task_names: Optional[List[str]] = None,
                        title: str = "Time Series Comparison",
                        save_name: str = "time_series.png") -> None:
        """
        Plot time series comparison for multiple tasks
        
        Args:
            predictions: Dictionary of predictions for each task
            targets: Dictionary of targets for each task
            timestamps: Time indices for x-axis
            task_names: List of task names to plot
            title: Plot title
            save_name: Filename to save plot
        """
        if task_names is None:
            task_names = list(predictions.keys())
        
        n_tasks = len(task_names)
        fig, axes = plt.subplots(n_tasks, 1, figsize=(self.figsize[0], 4*n_tasks))
        
        if n_tasks == 1:
            axes = [axes]
        
        for idx, task_name in enumerate(task_names):
            if task_name not in predictions or task_name not in targets:
                continue
                
            pred = predictions[task_name]
            target = targets[task_name]
            
            # Handle different shapes
            if pred.ndim > 1:
                pred = pred.flatten()
            if target.ndim > 1:
                target = target.flatten()
            
            # Create time indices if not provided
            if timestamps is None:
                timestamps = np.arange(len(pred))
            
            # Trim to common length
            min_len = min(len(pred), len(target), len(timestamps))
            pred_trimmed = pred[:min_len]
            target_trimmed = target[:min_len]
            time_trimmed = timestamps[:min_len]
            
            # Plot
            ax = axes[idx]
            ax.plot(time_trimmed, target_trimmed, 'b-', label='Observed', alpha=0.7, linewidth=1.5)
            ax.plot(time_trimmed, pred_trimmed, 'r-', label='Predicted', alpha=0.7, linewidth=1.5)
            
            ax.set_xlabel('Time')
            ax.set_ylabel(task_name)
            ax.set_title(f'{task_name} - Time Series')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add metrics as text
            metrics = self._calculate_basic_metrics(pred_trimmed, target_trimmed)
            metrics_text = f"NSE: {metrics['nse']:.3f}, KGE: {metrics['kge']:.3f}, RMSE: {metrics['rmse']:.3f}"
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved time series plot to {self.save_dir / save_name}")
    
    def plot_scatter(self, 
                    predictions: Dict[str, np.ndarray],
                    targets: Dict[str, np.ndarray],
                    task_names: Optional[List[str]] = None,
                    title: str = "Predicted vs Observed",
                    save_name: str = "scatter_plot.png") -> None:
        """
        Create scatter plots of predicted vs observed values
        
        Args:
            predictions: Dictionary of predictions
            targets: Dictionary of targets
            task_names: List of task names
            title: Plot title
            save_name: Filename to save plot
        """
        if task_names is None:
            task_names = list(predictions.keys())
        
        n_tasks = len(task_names)
        n_cols = min(2, n_tasks)
        n_rows = (n_tasks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
        
        if n_tasks == 1:
            axes = np.array([axes])
        if axes.ndim == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, task_name in enumerate(task_names):
            if task_name not in predictions or task_name not in targets:
                continue
                
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            pred = predictions[task_name].flatten()
            target = targets[task_name].flatten()
            
            # Remove NaN values
            mask = ~np.isnan(target) & ~np.isnan(pred)
            pred_clean = pred[mask]
            target_clean = target[mask]
            
            if len(pred_clean) == 0:
                continue
            
            # Create scatter plot
            scatter = ax.scatter(target_clean, pred_clean, alpha=0.6, s=20, 
                               c=np.abs(pred_clean - target_clean), cmap='viridis')
            
            # Add 1:1 line
            min_val = min(target_clean.min(), pred_clean.min())
            max_val = max(target_clean.max(), pred_clean.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='1:1 line')
            
            # Add regression line
            if len(target_clean) > 1:
                z = np.polyfit(target_clean, pred_clean, 1)
                p = np.poly1d(z)
                ax.plot(target_clean, p(target_clean), "g--", alpha=0.8, linewidth=2, label='Regression')
            
            ax.set_xlabel('Observed')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{task_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add metrics
            metrics = self._calculate_basic_metrics(pred_clean, target_clean)
            metrics_text = f"R²: {metrics['r2']:.3f}\nMAE: {metrics['mae']:.3f}"
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Add colorbar for errors
            if idx == 0:
                plt.colorbar(scatter, ax=ax, label='Absolute Error')
        
        # Hide empty subplots
        for idx in range(len(task_names), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved scatter plot to {self.save_dir / save_name}")
    
    def plot_error_distribution(self, 
                               predictions: Dict[str, np.ndarray],
                               targets: Dict[str, np.ndarray],
                               task_names: Optional[List[str]] = None,
                               title: str = "Error Distribution",
                               save_name: str = "error_distribution.png") -> None:
        """
        Plot error distribution for each task
        
        Args:
            predictions: Dictionary of predictions
            targets: Dictionary of targets
            task_names: List of task names
            title: Plot title
            save_name: Filename to save plot
        """
        if task_names is None:
            task_names = list(predictions.keys())
        
        n_tasks = len(task_names)
        fig, axes = plt.subplots(1, n_tasks, figsize=(5*n_tasks, 5))
        
        if n_tasks == 1:
            axes = [axes]
        
        for idx, task_name in enumerate(task_names):
            if task_name not in predictions or task_name not in targets:
                continue
                
            pred = predictions[task_name].flatten()
            target = targets[task_name].flatten()
            
            # Calculate errors
            errors = pred - target
            errors = errors[~np.isnan(errors)]
            
            if len(errors) == 0:
                continue
            
            # Plot histogram
            ax = axes[idx]
            ax.hist(errors, bins=50, alpha=0.7, density=True, edgecolor='black')
            
            # Add normal distribution fit
            from scipy.stats import norm
            mu, std = norm.fit(errors)
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, 'r-', linewidth=2, label=f'Normal fit\nμ={mu:.3f}, σ={std:.3f}')
            
            ax.set_xlabel('Error')
            ax.set_ylabel('Density')
            ax.set_title(f'{task_name} - Error Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f"Mean: {errors.mean():.3f}\nStd: {errors.std():.3f}\nSkew: {pd.Series(errors).skew():.3f}"
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved error distribution plot to {self.save_dir / save_name}")
    
    def plot_correlation_heatmap(self,
                                predictions: Dict[str, np.ndarray],
                                targets: Dict[str, np.ndarray],
                                title: str = "Correlation Heatmap",
                                save_name: str = "correlation_heatmap.png") -> None:
        """
        Plot correlation heatmap between tasks
        
        Args:
            predictions: Dictionary of predictions
            targets: Dictionary of targets
            title: Plot title
            save_name: Filename to save plot
        """
        # Combine predictions and errors
        all_data = {}
        for task_name in predictions.keys():
            if task_name in targets:
                pred = predictions[task_name].flatten()
                target = targets[task_name].flatten()
                
                # Remove NaN
                mask = ~np.isnan(target) & ~np.isnan(pred)
                pred_clean = pred[mask]
                target_clean = target[mask]
                
                if len(pred_clean) > 0:
                    all_data[f'{task_name}_pred'] = pred_clean
                    all_data[f'{task_name}_obs'] = target_clean
                    all_data[f'{task_name}_error'] = pred_clean - target_clean
        
        if len(all_data) == 0:
            logger.warning("No valid data for correlation heatmap")
            return
        
        # Create DataFrame
        max_len = min(len(v) for v in all_data.values())
        data_truncated = {k: v[:max_len] for k, v in all_data.items()}
        df = pd.DataFrame(data_truncated)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add text annotations
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)
        
        # Set labels
        ax.set_xticks(np.arange(len(corr_matrix.columns)))
        ax.set_yticks(np.arange(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)
        
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved correlation heatmap to {self.save_dir / save_name}")
    
    def create_summary_report(self,
                             predictions: Dict[str, np.ndarray],
                             targets: Dict[str, np.ndarray],
                             metrics_dict: Optional[Dict[str, Dict[str, float]]] = None,
                             report_name: str = "model_evaluation_report.html") -> None:
        """
        Create HTML summary report with all visualizations
        
        Args:
            predictions: Dictionary of predictions
            targets: Dictionary of targets
            metrics_dict: Dictionary of calculated metrics
            report_name: Filename for HTML report
        """
        try:
            import jinja2
        except ImportError:
            logger.warning("Jinja2 not installed, skipping HTML report")
            return
        
        # Generate all plots
        plot_files = []
        
        # Time series plot
        ts_plot = "time_series_summary.png"
        self.plot_time_series(predictions, targets, save_name=ts_plot)
        plot_files.append(ts_plot)
        
        # Scatter plot
        scatter_plot = "scatter_summary.png"
        self.plot_scatter(predictions, targets, save_name=scatter_plot)
        plot_files.append(scatter_plot)
        
        # Error distribution
        error_plot = "error_distribution_summary.png"
        self.plot_error_distribution(predictions, targets, save_name=error_plot)
        plot_files.append(error_plot)
        
        # Correlation heatmap
        corr_plot = "correlation_summary.png"
        self.plot_correlation_heatmap(predictions, targets, save_name=corr_plot)
        plot_files.append(corr_plot)
        
        # Calculate metrics if not provided
        if metrics_dict is None:
            metrics_dict = {}
            for task_name in predictions.keys():
                if task_name in targets:
                    pred = predictions[task_name].flatten()
                    target = targets[task_name].flatten()
                    mask = ~np.isnan(target) & ~np.isnan(pred)
                    pred_clean = pred[mask]
                    target_clean = target[mask]
                    
                    if len(pred_clean) > 0:
                        metrics_dict[task_name] = self._calculate_basic_metrics(pred_clean, target_clean)
        
        # Create HTML report
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2c3e50; }
                h2 { color: #34495e; margin-top: 30px; }
                .metrics-table { border-collapse: collapse; width: 100%; }
                .metrics-table th, .metrics-table td { border: 1px solid #ddd; padding: 8px; text-align: center; }
                .metrics-table th { background-color: #4CAF50; color: white; }
                .plot-container { margin: 20px 0; text-align: center; }
                .plot-container img { max-width: 100%; height: auto; border: 1px solid #ddd; }
            </style>
        </head>
        <body>
            <h1>HydroMTL Model Evaluation Report</h1>
            
            <h2>Model Performance Metrics</h2>
            <table class="metrics-table">
                <tr>
                    <th>Task</th>
                    <th>NSE</th>
                    <th>KGE</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>R²</th>
                </tr>
                {% for task, metrics in metrics_dict.items() %}
                <tr>
                    <td>{{ task }}</td>
                    <td>{{ metrics.get('nse', 'N/A') | round(3) }}</td>
                    <td>{{ metrics.get('kge', 'N/A') | round(3) }}</td>
                    <td>{{ metrics.get('rmse', 'N/A') | round(3) }}</td>
                    <td>{{ metrics.get('mae', 'N/A') | round(3) }}</td>
                    <td>{{ metrics.get('r2', 'N/A') | round(3) }}</td>
                </tr>
                {% endfor %}
            </table>
            
            {% for plot_file in plot_files %}
            <h2>{{ plot_file.replace('_summary.png', '').replace('_', ' ').title() }}</h2>
            <div class="plot-container">
                <img src="{{ plot_file }}" alt="{{ plot_file }}">
            </div>
            {% endfor %}
        </body>
        </html>
        """
        
        # Render template
        template = jinja2.Template(template_str)
        html_content = template.render(
            metrics_dict=metrics_dict,
            plot_files=plot_files
        )
        
        # Save HTML file
        report_path = self.save_dir / report_name
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Saved HTML report to {report_path}")
    
    def _calculate_basic_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Calculate basic evaluation metrics"""
        if len(pred) == 0 or len(target) == 0:
            return {}
        
        # Remove any remaining NaN
        mask = ~np.isnan(pred) & ~np.isnan(target)
        pred_clean = pred[mask]
        target_clean = target[mask]
        
        if len(pred_clean) == 0:
            return {}
        
        # Calculate metrics
        mse = np.mean((pred_clean - target_clean) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_clean - target_clean))
        
        # R²
        ss_res = np.sum((target_clean - pred_clean) ** 2)
        ss_tot = np.sum((target_clean - np.mean(target_clean)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8) if ss_tot > 0 else 0
        
        # NSE
        nse = 1 - ss_res / (ss_tot + 1e-8)
        
        # KGE components
        r = np.corrcoef(pred_clean, target_clean)[0, 1] if len(pred_clean) > 1 else 0
        alpha = np.std(pred_clean) / (np.std(target_clean) + 1e-8)
        beta = np.mean(pred_clean) / (np.mean(target_clean) + 1e-8)
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        
        return {
            'nse': nse,
            'kge': kge,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mse': mse
        }