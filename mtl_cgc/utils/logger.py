"""
Logging utilities for HydroMTL_CGC
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import datetime


def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with console output
    
    Args:
        name: Logger name
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger


def setup_file_logger(name: str, log_file: Path, 
                     log_level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with file output
    
    Args:
        name: Logger name
        log_file: Path to log file
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger


def setup_experiment_logger(experiment_dir: Path, 
                           experiment_name: str) -> logging.Logger:
    """
    Setup comprehensive logger for an experiment
    
    Args:
        experiment_dir: Experiment directory
        experiment_name: Name of the experiment
        
    Returns:
        Configured logger
    """
    # Create log directory
    log_dir = experiment_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{experiment_name}_{timestamp}.log'
    
    # Setup file logger
    logger = setup_file_logger('experiment', log_file, logging.INFO)
    
    # Also add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


class ExperimentLogger:
    """Custom logger for experiment tracking"""
    
    def __init__(self, experiment_dir: Path, experiment_name: str):
        """
        Initialize experiment logger
        
        Args:
            experiment_dir: Experiment directory
            experiment_name: Name of the experiment
        """
        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_name
        
        # Setup logger
        self.logger = setup_experiment_logger(experiment_dir, experiment_name)
        
        # Metrics tracking
        self.metrics_history = {}
    
    def log_config(self, config: dict):
        """Log configuration"""
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info("Configuration:")
        for key, value in config.items():
            if isinstance(value, dict):
                self.logger.info(f"  {key}:")
                for subkey, subvalue in value.items():
                    self.logger.info(f"    {subkey}: {subvalue}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_metric(self, epoch: int, metric_name: str, 
                  metric_value: float, phase: str = 'train'):
        """
        Log metric value
        
        Args:
            epoch: Epoch number
            metric_name: Name of the metric
            metric_value: Value of the metric
            phase: Phase (train/val/test)
        """
        key = f"{phase}_{metric_name}"
        if key not in self.metrics_history:
            self.metrics_history[key] = []
        
        self.metrics_history[key].append((epoch, metric_value))
        
        self.logger.info(f"Epoch {epoch} - {phase} {metric_name}: {metric_value:.4f}")
    
    def log_message(self, message: str, level: str = 'info'):
        """
        Log general message
        
        Args:
            message: Message to log
            level: Log level (info/warning/error)
        """
        if level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
    
    def save_metrics(self):
        """Save metrics history to file"""
        import json
        import numpy as np
        
        # Convert to JSON serializable format
        serializable_metrics = {}
        for key, values in self.metrics_history.items():
            serializable_metrics[key] = [
                (int(epoch), float(value) if not np.isnan(value) else None)
                for epoch, value in values
            ]
        
        # Save to file
        metrics_file = self.experiment_dir / 'metrics_history.json'
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        self.logger.info(f"Metrics history saved to {metrics_file}")