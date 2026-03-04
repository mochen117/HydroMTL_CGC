"""
Configuration parser for HydroMTL_CGC
Handles YAML config files with environment variable substitution
"""

import os
import yaml
import re
import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration dataclass"""
    dataset: str
    data_root: str
    static_features: list
    dynamic_features: list
    targets: list
    sequence_length: int
    prediction_horizon: int
    train_period: list
    val_period: list
    test_period: list
    batch_size: int
    num_workers: int
    normalize: bool = True
    forecast_history: int = 365
    fill_missing: bool = True

    def __post_init__(self):
        """Validate configuration after initialization"""
        if len(self.targets) != len(set(t['name'] for t in self.targets)):
            raise ValueError("Target names must be unique")

        if self.prediction_horizon < 1:
            raise ValueError("Prediction horizon must be >= 1")


@dataclass
class ModelConfig:
    """Model configuration dataclass"""
    cgc: Dict[str, Any]
    encoder: Dict[str, Any]
    task_towers: list
    physics_constraints: Dict[str, Any]

    def __post_init__(self):
        """Validate model configuration"""
        if self.cgc['shared_experts'] < 1:
            raise ValueError("Number of shared experts must be >= 1")

        if len(self.cgc['task_experts']) != len(self.task_towers):
            raise ValueError(
                f"Number of task experts ({len(self.cgc['task_experts'])}) "
                f"must match number of task towers ({len(self.task_towers)})"
            )


@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    optimizer: str
    learning_rate: float
    epochs: int
    loss: Dict[str, Any]
    patience: int
    save_frequency: int

    # Added missing fields with default values
    weight_decay: float = 0.0001
    clip_grad_norm: float = 1.0
    warmup_epochs: int = 10
    dropout: float = 0.2
    batch_norm: bool = True
    save_best_only: bool = True

    # New fields required by trainer
    scheduler: Dict[str, Any] = field(default_factory=dict)
    early_stopping: Dict[str, Any] = field(default_factory=dict)
    checkpoint: Dict[str, Any] = field(default_factory=dict)
    device: str = "auto"

    def __post_init__(self):
        """Validate training configuration"""
        valid_optimizers = ['adam', 'sgd', 'rmsprop', 'adamw']
        if self.optimizer.lower() not in valid_optimizers:
            raise ValueError(f"Optimizer must be one of {valid_optimizers}")

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if self.weight_decay < 0:
            raise ValueError("Weight decay must be non-negative")


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    experiment: Dict[str, Any]
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: Dict[str, Any]
    logging: Dict[str, Any]

    def __post_init__(self):
        """Set experiment timestamp and create save directory"""
        if not self.experiment.get('timestamp'):
            self.experiment['timestamp'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        save_dir = Path(self.experiment['save_dir']) / f"{self.experiment['name']}_{self.experiment['timestamp']}"
        save_dir.mkdir(parents=True, exist_ok=True)
        self.experiment['save_dir'] = str(save_dir)


class ConfigParser:
    """Parse and manage configuration files"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration parser

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = None

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> ExperimentConfig:
        """
        Load configuration from YAML file

        Args:
            config_path: Path to YAML configuration file

        Returns:
            ExperimentConfig object
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        config_dict = self._substitute_env_vars(config_dict)

        # Helper to filter unknown keyword arguments
        def filter_kwargs(cls, source_dict):
            valid_fields = {f.name for f in fields(cls)}
            return {k: v for k, v in source_dict.items() if k in valid_fields}

        data_config = DataConfig(**filter_kwargs(DataConfig, config_dict['data']))
        model_config = ModelConfig(**filter_kwargs(ModelConfig, config_dict['model']))
        training_config = TrainingConfig(**filter_kwargs(TrainingConfig, config_dict['training']))

        self.config = ExperimentConfig(
            experiment=config_dict['experiment'],
            data=data_config,
            model=model_config,
            training=training_config,
            evaluation=config_dict['evaluation'],
            logging=config_dict['logging']
        )

        return self.config

    def _substitute_env_vars(self, config_dict: Dict) -> Dict:
        """
        Substitute environment variables in configuration

        Args:
            config_dict: Configuration dictionary

        Returns:
            Updated configuration dictionary
        """
        def substitute(value):
            if isinstance(value, str):
                match = re.match(r'\$\{(.+)\}', value)
                if match:
                    env_var = match.group(1)
                    return os.environ.get(env_var, value)
            return value

        def recursive_substitute(obj):
            if isinstance(obj, dict):
                return {k: recursive_substitute(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_substitute(item) for item in obj]
            else:
                return substitute(obj)

        return recursive_substitute(config_dict)

    def save_config(self, save_path: Optional[str] = None):
        """
        Save configuration to file

        Args:
            save_path: Path to save configuration file
        """
        if not self.config:
            raise ValueError("No configuration loaded")

        if not save_path:
            save_path = Path(self.config.experiment['save_dir']) / 'config.yaml'

        config_dict = {
            'experiment': self.config.experiment,
            'data': self.config.data.__dict__,
            'model': self.config.model.__dict__,
            'training': self.config.training.__dict__,
            'evaluation': self.config.evaluation,
            'logging': self.config.logging
        }

        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        print(f"Configuration saved to {save_path}")

    def update_from_args(self, args_dict: Dict[str, Any]):
        """
        Update configuration from command line arguments

        Args:
            args_dict: Dictionary of command line arguments
        """
        if not self.config:
            raise ValueError("No configuration loaded")

        if 'experiment_name' in args_dict and args_dict['experiment_name']:
            self.config.experiment['name'] = args_dict['experiment_name']

        if 'learning_rate' in args_dict and args_dict['learning_rate']:
            self.config.training.learning_rate = args_dict['learning_rate']

        if 'batch_size' in args_dict and args_dict['batch_size']:
            self.config.data.batch_size = args_dict['batch_size']

        if 'epochs' in args_dict and args_dict['epochs']:
            self.config.training.epochs = args_dict['epochs']

        if 'shared_experts' in args_dict and args_dict['shared_experts']:
            self.config.model.cgc['shared_experts'] = args_dict['shared_experts']