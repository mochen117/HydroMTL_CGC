import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Any
import logging

from .data_set import build_multi_basin_datasets

logger = logging.getLogger(__name__)


def create_data_loaders(config: Any, basin_ids: List[str]) -> Dict[str, Any]:
    data_root = Path(getattr(config, 'data_root', './'))
    nc_files = [str(data_root / f"gage_{basin_id}.nc") for basin_id in basin_ids]

    dynamic_features = getattr(config, 'dynamic_features', [])
    static_features = getattr(config, 'static_features', [])
    categorical_features = getattr(config, 'categorical_static_features', [])
    
    target_features = []
    if hasattr(config, 'targets'):
        target_features = [t['name'] if isinstance(t, dict) else t for t in config.targets]

    train_period = getattr(config, 'train_period', None)
    val_period = getattr(config, 'val_period', None)
    test_period = getattr(config, 'test_period', None)
    train_ratio = getattr(config, 'train_ratio', 0.7)
    val_ratio = getattr(config, 'val_ratio', 0.15)
    sequence_length = getattr(config, 'sequence_length', 365)
    prediction_horizon = getattr(config, 'prediction_horizon', 1)

    logger.info("Building datasets from NetCDF files...")
    
    train_dataset, val_dataset, test_dataset, basin_scalers = build_multi_basin_datasets(
        nc_files=nc_files,
        basin_ids=basin_ids,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        dynamic_features=dynamic_features,
        static_features=static_features,
        categorical_features=categorical_features,
        target_features=target_features,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        train_period=train_period,
        val_period=val_period,
        test_period=test_period
    )

    batch_size = getattr(config, 'batch_size', 256)
    num_workers = getattr(config, 'num_workers', 8)
    prefetch_factor = getattr(config, 'prefetch_factor', 2)

    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
    }
    
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs['prefetch_factor'] = prefetch_factor

    logger.info(f"Creating DataLoaders with batch_size={batch_size}, num_workers={num_workers}")
    
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    
    if test_dataset and len(test_dataset) > 0:
        test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    else:
        test_loader = None

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'basin_scalers': basin_scalers
    }