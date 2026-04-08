"""
Data loader factory for creating train/val/test dataloaders
"""

import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple
import logging
from .data_set import build_multi_basin_datasets, HydroBasinDataset

logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """Factory for creating data loaders for multi-task learning"""

    @staticmethod
    def create_dataloaders(
        nc_files: List[str],
        basin_ids: List[str],
        config: Any,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle_train: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
        """
        Create train, validation, and test dataloaders

        Args:
            nc_files: List of NetCDF file paths
            basin_ids: List of basin identifiers
            config: Data configuration object (DataConfig)
            batch_size: Batch size
            num_workers: Number of worker processes
            shuffle_train: Whether to shuffle training data

        Returns:
            Tuple of (train_loader, val_loader, test_loader, stats)
        """

        # Extract configuration parameters
        sequence_length = getattr(config, 'sequence_length', 365)
        prediction_horizon = getattr(config, 'prediction_horizon', 1)
        dynamic_features = getattr(config, 'dynamic_features', None)
        static_features = getattr(config, 'static_features', None)
        target_features = [t['name'] for t in getattr(config, 'targets', [])]
        train_ratio = getattr(config, 'train_ratio', 0.7)
        val_ratio = getattr(config, 'val_ratio', 0.15)

        # Check for date-based splitting parameters
        train_period = getattr(config, 'train_period', None)
        val_period = getattr(config, 'val_period', None)
        test_period = getattr(config, 'test_period', None)

        logger.info(f"Creating dataloaders for {len(basin_ids)} basins")
        logger.info(f"Sequence length: {sequence_length}, Prediction horizon: {prediction_horizon}")

        # Build the three datasets with correct normalization
        train_dataset, val_dataset, test_dataset = build_multi_basin_datasets(
            nc_files=nc_files,
            basin_ids=basin_ids,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            dynamic_features=dynamic_features,
            static_features=static_features,
            target_features=target_features,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            train_period=train_period,
            val_period=val_period,
            test_period=test_period
        )

        logger.info(f"Train sequences: {len(train_dataset)}")
        logger.info(f"Validation sequences: {len(val_dataset)}")
        logger.info(f"Test sequences: {len(test_dataset)}")

        # Extract basin_scalers from the training dataset (each basin's target scalers)
        # Format: list of dicts, each dict maps task name to StandardScaler
        basin_scalers = []
        for basin_ds in train_dataset.basin_datasets:
            task_scalers = {}
            target_scalers = basin_ds.scalers['targets']  # list of scalers
            for task_name, scaler in zip(basin_ds.target_features, target_scalers):
                task_scalers[task_name] = scaler
            basin_scalers.append(task_scalers)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        # Compute input dimension from a sample
        sample = train_dataset[0]
        input_dim = sample['features'].shape[-1]

        logger.info(f"Input dimension: {input_dim}")
        logger.info(f"Batch size: {batch_size}")

        # Return minimal stats and basin_scalers
        stats = {
            'input_dim': input_dim,
            'num_basins': len(basin_ids),
            'basin_scalers': basin_scalers   # for inverse transform in trainer
        }

        return train_loader, val_loader, test_loader, stats

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for multi-task learning

        Args:
            batch: List of samples

        Returns:
            Collated batch
        """
        collated = {}
        collated['features'] = torch.stack([item['features'] for item in batch])
        collated['streamflow'] = torch.stack([item['streamflow'] for item in batch])
        collated['evapotranspiration'] = torch.stack([item['evapotranspiration'] for item in batch])
        collated['basin_idx'] = torch.stack([item['basin_idx'] for item in batch])
        collated['basin_id'] = [item['basin_id'] for item in batch]
        collated['time_index'] = torch.tensor([item['time_index'] for item in batch])
        return collated


def create_data_loaders(config: Any, basin_ids: List[str]) -> Dict[str, Any]:
    """
    Create data loaders for training, validation, and testing

    Args:
        config: Data configuration object (DataConfig)
        basin_ids: List of basin IDs

    Returns:
        Dictionary containing data loaders and stats
    """
    data_config = config
    data_root = getattr(data_config, 'data_root', './data')
    import os
    nc_files = [os.path.join(data_root, f'gage_{basin_id}.nc') for basin_id in basin_ids]

    train_loader, val_loader, test_loader, stats = DataLoaderFactory.create_dataloaders(
        nc_files=nc_files,
        basin_ids=basin_ids,
        config=data_config,
        batch_size=getattr(data_config, 'batch_size', 32),
        num_workers=getattr(data_config, 'num_workers', 4),
        shuffle_train=getattr(data_config, 'shuffle_train', True)
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'stats': stats,
        'basin_ids': basin_ids,
        'input_dim': stats['input_dim'],
        'basin_scalers': stats['basin_scalers']
    }