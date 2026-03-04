"""
Data loader factory for creating train/val/test dataloaders
"""

import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple
import logging
from .data_set import MultiBasinDataset  # Changed from .dataset to .data_set

logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """Factory for creating data loaders for multi-task learning"""
    
    @staticmethod
    def create_dataloaders(
        nc_files: List[str],
        basin_ids: List[str],
        config: Any,  # Should be DataConfig, not Dict
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle_train: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
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
        
        # Extract configuration using getattr for DataConfig objects
        sequence_length = getattr(config, 'sequence_length', 365)
        prediction_horizon = getattr(config, 'prediction_horizon', 1)
        dynamic_features = getattr(config, 'dynamic_features', None)
        static_features = getattr(config, 'static_features', None)
        if hasattr(config, 'targets') and config.targets:
            target_features = [t['name'] for t in config.targets]
        else:
            target_features = getattr(config, 'target_features', ['streamflow', 'evapotranspiration'])
        normalize = getattr(config, 'normalize', True)
        train_ratio = getattr(config, 'train_ratio', 0.7)
        val_ratio = getattr(config, 'val_ratio', 0.15)
        
        logger.info(f"Creating dataloaders for {len(basin_ids)} basins")
        logger.info(f"Sequence length: {sequence_length}, Prediction horizon: {prediction_horizon}")
        
        # Create datasets for each split
        train_dataset = MultiBasinDataset(
            nc_files=nc_files,
            basin_ids=basin_ids,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            dynamic_features=dynamic_features,
            static_features=static_features,
            target_features=target_features,
            normalize=normalize,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            mode='train'
        )
        
        val_dataset = MultiBasinDataset(
            nc_files=nc_files,
            basin_ids=basin_ids,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            dynamic_features=dynamic_features,
            static_features=static_features,
            target_features=target_features,
            normalize=normalize,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            mode='val'
        )
        
        test_dataset = MultiBasinDataset(
            nc_files=nc_files,
            basin_ids=basin_ids,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            dynamic_features=dynamic_features,
            static_features=static_features,
            target_features=target_features,
            normalize=normalize,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            mode='test'
        )
        
        # Get dataset statistics
        train_stats = train_dataset.get_dataset_stats()
        val_stats = val_dataset.get_dataset_stats()
        test_stats = test_dataset.get_dataset_stats()
        
        logger.info(f"Train sequences: {len(train_dataset)}")
        logger.info(f"Validation sequences: {len(val_dataset)}")
        logger.info(f"Test sequences: {len(test_dataset)}")
        
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
        
        # Calculate input dimension
        sample = train_dataset[0]
        input_dim = sample['features'].shape[-1]
        
        logger.info(f"Input dimension: {input_dim}")
        logger.info(f"Batch size: {batch_size}")
        
        return train_loader, val_loader, test_loader, {
            'train_stats': train_stats,
            'val_stats': val_stats,
            'test_stats': test_stats,
            'input_dim': input_dim,
            'num_basins': len(basin_ids)
        }
    
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
        
        # Stack features
        collated['features'] = torch.stack([item['features'] for item in batch])
        
        # Stack targets for each task
        collated['streamflow'] = torch.stack([item['streamflow'] for item in batch])
        collated['evapotranspiration'] = torch.stack([item['evapotranspiration'] for item in batch])
        
        # Stack basin indices
        collated['basin_idx'] = torch.stack([item['basin_idx'] for item in batch])
        
        # Keep basin IDs as list
        collated['basin_id'] = [item['basin_id'] for item in batch]
        
        # Keep time indices
        collated['time_index'] = torch.tensor([item['time_index'] for item in batch])
        
        return collated


def create_data_loaders(config: Any, basin_ids: List[str]) -> Dict[str, Any]:
    """
    Create data loaders for training, validation, and testing
    
    Args:
        config: Data configuration object (DataConfig)
        basin_ids: List of basin IDs
        
    Returns:
        Dictionary containing data loaders
    """
    # Extract necessary configuration
    data_config = config
    
    # Get NetCDF files from config or use default pattern
    nc_files = getattr(data_config, 'nc_files', [])
    if not nc_files:
        # If no files specified, use a default pattern
        data_root = getattr(data_config, 'data_root', './data')
        import os
        nc_files = [os.path.join(data_root, f'gage_{basin_id}.nc') for basin_id in basin_ids]
    
    # Create data loaders using the factory
    train_loader, val_loader, test_loader, stats = DataLoaderFactory.create_dataloaders(
        nc_files=nc_files,
        basin_ids=basin_ids,
        config=data_config,
        batch_size=getattr(data_config, 'batch_size', 32),
        num_workers=getattr(data_config, 'num_workers', 4),
        shuffle_train=getattr(data_config, 'shuffle_train', True)
    )
    
    # Return a dictionary with all loaders and metadata
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'stats': stats,
        'basin_ids': basin_ids,
        'input_dim': stats['input_dim']
    }