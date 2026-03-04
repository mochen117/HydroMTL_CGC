"""
Dataset classes for multi-task hydrological modeling
Handles multiple basins, multiple tasks, and temporal sequences
"""

import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.preprocessing import StandardScaler
import pickle
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class HydroBasinDataset(Dataset):
    """
    Dataset for a single hydrological basin
    Loads data from NetCDF files and prepares sequences for training
    """

    def __init__(self,
                 nc_file: str,
                 basin_id: str,
                 sequence_length: int = 365,
                 prediction_horizon: int = 1,
                 dynamic_features: List[str] = None,
                 static_features: List[str] = None,
                 target_features: List[str] = None,
                 normalize: bool = True,
                 scalers: Dict[str, Any] = None):
        """
        Initialize dataset for a single basin

        Args:
            nc_file: Path to NetCDF file
            basin_id: Basin identifier
            sequence_length: Length of input sequence (days)
            prediction_horizon: Number of days to predict ahead
            dynamic_features: List of dynamic feature names
            static_features: List of static feature names
            target_features: List of target feature names
            normalize: Whether to normalize features
            scalers: Pre-trained scalers for normalization
        """
        self.nc_file = nc_file
        self.basin_id = basin_id
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize

        # Default feature lists based on your NC file
        if dynamic_features is None:
            self.dynamic_features = [
                'total_precipitation',  # mm/day
                'temperature',          # degree_C
                'specific_humidity',    # kg/kg
                'shortwave_radiation',  # W/m^2
                'potential_energy'      # J/kg
            ]
        else:
            self.dynamic_features = dynamic_features

        if static_features is None:
            self.static_features = [
                'elev_mean',            # m
                'slope_mean',           # m/km
                'area_gages2',          # km^2
                'frac_forest',          # 1
                'lai_max',              # 1
                'lai_diff',             # 1
                'soil_porosity',        # 1
                'soil_conductivity',    # cm/hr
                'max_water_content',    # m
                'geol_porosity',        # 1
                'geol_permeability'     # m^2
            ]
        else:
            self.static_features = static_features

        if target_features is None:
            # Multi-task targets: streamflow and evapotranspiration
            self.target_features = ['streamflow', 'evapotranspiration']
        else:
            self.target_features = target_features

        # Load and prepare data
        self.data = self._load_and_prepare_data()

        # Initialize scalers if not provided
        if scalers is None:
            self.scalers = self._initialize_scalers()
        else:
            self.scalers = scalers

        # Normalize data if requested
        if normalize:
            self.data = self._normalize_data()

        # Create sequences
        self.sequences = self._create_sequences()

        logger.info(f"Loaded basin {basin_id}: {len(self.sequences)} sequences")

    def _load_and_prepare_data(self) -> Dict[str, np.ndarray]:
        """Load data from NetCDF file and prepare for training"""

        # Open NetCDF file
        ds = xr.open_dataset(self.nc_file)

        # Extract dynamic features
        dynamic_data = {}
        for feature in self.dynamic_features:
            if feature in ds:
                data = ds[feature].values
                # Handle missing values (NaN)
                if np.any(np.isnan(data)):
                    # Linear interpolation for missing values
                    mask = np.isnan(data)
                    data[mask] = np.interp(
                        np.where(mask)[0],
                        np.where(~mask)[0],
                        data[~mask]
                    )
                dynamic_data[feature] = data
            else:
                logger.warning(f"Dynamic feature {feature} not found in {self.nc_file}")
                dynamic_data[feature] = np.zeros(len(ds.time))

        # Extract static features
        static_data = {}
        for feature in self.static_features:
            if feature in ds:
                # Static features are scalar values, repeat for all time steps
                static_data[feature] = np.full(
                    len(ds.time),
                    float(ds[feature].values)
                )
            else:
                logger.warning(f"Static feature {feature} not found in {self.nc_file}")
                static_data[feature] = np.zeros(len(ds.time))

        # Extract target features
        target_data = {}
        for feature in self.target_features:
            if feature in ds:
                data = ds[feature].values
                # Handle missing values
                if np.any(np.isnan(data)):
                    mask = np.isnan(data)
                    data[mask] = np.interp(
                        np.where(mask)[0],
                        np.where(~mask)[0],
                        data[~mask]
                    )
                target_data[feature] = data
            else:
                logger.warning(f"Target feature {feature} not found in {self.nc_file}")
                target_data[feature] = np.zeros(len(ds.time))

        # Extract time information
        time_data = ds.time.values

        # Close dataset
        ds.close()

        # Combine all data
        data = {
            'dynamic': np.column_stack([dynamic_data[f] for f in self.dynamic_features]),
            'static': np.column_stack([static_data[f] for f in self.static_features]),
            'targets': np.column_stack([target_data[f] for f in self.target_features]),
            'time': time_data,
            'basin_id': self.basin_id
        }

        # Add metadata
        data['dynamic_feature_names'] = self.dynamic_features
        data['static_feature_names'] = self.static_features
        data['target_feature_names'] = self.target_features

        return data

    def _initialize_scalers(self) -> Dict[str, StandardScaler]:
        """Initialize scalers for normalization"""
        scalers = {}

        # Scaler for dynamic features
        scalers['dynamic'] = StandardScaler()
        scalers['dynamic'].fit(self.data['dynamic'])

        # Scaler for static features
        scalers['static'] = StandardScaler()
        scalers['static'].fit(self.data['static'])

        # Scaler for targets
        scalers['targets'] = StandardScaler()
        scalers['targets'].fit(self.data['targets'])

        return scalers

    def _normalize_data(self) -> Dict[str, np.ndarray]:
        """Normalize data using fitted scalers"""
        normalized_data = self.data.copy()

        # Normalize dynamic features
        if self.data['dynamic'].shape[0] > 0:
            normalized_data['dynamic'] = self.scalers['dynamic'].transform(
                self.data['dynamic']
            )

        # Normalize static features
        if self.data['static'].shape[0] > 0:
            normalized_data['static'] = self.scalers['static'].transform(
                self.data['static']
            )

        # Normalize targets
        if self.data['targets'].shape[0] > 0:
            normalized_data['targets'] = self.scalers['targets'].transform(
                self.data['targets']
            )

        return normalized_data

    def _create_sequences(self) -> List[Dict[str, np.ndarray]]:
        """Create input-output sequences for training"""
        sequences = []
        n_samples = len(self.data['dynamic'])

        # Create sliding windows
        for i in range(self.sequence_length, n_samples - self.prediction_horizon):
            # Input sequence: dynamic features for past sequence_length days
            dynamic_seq = self.data['dynamic'][i-self.sequence_length:i, :]

            # Static features (same for all time steps in sequence)
            static_features = self.data['static'][i, :]  # Use current time step

            # Targets: predict next prediction_horizon days
            target_start = i
            target_end = i + self.prediction_horizon
            targets = self.data['targets'][target_start:target_end, :]

            # Combine dynamic and static features
            # Repeat static features for each time step
            static_seq = np.tile(static_features, (self.sequence_length, 1))

            # Combine features
            features = np.concatenate([dynamic_seq, static_seq], axis=1)

            sequence = {
                'features': features.astype(np.float32),
                'targets': targets.astype(np.float32),
                'basin_id': self.basin_id,
                'time_index': i,
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon
            }

            sequences.append(sequence)

        return sequences

    def __len__(self) -> int:
        """Return number of sequences"""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence by index"""
        sequence = self.sequences[idx]

        # Convert to torch tensors
        features = torch.from_numpy(sequence['features'])
        targets = torch.from_numpy(sequence['targets'])

        # For multi-task learning, we need to separate targets
        # First column is streamflow, second is evapotranspiration
        streamflow_target = targets[:, 0:1]  # Keep as 2D tensor
        et_target = targets[:, 1:2]  # Keep as 2D tensor

        return {
            'features': features,
            'streamflow': streamflow_target,
            'evapotranspiration': et_target,
            'basin_id': sequence['basin_id'],
            'time_index': sequence['time_index']
        }

    def get_data_stats(self) -> Dict[str, Any]:
        """Get statistics about the dataset"""
        stats = {
            'basin_id': self.basin_id,
            'num_sequences': len(self),
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'dynamic_features': self.dynamic_features,
            'static_features': self.static_features,
            'target_features': self.target_features,
            'num_dynamic_features': len(self.dynamic_features),
            'num_static_features': len(self.static_features),
            'num_target_features': len(self.target_features),
            'total_features': len(self.dynamic_features) + len(self.static_features)
        }

        # Add mean and std of original data
        if not self.normalize:
            stats['dynamic_mean'] = np.mean(self.data['dynamic'], axis=0).tolist()
            stats['dynamic_std'] = np.std(self.data['dynamic'], axis=0).tolist()
            stats['target_mean'] = np.mean(self.data['targets'], axis=0).tolist()
            stats['target_std'] = np.std(self.data['targets'], axis=0).tolist()

        return stats


class MultiBasinDataset(Dataset):
    """
    Dataset combining multiple hydrological basins
    Supports multi-task learning across different basins
    """

    def __init__(self,
                 nc_files: List[str],
                 basin_ids: List[str],
                 sequence_length: int = 365,
                 prediction_horizon: int = 1,
                 dynamic_features: List[str] = None,
                 static_features: List[str] = None,
                 target_features: List[str] = None,
                 normalize: bool = True,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 mode: str = 'train'):
        """
        Initialize multi-basin dataset

        Args:
            nc_files: List of NetCDF file paths
            basin_ids: List of basin identifiers
            sequence_length: Length of input sequence
            prediction_horizon: Number of days to predict ahead
            dynamic_features: List of dynamic feature names
            static_features: List of static feature names
            target_features: List of target feature names
            normalize: Whether to normalize features
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            mode: 'train', 'val', or 'test'
        """
        self.nc_files = nc_files
        self.basin_ids = basin_ids
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.mode = mode

        # Validate inputs
        if len(nc_files) != len(basin_ids):
            raise ValueError("Number of nc_files must match number of basin_ids")

        # Load individual basin datasets
        self.basin_datasets = []
        self.all_sequences = []
        self.failed_basins = []  # track failed basins for debugging

        for nc_file, basin_id in zip(nc_files, basin_ids):
            try:
                basin_dataset = HydroBasinDataset(
                    nc_file=nc_file,
                    basin_id=basin_id,
                    sequence_length=sequence_length,
                    prediction_horizon=prediction_horizon,
                    dynamic_features=dynamic_features,
                    static_features=static_features,
                    target_features=target_features,
                    normalize=normalize,
                    scalers=None
                )

                self.basin_datasets.append(basin_dataset)

                # Split sequences for this basin
                basin_sequences = self._split_basin_sequences(basin_dataset)
                self.all_sequences.extend(basin_sequences)

                logger.info(f"Loaded basin {basin_id}: {len(basin_sequences)} sequences for {mode}")

            except Exception as e:
                # Record failure and continue with next basin
                error_msg = f"Failed to load basin {basin_id} from {nc_file}: {str(e)}"
                logger.error(error_msg)
                self.failed_basins.append((basin_id, nc_file, str(e)))
                continue

        if not self.all_sequences:
            # No basins loaded successfully, raise detailed error
            error_lines = ["No basins were successfully loaded. Failures:"]
            for basin_id, nc_file, err in self.failed_basins:
                error_lines.append(f"  - {basin_id} ({nc_file}): {err}")
            raise ValueError("\n".join(error_lines))

        logger.info(f"MultiBasinDataset initialized with {len(self.all_sequences)} total sequences")
        if self.failed_basins:
            logger.warning(f"Skipped {len(self.failed_basins)} basins due to errors")

    def _split_basin_sequences(self, basin_dataset: HydroBasinDataset) -> List[Dict[str, Any]]:
        """Split sequences for a single basin into train/val/test"""
        all_sequences = basin_dataset.sequences
        n_sequences = len(all_sequences)

        # Calculate split indices
        train_end = int(n_sequences * self.train_ratio)
        val_end = train_end + int(n_sequences * self.val_ratio)

        # Split sequences
        if self.mode == 'train':
            return all_sequences[:train_end]
        elif self.mode == 'val':
            return all_sequences[train_end:val_end]
        elif self.mode == 'test':
            return all_sequences[val_end:]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def __len__(self) -> int:
        """Return total number of sequences"""
        return len(self.all_sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence by index"""
        sequence = self.all_sequences[idx]

        # Convert to torch tensors
        features = torch.from_numpy(sequence['features'])
        targets = torch.from_numpy(sequence['targets'])

        # Separate targets for multi-task learning
        streamflow_target = targets[:, 0:1]
        et_target = targets[:, 1:2]

        # Get basin index for embedding
        basin_id = sequence['basin_id']
        basin_idx = self.basin_ids.index(basin_id)

        return {
            'features': features,
            'streamflow': streamflow_target,
            'evapotranspiration': et_target,
            'basin_id': basin_id,
            'basin_idx': torch.tensor(basin_idx, dtype=torch.long),
            'time_index': sequence['time_index']
        }

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the multi-basin dataset"""
        stats = {
            'num_basins': len(self.basin_datasets),
            'total_sequences': len(self),
            'mode': self.mode,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'basin_ids': self.basin_ids
        }

        # Get stats from each basin dataset
        basin_stats = []
        for dataset in self.basin_datasets:
            basin_stats.append(dataset.get_data_stats())

        stats['basin_details'] = basin_stats

        return stats