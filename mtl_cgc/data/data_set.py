"""
Dataset classes for multi-task hydrological modeling
Handles multiple basins, multiple tasks, and temporal sequences
"""

import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
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
            scalers: Pre-trained scalers for normalization. Expected format:
                     {'dynamic': StandardScaler,
                      'static': StandardScaler,
                      'targets': List[StandardScaler]} where the list length equals
                     number of target features. If None and normalize=True,
                     scalers will be fitted on the entire dataset (not recommended).
        """
        self.nc_file = nc_file
        self.basin_id = basin_id
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize

        if dynamic_features is None:
            self.dynamic_features = [
                'total_precipitation',
                'temperature',
                'specific_humidity',
                'shortwave_radiation',
                'potential_energy'
            ]
        else:
            self.dynamic_features = dynamic_features

        if static_features is None:
            self.static_features = [
                'elev_mean',
                'slope_mean',
                'area_gages2',
                'frac_forest',
                'lai_max',
                'lai_diff',
                'soil_porosity',
                'soil_conductivity',
                'max_water_content',
                'geol_porosity',
                'geol_permeability'
            ]
        else:
            self.static_features = static_features

        if target_features is None:
            self.target_features = ['streamflow', 'evapotranspiration']
        else:
            self.target_features = target_features

        self.data = self._load_and_prepare_data()

        if scalers is None and normalize:
            logger.warning(f"Fitting scalers on entire dataset for basin {basin_id}. "
                           "This may cause data leakage. Prefer passing pre-trained scalers.")
            self.scalers = self._initialize_scalers()
        else:
            self.scalers = self._process_scalers(scalers)

        if normalize and self.scalers is not None:
            self.data = self._normalize_data()
        elif normalize and self.scalers is None:
            raise ValueError("normalize=True but no scalers provided or fitted.")

        self.sequences = self._create_sequences()
        logger.info(f"Loaded basin {basin_id}: {len(self.sequences)} sequences")

    def _process_scalers(self, scalers: Dict[str, Any]) -> Dict[str, Any]:
        processed = {}
        processed['dynamic'] = scalers.get('dynamic')
        processed['static'] = scalers.get('static')
        target_scalers = scalers.get('targets')
        if target_scalers is None:
            processed['targets'] = None
        elif isinstance(target_scalers, list):
            processed['targets'] = target_scalers
        else:
            n_targets = len(self.target_features)
            if hasattr(target_scalers, 'mean_') and target_scalers.mean_.shape[0] == n_targets:
                split_scalers = []
                for i in range(n_targets):
                    s = StandardScaler()
                    s.mean_ = np.array([target_scalers.mean_[i]])
                    s.scale_ = np.array([target_scalers.scale_[i]])
                    if hasattr(target_scalers, 'var_'):
                        s.var_ = np.array([target_scalers.var_[i]])
                    s.n_samples_seen_ = target_scalers.n_samples_seen_
                    split_scalers.append(s)
                processed['targets'] = split_scalers
            else:
                raise ValueError(f"Unexpected target scaler format. Expected list of scalers or single scaler "
                                 f"with {n_targets} features, got {target_scalers.mean_.shape[0] if hasattr(target_scalers, 'mean_') else 'unknown'}.")
        return processed

    def _load_and_prepare_data(self) -> Dict[str, np.ndarray]:
        ds = xr.open_dataset(self.nc_file)

        dynamic_data = {}
        for feature in self.dynamic_features:
            if feature in ds:
                data = ds[feature].values
                if np.any(np.isnan(data)):
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

        static_data = {}
        for feature in self.static_features:
            if feature in ds:
                static_data[feature] = np.full(
                    len(ds.time),
                    float(ds[feature].values)
                )
            else:
                logger.warning(f"Static feature {feature} not found in {self.nc_file}")
                static_data[feature] = np.zeros(len(ds.time))

        target_data = {}
        for feature in self.target_features:
            if feature in ds:
                data = ds[feature].values
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

        time_data = ds.time.values
        ds.close()

        data = {
            'dynamic': np.column_stack([dynamic_data[f] for f in self.dynamic_features]),
            'static': np.column_stack([static_data[f] for f in self.static_features]),
            'targets': np.column_stack([target_data[f] for f in self.target_features]),
            'time': time_data,
            'basin_id': self.basin_id,
            'dynamic_feature_names': self.dynamic_features,
            'static_feature_names': self.static_features,
            'target_feature_names': self.target_features
        }
        return data

    def _initialize_scalers(self) -> Dict[str, Any]:
        scalers = {}
        scalers['dynamic'] = StandardScaler().fit(self.data['dynamic'])
        scalers['static'] = StandardScaler().fit(self.data['static'])
        target_scalers = []
        for i in range(self.data['targets'].shape[1]):
            col = self.data['targets'][:, i:i+1]
            target_scalers.append(StandardScaler().fit(col))
        scalers['targets'] = target_scalers
        return scalers

    def _normalize_data(self) -> Dict[str, np.ndarray]:
        normalized = self.data.copy()
        if self.data['dynamic'].shape[0] > 0 and self.scalers['dynamic'] is not None:
            normalized['dynamic'] = self.scalers['dynamic'].transform(self.data['dynamic'])
        if self.data['static'].shape[0] > 0 and self.scalers['static'] is not None:
            normalized['static'] = self.scalers['static'].transform(self.data['static'])
        if self.data['targets'].shape[0] > 0 and self.scalers['targets'] is not None:
            target_scalers = self.scalers['targets']
            normalized_targets = np.zeros_like(self.data['targets'])
            for i, scaler in enumerate(target_scalers):
                col = self.data['targets'][:, i:i+1]
                normalized_targets[:, i:i+1] = scaler.transform(col)
            normalized['targets'] = normalized_targets
        return normalized

    def _create_sequences(self) -> List[Dict[str, np.ndarray]]:
        sequences = []
        n_samples = len(self.data['dynamic'])
        for i in range(self.sequence_length, n_samples - self.prediction_horizon):
            dynamic_seq = self.data['dynamic'][i - self.sequence_length:i, :]
            static_features = self.data['static'][i, :]
            target_start = i
            target_end = i + self.prediction_horizon
            targets = self.data['targets'][target_start:target_end, :]

            static_seq = np.tile(static_features, (self.sequence_length, 1))
            features = np.concatenate([dynamic_seq, static_seq], axis=1)

            sequences.append({
                'features': features.astype(np.float32),
                'targets': targets.astype(np.float32),
                'basin_id': self.basin_id,
                'time_index': i,
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon
            })
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        features = torch.from_numpy(seq['features'])
        targets = torch.from_numpy(seq['targets'])
        # Keep 2D shape (prediction_horizon, 1)
        streamflow_target = targets[:, 0:1]
        evapotranspiration_target = targets[:, 1:2]

        return {
            'features': features,
            'streamflow': streamflow_target,
            'evapotranspiration': evapotranspiration_target,
            'basin_id': seq['basin_id'],
            'time_index': seq['time_index']
        }

    def get_data_stats(self) -> Dict[str, Any]:
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
        if not self.normalize:
            stats['dynamic_mean'] = np.mean(self.data['dynamic'], axis=0).tolist()
            stats['dynamic_std'] = np.std(self.data['dynamic'], axis=0).tolist()
            stats['target_mean'] = np.mean(self.data['targets'], axis=0).tolist()
            stats['target_std'] = np.std(self.data['targets'], axis=0).tolist()
        return stats

    @staticmethod
    def _get_raw_data(nc_file: str,
                      basin_id: str,
                      dynamic_features: List[str],
                      static_features: List[str],
                      target_features: List[str]) -> Dict[str, np.ndarray]:
        ds = xr.open_dataset(nc_file)
        dynamic = []
        for f in dynamic_features:
            if f in ds:
                arr = ds[f].values
                if np.any(np.isnan(arr)):
                    mask = np.isnan(arr)
                    arr[mask] = np.interp(np.where(mask)[0], np.where(~mask)[0], arr[~mask])
                dynamic.append(arr)
            else:
                dynamic.append(np.zeros(len(ds.time)))
        static = []
        for f in static_features:
            if f in ds:
                static.append(np.full(len(ds.time), float(ds[f].values)))
            else:
                static.append(np.zeros(len(ds.time)))
        targets = []
        for f in target_features:
            if f in ds:
                arr = ds[f].values
                if np.any(np.isnan(arr)):
                    mask = np.isnan(arr)
                    arr[mask] = np.interp(np.where(mask)[0], np.where(~mask)[0], arr[~mask])
                targets.append(arr)
            else:
                targets.append(np.zeros(len(ds.time)))
        time = ds.time.values
        ds.close()

        return {
            'dynamic': np.column_stack(dynamic),
            'static': np.column_stack(static),
            'targets': np.column_stack(targets),
            'time': time,
            'basin_id': basin_id
        }


class MultiBasinDataset(Dataset):
    """
    Dataset combining multiple hydrological basins.
    Sequences are pre-filtered according to the desired mode (train/val/test).
    """

    def __init__(self,
                 basin_datasets: List[HydroBasinDataset],
                 basin_sequences_indices: List[List[int]]):
        self.basin_datasets = basin_datasets
        self.basin_ids = [ds.basin_id for ds in basin_datasets]

        self.index_map = []
        for ds_idx, indices in enumerate(basin_sequences_indices):
            for seq_idx in indices:
                self.index_map.append((ds_idx, seq_idx))

        logger.info(f"MultiBasinDataset created with {len(self)} sequences.")

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ds_idx, seq_idx = self.index_map[idx]
        basin_ds = self.basin_datasets[ds_idx]
        seq = basin_ds.sequences[seq_idx]

        features = torch.from_numpy(seq['features'])
        targets = torch.from_numpy(seq['targets'])
        # Keep 2D shape (prediction_horizon, 1)
        streamflow_target = targets[:, 0:1]
        evapotranspiration_target = targets[:, 1:2]

        basin_id = seq['basin_id']
        basin_idx = self.basin_ids.index(basin_id)

        return {
            'features': features,
            'streamflow': streamflow_target,
            'evapotranspiration': evapotranspiration_target,
            'basin_id': basin_id,
            'basin_idx': torch.tensor(basin_idx, dtype=torch.long),
            'time_index': seq['time_index']
        }

    def get_dataset_stats(self) -> Dict[str, Any]:
        stats = {
            'num_basins': len(self.basin_datasets),
            'total_sequences': len(self),
            'basin_ids': self.basin_ids
        }
        basin_stats = [ds.get_data_stats() for ds in self.basin_datasets]
        stats['basin_details'] = basin_stats
        return stats


def build_multi_basin_datasets(
        nc_files: List[str],
        basin_ids: List[str],
        sequence_length: int = 365,
        prediction_horizon: int = 1,
        dynamic_features: List[str] = None,
        static_features: List[str] = None,
        target_features: List[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        train_period: Optional[Tuple[str, str]] = None,
        val_period: Optional[Tuple[str, str]] = None,
        test_period: Optional[Tuple[str, str]] = None,
        ) -> Tuple[MultiBasinDataset, MultiBasinDataset, MultiBasinDataset]:
    """
    Build train, validation, and test MultiBasinDatasets with correct normalization.
    If train_period/val_period/test_period are provided, they override train_ratio/val_ratio.
    """
    if len(nc_files) != len(basin_ids):
        raise ValueError("Number of nc_files must match number of basin_ids")

    if dynamic_features is None:
        dynamic_features = [
            'total_precipitation', 'temperature', 'specific_humidity',
            'shortwave_radiation', 'potential_energy'
        ]
    if static_features is None:
        static_features = [
            'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest',
            'lai_max', 'lai_diff', 'soil_porosity', 'soil_conductivity',
            'max_water_content', 'geol_porosity', 'geol_permeability'
        ]
    if target_features is None:
        target_features = ['streamflow', 'evapotranspiration']

    raw_data_list = []
    all_train_abs = []
    all_val_abs = []
    all_test_abs = []

    total_basins = len(nc_files)
    for idx, (nc_file, basin_id) in enumerate(tqdm(zip(nc_files, basin_ids), total=total_basins, desc="Loading basins")):
        raw = HydroBasinDataset._get_raw_data(
            nc_file, basin_id, dynamic_features, static_features, target_features
        )
        raw_data_list.append(raw)

        time = raw['time']
        n_total = len(time)
        first_seq_start = sequence_length
        last_seq_start = n_total - prediction_horizon - 1
        if first_seq_start > last_seq_start:
            raise ValueError(f"Basin {basin_id} has insufficient data: n_total={n_total}, "
                             f"sequence_length={sequence_length}, prediction_horizon={prediction_horizon}")

        if train_period is not None and val_period is not None and test_period is not None:
            train_start = np.datetime64(train_period[0])
            train_end   = np.datetime64(train_period[1])
            val_start   = np.datetime64(val_period[0])
            val_end     = np.datetime64(val_period[1])
            test_start  = np.datetime64(test_period[0])
            test_end    = np.datetime64(test_period[1])

            seq_dates = time[first_seq_start:last_seq_start+1]

            train_mask = (seq_dates >= train_start) & (seq_dates <= train_end)
            val_mask   = (seq_dates >= val_start)   & (seq_dates <= val_end)
            test_mask  = (seq_dates >= test_start)  & (seq_dates <= test_end)

            train_abs = np.where(train_mask)[0] + first_seq_start
            val_abs   = np.where(val_mask)[0]   + first_seq_start
            test_abs  = np.where(test_mask)[0]  + first_seq_start

            if len(train_abs) == 0:
                logger.warning(f"No training sequences found for basin {basin_id} in period {train_period}")
            if len(val_abs) == 0:
                logger.warning(f"No validation sequences found for basin {basin_id} in period {val_period}")
            if len(test_abs) == 0:
                logger.warning(f"No test sequences found for basin {basin_id} in period {test_period}")

        else:
            n_seq = last_seq_start - first_seq_start + 1
            train_end_abs = first_seq_start + int(n_seq * train_ratio)
            val_end_abs   = train_end_abs + int(n_seq * val_ratio)
            train_abs = list(range(first_seq_start, train_end_abs))
            val_abs   = list(range(train_end_abs, val_end_abs))
            test_abs  = list(range(val_end_abs, last_seq_start + 1))

        all_train_abs.append(train_abs)
        all_val_abs.append(val_abs)
        all_test_abs.append(test_abs)

    basin_datasets = []
    train_seq_indices = []
    val_seq_indices = []
    test_seq_indices = []

    for i, (raw, basin_id, nc_file) in enumerate(zip(raw_data_list, basin_ids, nc_files)):
        train_abs = all_train_abs[i]
        if len(train_abs) == 0:
            raise ValueError(f"Basin {basin_id} has no training sequences.")

        first_train_start = train_abs[0] - sequence_length
        last_train_end = train_abs[-1] + prediction_horizon
        first_train_start = max(0, first_train_start)
        last_train_end = min(len(raw['time']), last_train_end)
        train_slice = slice(first_train_start, last_train_end)

        scaler_dynamic = StandardScaler().fit(raw['dynamic'][train_slice])
        scaler_static = StandardScaler().fit(raw['static'][train_slice])
        target_scalers = []
        for t_idx in range(raw['targets'].shape[1]):
            col = raw['targets'][train_slice, t_idx:t_idx+1]
            target_scalers.append(StandardScaler().fit(col))

        scalers = {
            'dynamic': scaler_dynamic,
            'static': scaler_static,
            'targets': target_scalers
        }

        ds = HydroBasinDataset(
            nc_file=nc_file,
            basin_id=basin_id,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            dynamic_features=dynamic_features,
            static_features=static_features,
            target_features=target_features,
            normalize=True,
            scalers=scalers
        )
        basin_datasets.append(ds)

        base = sequence_length
        train_seq = [s - base for s in train_abs if s - base >= 0]
        val_seq   = [s - base for s in all_val_abs[i]   if s - base >= 0]
        test_seq  = [s - base for s in all_test_abs[i]  if s - base >= 0]

        train_seq_indices.append(train_seq)
        val_seq_indices.append(val_seq)
        test_seq_indices.append(test_seq)

    train_dataset = MultiBasinDataset(basin_datasets, train_seq_indices)
    val_dataset   = MultiBasinDataset(basin_datasets, val_seq_indices)
    test_dataset  = MultiBasinDataset(basin_datasets, test_seq_indices)

    return train_dataset, val_dataset, test_dataset