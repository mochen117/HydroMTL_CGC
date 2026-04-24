"""
Dataset classes for multi-task hydrological modeling
Handles multiple basins, multiple tasks, temporal sequences, and correct missing value imputation.
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
    def __init__(self,
                 nc_file: str,
                 basin_id: str,
                 sequence_length: int = 365,
                 prediction_horizon: int = 1,
                 dynamic_features: List[str] = None,
                 static_features: List[str] = None,
                 categorical_features: List[str] = None,
                 target_features: List[str] = None,
                 normalize: bool = True,
                 scalers: Dict[str, Any] = None):
        
        self.nc_file = nc_file
        self.basin_id = basin_id
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize

        self.dynamic_features = dynamic_features or [
            'total_precipitation', 'temperature', 'specific_humidity',
            'shortwave_radiation', 'potential_energy'
        ]
        
        self.static_features = static_features or [
            'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest',
            'lai_max', 'lai_diff', 'soil_porosity', 'soil_conductivity',
            'max_water_content', 'geol_porosity', 'geol_permeability'
        ]
        
        self.categorical_features = categorical_features or []
        
        self.target_features = target_features or ['streamflow', 'evapotranspiration']

        self.data = self._load_and_prepare_data()

        if scalers is None and normalize:
            logger.warning(f"Fitting scalers on dataset for basin {basin_id}. "
                           "Ensure this is only done on the training set to prevent data leakage.")
            self.scalers = self._initialize_scalers()
        else:
            self.scalers = self._process_scalers(scalers)

        if normalize and self.scalers is not None:
            self.data = self._normalize_data()
        elif normalize and self.scalers is None:
            raise ValueError("normalize=True but no scalers provided or fitted.")

        self.sequences = self._create_sequences()

    def _process_scalers(self, scalers: Dict[str, Any]) -> Dict[str, Any]:
        if scalers is None:
            return None
            
        processed = {
            'dynamic': scalers.get('dynamic'),
            'static': scalers.get('static')
        }
        
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
                raise ValueError("Unexpected target scaler format.")
        return processed

    def _load_and_prepare_data(self) -> Dict[str, np.ndarray]:
        ds = xr.open_dataset(self.nc_file)
        time_len = len(ds.time)

        dynamic_data = {}
        for feature in self.dynamic_features:
            if feature in ds:
                dynamic_data[feature] = ds[feature].values.astype(np.float32)
            else:
                dynamic_data[feature] = np.full(time_len, np.nan, dtype=np.float32)

        static_data = {}
        for feature in self.static_features:
            if feature in ds:
                static_data[feature] = np.full(time_len, float(ds[feature].values), dtype=np.float32)
            else:
                static_data[feature] = np.full(time_len, np.nan, dtype=np.float32)

        categorical_data = {}
        for feature in self.categorical_features:
            if feature in ds:
                arr = ds[feature].values
                # Fill NaN with 0 for unknown/missing category
                arr = np.nan_to_num(arr, nan=0.0).astype(np.int64)
                categorical_data[feature] = np.full(time_len, arr, dtype=np.int64)
            else:
                categorical_data[feature] = np.zeros(time_len, dtype=np.int64)

        target_data = {}
        for feature in self.target_features:
            if feature in ds:
                # CRITICAL: Keep NaNs in targets, do not interpolate
                target_data[feature] = ds[feature].values.astype(np.float32)
            else:
                target_data[feature] = np.full(time_len, np.nan, dtype=np.float32)

        time_data = ds.time.values
        ds.close()

        data = {
            'dynamic': np.column_stack([dynamic_data[f] for f in self.dynamic_features]) if self.dynamic_features else np.empty((time_len, 0)),
            'static': np.column_stack([static_data[f] for f in self.static_features]) if self.static_features else np.empty((time_len, 0)),
            'categorical': np.column_stack([categorical_data[f] for f in self.categorical_features]) if self.categorical_features else np.empty((time_len, 0)),
            'targets': np.column_stack([target_data[f] for f in self.target_features]) if self.target_features else np.empty((time_len, 0)),
            'time': time_data
        }
        return data

    def _initialize_scalers(self) -> Dict[str, Any]:
        scalers = {}
        
        # StandardScaler ignores NaNs during fit natively
        if self.data['dynamic'].shape[1] > 0:
            scalers['dynamic'] = StandardScaler().fit(self.data['dynamic'])
        else:
            scalers['dynamic'] = None
            
        if self.data['static'].shape[1] > 0:
            scalers['static'] = StandardScaler().fit(self.data['static'])
        else:
            scalers['static'] = None

        target_scalers = []
        for i in range(self.data['targets'].shape[1]):
            col = self.data['targets'][:, i:i+1]
            valid_mask = ~np.isnan(col)
            s = StandardScaler()
            if valid_mask.any():
                s.fit(col[valid_mask].reshape(-1, 1))
            target_scalers.append(s)
            
        scalers['targets'] = target_scalers
        return scalers

    def _normalize_data(self) -> Dict[str, np.ndarray]:
            normalized = self.data.copy()
            
            if self.data['dynamic'].shape[1] > 0 and self.scalers['dynamic'] is not None:
                dyn_norm = self.scalers['dynamic'].transform(self.data['dynamic'])
                normalized['dynamic'] = np.nan_to_num(dyn_norm, nan=0.0, posinf=0.0, neginf=0.0)
                
            if self.data['static'].shape[1] > 0 and self.scalers['static'] is not None:
                stat_norm = self.scalers['static'].transform(self.data['static'])
                normalized['static'] = np.nan_to_num(stat_norm, nan=0.0, posinf=0.0, neginf=0.0)
                
            if self.data['targets'].shape[1] > 0 and self.scalers['targets'] is not None:
                norm_targets = np.full_like(self.data['targets'], np.nan, dtype=np.float32)
                for i, scaler in enumerate(self.scalers['targets']):
                    col = self.data['targets'][:, i]
                    valid_mask = ~np.isnan(col)
                    if valid_mask.any():
                        norm_targets[valid_mask, i] = scaler.transform(col[valid_mask].reshape(-1, 1)).flatten()
                normalized['targets'] = norm_targets
                
            return normalized

    def _create_sequences(self) -> List[Dict[str, np.ndarray]]:
        sequences = []
        n_samples = len(self.data['dynamic'])
        
        for i in range(self.sequence_length, n_samples - self.prediction_horizon):
            dynamic_seq = self.data['dynamic'][i - self.sequence_length:i, :]
            static_features = self.data['static'][i, :]
            categorical_features = self.data['categorical'][i, :]
            
            target_start = i
            target_end = i + self.prediction_horizon
            targets = self.data['targets'][target_start:target_end, :]

            static_seq = np.tile(static_features, (self.sequence_length, 1))
            cat_seq = np.tile(categorical_features, (self.sequence_length, 1))
            
            features = np.concatenate([dynamic_seq, static_seq], axis=1) if static_features.size > 0 else dynamic_seq

            sequences.append({
                'features': features.astype(np.float32),
                'categorical_features': cat_seq.astype(np.int64),
                'targets': targets.astype(np.float32),
                'basin_id': self.basin_id,
                'time_index': i
            })
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        return {
            'features': torch.from_numpy(seq['features']),
            'categorical_features': torch.from_numpy(seq['categorical_features']),
            'streamflow': torch.from_numpy(seq['targets'][:, 0:1]),
            'evapotranspiration': torch.from_numpy(seq['targets'][:, 1:2]),
            'basin_id': seq['basin_id'],
            'time_index': seq['time_index']
        }

    @staticmethod
    def _get_raw_data(nc_file: str,
                      basin_id: str,
                      dynamic_features: List[str],
                      static_features: List[str],
                      categorical_features: List[str],
                      target_features: List[str]) -> Dict[str, np.ndarray]:
        ds = xr.open_dataset(nc_file)
        time_len = len(ds.time)
        
        dynamic = []
        for f in dynamic_features:
            arr = ds[f].values if f in ds else np.full(time_len, np.nan)
            dynamic.append(arr)
            
        static = []
        for f in static_features:
            arr = np.full(time_len, float(ds[f].values)) if f in ds else np.full(time_len, np.nan)
            static.append(arr)
            
        categorical = []
        for f in categorical_features:
            if f in ds:
                arr = np.nan_to_num(ds[f].values, nan=0.0).astype(np.int64)
                categorical.append(np.full(time_len, arr))
            else:
                categorical.append(np.zeros(time_len, dtype=np.int64))

        targets = []
        for f in target_features:
            arr = ds[f].values if f in ds else np.full(time_len, np.nan)
            targets.append(arr)
            
        time = ds.time.values
        ds.close()

        return {
            'dynamic': np.column_stack(dynamic) if dynamic else np.empty((time_len, 0)),
            'static': np.column_stack(static) if static else np.empty((time_len, 0)),
            'categorical': np.column_stack(categorical) if categorical else np.empty((time_len, 0)),
            'targets': np.column_stack(targets) if targets else np.empty((time_len, 0)),
            'time': time,
            'basin_id': basin_id
        }


class MultiBasinDataset(Dataset):
    def __init__(self,
                 basin_datasets: List[HydroBasinDataset],
                 basin_sequences_indices: List[List[int]]):
        self.basin_datasets = basin_datasets
        self.basin_ids = [ds.basin_id for ds in basin_datasets]

        self.index_map = []
        for ds_idx, indices in enumerate(basin_sequences_indices):
            for seq_idx in indices:
                self.index_map.append((ds_idx, seq_idx))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ds_idx, seq_idx = self.index_map[idx]
        basin_ds = self.basin_datasets[ds_idx]
        seq = basin_ds.sequences[seq_idx]

        return {
            'features': torch.from_numpy(seq['features']),
            'categorical_features': torch.from_numpy(seq['categorical_features']),
            'streamflow': torch.from_numpy(seq['targets'][:, 0:1]),
            'evapotranspiration': torch.from_numpy(seq['targets'][:, 1:2]),
            'basin_id': seq['basin_id'],
            'basin_idx': torch.tensor(ds_idx, dtype=torch.long),
            'time_index': seq['time_index']
        }


def build_multi_basin_datasets(
        nc_files: List[str],
        basin_ids: List[str],
        sequence_length: int = 365,
        prediction_horizon: int = 1,
        dynamic_features: List[str] = None,
        static_features: List[str] = None,
        categorical_features: List[str] = None,
        target_features: List[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        train_period: Optional[Tuple[str, str]] = None,
        val_period: Optional[Tuple[str, str]] = None,
        test_period: Optional[Tuple[str, str]] = None,
        ) -> Tuple[MultiBasinDataset, MultiBasinDataset, MultiBasinDataset, List[Dict[str, Any]]]:
    
    if len(nc_files) != len(basin_ids):
        raise ValueError("Number of nc_files must match number of basin_ids")

    dynamic_features = dynamic_features or []
    static_features = static_features or []
    categorical_features = categorical_features or []
    target_features = target_features or ['streamflow', 'evapotranspiration']

    raw_data_list = []
    valid_nc_files = []
    valid_basin_ids = []

    total_basins = len(nc_files)
    for idx, (nc_file, basin_id) in enumerate(tqdm(zip(nc_files, basin_ids), total=total_basins, desc="Loading basins")):
        raw = HydroBasinDataset._get_raw_data(
            nc_file, basin_id, dynamic_features, static_features, categorical_features, target_features
        )
        
        n_total = len(raw['time'])
        first_seq_start = sequence_length
        last_seq_start = n_total - prediction_horizon - 1
        
        if first_seq_start > last_seq_start:
            logger.warning(f"Basin {basin_id} has insufficient data (len={n_total}). Skipping.")
            continue
            
        raw_data_list.append(raw)
        valid_nc_files.append(nc_file)
        valid_basin_ids.append(basin_id)

    nc_files = valid_nc_files
    basin_ids = valid_basin_ids

    all_train_abs = []
    all_val_abs = []
    all_test_abs = []

    for raw, basin_id in zip(raw_data_list, basin_ids):
        time = raw['time']
        n_total = len(time)
        first_seq_start = sequence_length
        last_seq_start = n_total - prediction_horizon - 1

        if train_period is not None and val_period is not None and test_period is not None:
            train_start, train_end = np.datetime64(train_period[0]), np.datetime64(train_period[1])
            val_start, val_end = np.datetime64(val_period[0]), np.datetime64(val_period[1])
            test_start, test_end = np.datetime64(test_period[0]), np.datetime64(test_period[1])

            seq_dates = time[first_seq_start:last_seq_start+1]

            train_mask = (seq_dates >= train_start) & (seq_dates <= train_end)
            val_mask   = (seq_dates >= val_start)   & (seq_dates <= val_end)
            test_mask  = (seq_dates >= test_start)  & (seq_dates <= test_end)

            train_abs = np.where(train_mask)[0] + first_seq_start
            val_abs   = np.where(val_mask)[0]   + first_seq_start
            test_abs  = np.where(test_mask)[0]  + first_seq_start

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
    all_basin_scalers = []

    print() 

    for i, (raw, basin_id, nc_file) in enumerate(tqdm(zip(raw_data_list, basin_ids, nc_files), total=len(basin_ids), desc="Building sequences & scaling")):
        train_abs = all_train_abs[i]
        
        if len(train_abs) == 0:
            logger.warning(f"Basin {basin_id} has no valid training sequences in specified period. Defaulting scaler to None.")
            scaler_dynamic = None
            scaler_static = None
            target_scalers = [None for _ in range(raw['targets'].shape[1])]
        else:
            first_train_start = max(0, train_abs[0] - sequence_length)
            last_train_end = min(len(raw['time']), train_abs[-1] + prediction_horizon)
            train_slice = slice(first_train_start, last_train_end)

            scaler_dynamic = StandardScaler()
            if raw['dynamic'].shape[1] > 0:
                scaler_dynamic.fit(raw['dynamic'][train_slice])
            else:
                scaler_dynamic = None

            scaler_static = StandardScaler()
            if raw['static'].shape[1] > 0:
                scaler_static.fit(raw['static'][train_slice])
            else:
                scaler_static = None

            target_scalers = []
            for t_idx in range(raw['targets'].shape[1]):
                s = StandardScaler()
                col = raw['targets'][train_slice, t_idx:t_idx+1]
                valid_mask = ~np.isnan(col)
                if valid_mask.any():
                    s.fit(col[valid_mask].reshape(-1, 1))
                target_scalers.append(s)

        scalers = {
            'dynamic': scaler_dynamic,
            'static': scaler_static,
            'targets': target_scalers
        }
        
        task_scaler_dict = {feat: target_scalers[idx] for idx, feat in enumerate(target_features)}
        all_basin_scalers.append(task_scaler_dict)

        ds = HydroBasinDataset(
            nc_file=nc_file,
            basin_id=basin_id,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            dynamic_features=dynamic_features,
            static_features=static_features,
            categorical_features=categorical_features,
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

    return train_dataset, val_dataset, test_dataset, all_basin_scalers