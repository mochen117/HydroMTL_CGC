# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description: High-performance data pipeline for CAMELS dataset.
# Implements single-pass NetCDF extraction and numpy array yielding.
# ==============================================================================

import torch
import xarray as xr
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import sys
from tqdm import tqdm

from .data_sets import HydroDataset
from .data_scalers import HydroScaler

def impute_dynamic_features(dyn_array: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(dyn_array)
    df = df.interpolate(method='linear', limit_direction='both')
    df = df.ffill().bfill()
    df = df.fillna(0.0) 
    return df.values

def get_hydro_dataloaders(config: Dict, basin_ids: List[str], mode: str = "train") -> Tuple[DataLoader, DataLoader, Optional[DataLoader], HydroScaler]:
    data_cfg = config['data']
    data_root = Path(data_cfg['data_root'])
    
    print("Extracting time-series and static attributes from NetCDF archives...", flush=True)
    
    splits_to_load =[]
    if mode == "train":
        splits_to_load = [("train", data_cfg['train_period']), ("valid", data_cfg['val_period'])]
    elif mode == "test":
        splits_to_load =[("train", data_cfg['train_period']), ("test", data_cfg['test_period'])]
    
    extracted_data = {split_name: {'dyn': [], 's_num':[], 's_cat': [], 'y_dict': {t['name'].lower(): [] for t in data_cfg['targets']}} for split_name, _ in splits_to_load}
    
    pbar = tqdm(basin_ids, desc="Reading datasets", leave=True, file=sys.stdout, dynamic_ncols=True, bar_format="{l_bar}{bar:40}{r_bar}")
    
    for gid in pbar:
        ds = xr.open_dataset(data_root / f"gage_{gid}.nc")
        
        s_num_feats = []
        for v in data_cfg['static_features']:
            if v in ds: 
                s_num_feats.append(float(ds[v].values.item()))
            elif v in ds.attrs: 
                s_num_feats.append(float(ds.attrs[v]))
            else: 
                s_num_feats.append(np.nan)
                
        s_cat_feats =[]
        for v in data_cfg.get('categorical_static_features',[]):
            val = 0 
            if v in ds:
                raw_val = ds[v].values.item()
                if not pd.isna(raw_val) and not np.isnan(raw_val): 
                    val = int(raw_val)
            elif v in ds.attrs:
                try: 
                    val = int(ds.attrs[v])
                except (ValueError, TypeError): 
                    pass
            s_cat_feats.append(val)
        
        for split_name, period in splits_to_load:
            start_dt, end_dt = np.datetime64(period[0]), np.datetime64(period[1])
            ds_split = ds.sel(time=slice(start_dt, end_dt))
            
            raw_dyn = np.stack([ds_split[v].values for v in data_cfg['dynamic_features']], axis=-1)
            clean_dyn = impute_dynamic_features(raw_dyn)
            
            extracted_data[split_name]['dyn'].append(clean_dyn)
            extracted_data[split_name]['s_num'].append(np.array(s_num_feats, dtype=np.float32))
            extracted_data[split_name]['s_cat'].append(np.array(s_cat_feats, dtype=np.int64))
            
            for t in data_cfg['targets']:
                extracted_data[split_name]['y_dict'][t['name'].lower()].append(ds_split[t['name']].values)
                
        ds.close()
        
    print("Consolidating arrays into contiguous memory blocks...", flush=True)
    
    final_dicts = {}
    for split_name, _ in splits_to_load:
        dat = extracted_data[split_name]
        s_cat_stacked = np.stack(dat['s_cat']) if len(dat['s_cat'][0]) > 0 else None
        final_dicts[split_name] = {
            'dyn': np.stack(dat['dyn']),
            's_num': np.stack(dat['s_num']),
            's_cat': s_cat_stacked,
            'y_dict': {k: np.stack(v) for k, v in dat['y_dict'].items()}
        }
    
    bs = data_cfg.get('batch_size', 256)
    nw = data_cfg.get('num_workers', 16)
    pf = data_cfg.get('prefetch_factor', 4)

    train_ds = HydroDataset(final_dicts["train"], data_cfg, mode='train')
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=nw, prefetch_factor=pf)

    val_loader, test_loader = None, None
    if mode == "train":
        val_ds = HydroDataset(final_dicts["valid"], data_cfg, mode='valid', scaler=train_ds.scaler)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, prefetch_factor=pf)
    elif mode == "test":
        test_ds = HydroDataset(final_dicts["test"], data_cfg, mode='test', scaler=train_ds.scaler)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, prefetch_factor=pf)

    print("Dataloaders successfully initialized.\n", flush=True)
    return train_loader, val_loader, test_loader, train_ds.scaler