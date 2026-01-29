"""
IO utilities for Hydro Data Processor
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, List, Union
import pickle
import logging

logger = logging.getLogger(__name__)


def read_camels_file(filepath: str, **kwargs) -> pd.DataFrame:
    """Read CAMELS format file"""
    try:
        return pd.read_csv(filepath, **kwargs)
    except Exception as e:
        logger.warning(f"Error reading CAMELS file {filepath}: {e}")
        # Return empty DataFrame with common columns
        return pd.DataFrame(columns=['basin_id', 'date', 'streamflow'])


def build_file_path(base_path: str, basin_id: str, file_type: str) -> str:
    """Build file path for basin data"""
    path = Path(base_path) / basin_id / f"{file_type}.csv"
    return str(path)


def save_json(data: Dict, filepath: str, indent: int = 2):
    """Save data as JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: str):
    """Save data as pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Saved pickle to {filepath}")


def load_pickle(filepath: str) -> Any:
    """Load pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_dataset(dataset, filepath: str, format: str = 'csv', **kwargs):
    """Save dataset to file"""
    if format == 'csv':
        dataset.to_csv(filepath, **kwargs)
    elif format == 'parquet':
        dataset.to_parquet(filepath, **kwargs)
    elif format == 'hdf5':
        dataset.to_hdf(filepath, key='data', **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Saved dataset to {filepath} ({format})")
