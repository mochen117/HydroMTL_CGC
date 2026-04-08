"""
Forcing data loader for NLDAS dataset.
Fixed to handle CAMELS forcing format with year, month, day columns.
Modified to load only 5 required meteorological forcing variables from paper Table 1.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from hydro_data_processor.config.settings import DataSourceConfig

logger = logging.getLogger(__name__)


class ForcingLoader:
    """Loader for NLDAS forcing data from CAMELS dataset."""

    REQUIRED_VARIABLES = [
        'total_precipitation',
        'temperature', 
        'specific_humidity',
        'shortwave_radiation',
        'potential_energy'
    ]

    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.data_source_path = config.data_source_path
        logger.info(f"ForcingLoader initialized for {self.data_source_path}")

    def load(self, gage_ids: List[str], **kwargs) -> Optional[pd.DataFrame]:
        """Load forcing data for given gage IDs."""
        all_data = []
        successful_gages = []

        for gage_id in gage_ids:
            try:
                huc2 = kwargs.get('huc2')
                if not huc2:
                    huc2 = self._find_huc_by_scanning(gage_id)
                
                if not huc2:
                    logger.debug(f"No HUC2 found for gage {gage_id}")
                    continue
                
                huc2 = str(huc2).zfill(2)
                
                gage_data = self._load_single_gage(gage_id, huc2)
                if gage_data is not None and not gage_data.empty:
                    gage_data = self._ensure_required_variables(gage_data)
                    all_data.append(gage_data)
                    successful_gages.append(gage_id)
                    logger.debug(f"Loaded forcing for gage {gage_id}")
            except Exception as e:
                logger.debug(f"Failed to load forcing for gage {gage_id}: {e}")

        if not all_data:
            logger.error("No forcing data loaded")
            return None

        logger.debug(f"Loaded forcing for {len(successful_gages)} gages")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        missing_vars = [var for var in self.REQUIRED_VARIABLES if var not in combined_df.columns]
        if missing_vars:
            logger.warning(f"Missing required forcing variables: {missing_vars}")
        
        return combined_df

    def _ensure_required_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        for var in self.REQUIRED_VARIABLES:
            if var not in df.columns:
                logger.debug(f"Required forcing variable {var} not found, filling with NaN")
                df[var] = np.nan
        return df

    def _load_single_gage(self, gage_id: str, huc2: str) -> Optional[pd.DataFrame]:
        if hasattr(self.config, 'get_file_path'):
            file_path = self.config.get_file_path(gage_id, huc2)
        else:
            file_path = self._build_file_path(gage_id, huc2)
        
        if not file_path.exists():
            file_path = self._find_alternative_path(gage_id, huc2)
            if not file_path or not file_path.exists():
                logger.debug(f"Forcing file not found for gage {gage_id} in HUC2 {huc2}")
                return None
        
        try:
            return self._read_forcing_file(file_path, gage_id)
        except Exception as e:
            logger.error(f"Error reading forcing file for gage {gage_id}: {e}")
            return None

    def _build_file_path(self, gage_id: str, huc2: str) -> Path:
        path = self.data_source_path
        if self.config.subdirectory:
            subdir = self.config.subdirectory.format(huc2=huc2)
            path = path / subdir
        if self.config.file_pattern:
            filename = self.config.file_pattern.format(basin_id=gage_id, huc2=huc2)
            path = path / filename
        return path

    def _find_alternative_path(self, gage_id: str, huc2: str) -> Optional[Path]:
        huc2_2digit = str(huc2).zfill(2)
        possible_paths = [
            self.data_source_path / "basin_mean_forcing" / huc2_2digit / f"{gage_id}_lump_nldas_forcing_leap.txt",
            self.data_source_path / huc2_2digit / f"{gage_id}_lump_nldas_forcing_leap.txt",
            self.data_source_path / "basin_mean_forcing" / huc2 / f"{gage_id}_lump_nldas_forcing_leap.txt",
            self.data_source_path / huc2 / f"{gage_id}_lump_nldas_forcing_leap.txt"
        ]
        for alt_path in possible_paths:
            if alt_path.exists():
                return alt_path
        return None

    def _find_huc_by_scanning(self, gage_id: str) -> Optional[str]:
        if not self.data_source_path.exists():
            return None
        if (self.data_source_path / "basin_mean_forcing").exists():
            base_dir = self.data_source_path / "basin_mean_forcing"
        else:
            base_dir = self.data_source_path
        for huc_dir in base_dir.iterdir():
            if huc_dir.is_dir() and huc_dir.name.isdigit():
                huc2_name = huc_dir.name.zfill(2)
                possible_file = huc_dir / f"{gage_id}_lump_nldas_forcing_leap.txt"
                if possible_file.exists():
                    logger.debug(f"Found gage {gage_id} in HUC2 {huc2_name}")
                    return huc2_name
        return None

    def _read_forcing_file(self, file_path: Path, gage_id: str) -> pd.DataFrame:
        logger.debug(f"Reading forcing file: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
            has_header = 'Year' in first_line
            
            if has_header:
                df = pd.read_csv(file_path, sep=r'\s+', header=0)
                logger.debug(f"Forcing file has header with columns: {df.columns.tolist()}")
                column_mapping = {
                    'Year': 'year', 'Mnth': 'month', 'Day': 'day', 'Hr': 'hour',
                    'total_precipitation(kg/m^2)': 'total_precipitation',
                    'temperature(C)': 'temperature',
                    'specific_humidity(kg/kg)': 'specific_humidity',
                    'shortwave_radiation(W/m^2)': 'shortwave_radiation',
                    'potential_energy(J/kg)': 'potential_energy'
                }
                df = df.rename(columns=column_mapping)
                if 'gage_id' in df.columns:
                    df = df.drop(columns=['gage_id'])
            else:
                df = pd.read_csv(file_path, sep=r'\s+', header=None, dtype=str)
                num_cols = df.shape[1]
                if num_cols == 11:
                    forcing_vars = ['SWdown', 'LWdown', 'Tair', 'Qair', 'Wind', 'Rainf', 'Snowf', 'Psurf']
                    df.columns = ['year', 'month', 'day'] + forcing_vars
                elif num_cols == 12:
                    forcing_vars = ['SWdown', 'LWdown', 'Tair', 'Qair', 'Wind', 'Rainf', 'Snowf', 'Psurf']
                    df.columns = ['gage_id', 'year', 'month', 'day'] + forcing_vars
                    df = df.drop(columns=['gage_id'])
                elif num_cols >= 3:
                    df.columns = ['year', 'month', 'day'] + [f'var_{i}' for i in range(num_cols - 3)]
                    logger.warning(f"Unexpected forcing file format with {num_cols} columns")
                else:
                    raise ValueError(f"Too few columns in forcing file: {num_cols}")
        
        except Exception as e:
            logger.error(f"Failed to read forcing file {file_path}: {e}")
            raise
        
        if not all(col in df.columns for col in ['year', 'month', 'day']):
            date_cols = []
            for col in df.columns:
                col_lower = str(col).lower()
                if 'year' in col_lower:
                    date_cols.append(('year', col))
                elif 'month' in col_lower or 'mnth' in col_lower:
                    date_cols.append(('month', col))
                elif 'day' in col_lower:
                    date_cols.append(('day', col))
            if len(date_cols) == 3:
                for new_name, old_name in date_cols:
                    df = df.rename(columns={old_name: new_name})
                logger.debug(f"Renamed date columns: {date_cols}")
            else:
                logger.warning(f"Missing required date columns. Found: {df.columns.tolist()}")
                raise ValueError(f"Missing required date columns in {file_path}")
        
        for col in ['year', 'month', 'day']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['year', 'month', 'day'])
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
        df = df.dropna(subset=['date'])
        
        for col in df.columns:
            if col not in ['year', 'month', 'day', 'date', 'gage_id', 'hour']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['gage_id'] = gage_id
        cols_to_drop = ['year', 'month', 'day', 'hour']
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        required_cols = ['date', 'gage_id'] + [var for var in self.REQUIRED_VARIABLES if var in df.columns]
        df = df[required_cols]
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.debug(f"Loaded {len(df)} rows of forcing data for gage {gage_id}")
        return df

    def get_available_gages(self) -> List[str]:
        gages = []
        if not self.data_source_path.exists():
            return gages
        forcing_dir = self.data_source_path
        if (self.data_source_path / "basin_mean_forcing").exists():
            forcing_dir = self.data_source_path / "basin_mean_forcing"
        for huc_dir in forcing_dir.iterdir():
            if huc_dir.is_dir() and huc_dir.name.isdigit():
                for file in huc_dir.glob("*_lump_nldas_forcing_leap.txt"):
                    gage_id = file.name.split('_')[0]
                    gages.append(gage_id)
        return sorted(set(gages))