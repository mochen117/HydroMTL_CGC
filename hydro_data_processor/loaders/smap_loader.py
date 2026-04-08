"""
SMAP soil moisture data loader.
Modified to load only ssm variable for model evaluation as per paper requirements.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
import numpy as np
from datetime import datetime

from .base_loader import BaseDataLoader
from hydro_data_processor.config.settings import DataSourceConfig

logger = logging.getLogger(__name__)


class SMAPLoader(BaseDataLoader):
    """Loader for soil moisture data for model evaluation only."""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config, "smap")
        logger.info(f"SMAPLoader initialized for {config.data_source_path}")

    def load(self, gage_ids: List[str], **kwargs) -> pd.DataFrame:
        if not gage_ids:
            logger.warning("No gage IDs provided")
            return pd.DataFrame()

        all_data = []
        for gage_id in gage_ids:
            try:
                gage_data = self._load_single_gauge(gage_id, **kwargs)
                if gage_data is not None and not gage_data.empty:
                    gage_data = self._filter_to_ssm_only(gage_data)
                    all_data.append(gage_data)
                    logger.debug(f"Loaded SMAP data for gage {gage_id}")
                else:
                    logger.debug(f"No SMAP data found for gage {gage_id}")
            except Exception as e:
                logger.error(f"Error loading SMAP data for gage {gage_id}: {e}")

        if not all_data:
            return pd.DataFrame()

        combined_df = pd.concat(all_data, ignore_index=True)
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        if start_date or end_date:
            combined_df = self._filter_by_dates(combined_df, start_date, end_date)
        self.data = combined_df
        return combined_df

    def _filter_to_ssm_only(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        keep_cols = ['date', 'gage_id']
        if 'ssm' in df.columns:
            keep_cols.append('ssm')
        else:
            logger.warning("ssm column not found in SMAP data")
            df['ssm'] = np.nan
        return df[keep_cols]

    def _load_single_gauge(self, gage_id: str, **kwargs) -> Optional[pd.DataFrame]:
        huc2 = kwargs.get('huc2')
        if not huc2:
            huc2 = self._find_huc2_for_gauge(gage_id)
            if not huc2:
                logger.debug(f"Cannot determine HUC2 for gage {gage_id}")
                return None
        file_path = self._build_file_path(gage_id, huc2)
        if not file_path or not file_path.exists():
            file_path = self._find_alternative_path(gage_id, huc2)
            if not file_path or not file_path.exists():
                logger.debug(f"SMAP file not found for gage {gage_id} in HUC2 {huc2}")
                return None
        try:
            return self._read_and_process_file(file_path, gage_id)
        except Exception as e:
            logger.error(f"Failed to process SMAP file {file_path}: {e}")
            return None

    def _build_file_path(self, gage_id: str, huc2: str) -> Optional[Path]:
        if not self.config.data_source_path.exists():
            logger.warning(f"Data source path does not exist: {self.config.data_source_path}")
            return None
        if hasattr(self.config, 'get_file_path'):
            return self.config.get_file_path(gage_id, huc2)
        path = self.config.data_source_path
        if self.config.subdirectory:
            subdir = self.config.subdirectory.format(huc2=huc2)
            path = path / subdir
        if self.config.file_pattern:
            filename = self.config.file_pattern.replace('{basin_id}', gage_id).replace('{gage_id}', gage_id)
            path = path / filename
        return path

    def _find_huc2_for_gauge(self, gage_id: str) -> Optional[str]:
        if not self.config.data_source_path.exists():
            return None
        huc2_dirs = [d for d in self.config.data_source_path.iterdir()
                     if d.is_dir() and d.name.isdigit() and len(d.name) == 2]
        for huc2_dir in huc2_dirs:
            test_path = self._build_file_path(gage_id, huc2_dir.name)
            if test_path and test_path.exists():
                return huc2_dir.name
        return None

    def _find_alternative_path(self, gage_id: str, huc2: str) -> Optional[Path]:
        alt_path = self.config.data_source_path / "NASA_USDA_SMAP_CAMELS" / huc2 / f"{gage_id}_lump_nasa_usda_smap.txt"
        if alt_path.exists():
            return alt_path
        return None

    def _read_and_process_file(self, file_path: Path, gage_id: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path, sep=',', header=0)
            column_mapping = {
                'Year': 'year', 'Mnth': 'month', 'Day': 'day', 'Hr': 'hour',
                'ssm(mm)': 'ssm'
            }
            df = df.rename(columns=column_mapping)
            df['year'] = df['year'].astype(int)
            df['month'] = df['month'].astype(int)
            df['day'] = df['day'].astype(int)
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
            if 'ssm' in df.columns:
                df['ssm'] = pd.to_numeric(df['ssm'], errors='coerce')
            else:
                df['ssm'] = np.nan
            df['gage_id'] = gage_id
            cols_to_drop = ['year', 'month', 'day', 'hour']
            cols_to_drop = [col for col in cols_to_drop if col in df.columns]
            df = df.drop(columns=cols_to_drop)
            df = df[['date', 'gage_id', 'ssm']]
            logger.debug(f"Loaded SMAP data from {file_path}: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error reading SMAP file {file_path}: {e}")
            try:
                df = pd.read_csv(file_path, sep=r'\s+', header=0)
                return self._process_dataframe(df, gage_id)
            except Exception as e2:
                logger.error(f"Failed to read SMAP file with whitespace separator: {e2}")
                raise

    def _process_dataframe(self, df: pd.DataFrame, gage_id: str) -> pd.DataFrame:
        actual_columns = list(df.columns)
        column_mapping = {}
        for i, col in enumerate(actual_columns):
            col_lower = col.lower()
            if 'year' in col_lower:
                column_mapping[col] = 'year'
            elif 'mnth' in col_lower or 'month' in col_lower:
                column_mapping[col] = 'month'
            elif 'day' in col_lower:
                column_mapping[col] = 'day'
            elif 'hr' in col_lower or 'hour' in col_lower:
                column_mapping[col] = 'hour'
            elif 'ssm' in col_lower:
                column_mapping[col] = 'ssm'
        df = df.rename(columns=column_mapping)
        df['year'] = df['year'].astype(int)
        df['month'] = df['month'].astype(int)
        df['day'] = df['day'].astype(int)
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        if 'ssm' in df.columns:
            df['ssm'] = pd.to_numeric(df['ssm'], errors='coerce')
        else:
            df['ssm'] = np.nan
        df['gage_id'] = gage_id
        cols_to_drop = ['year', 'month', 'day', 'hour']
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        df = df[['date', 'gage_id', 'ssm']]
        return df

    def _filter_by_dates(self, df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        if df.empty or 'date' not in df.columns:
            return df
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        if start_date:
            df = df[df['date'] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df['date'] <= pd.Timestamp(end_date)]
        return df.reset_index(drop=True)