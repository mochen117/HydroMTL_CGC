"""
ET loader for CAMELS data.
Loads evapotranspiration data.
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


class ETLoader(BaseDataLoader):
    """Loader for evapotranspiration data."""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config, "et")
        logger.info(f"ETLoader initialized for {config.data_source_path}")

    def load(self, gauge_ids: List[str], **kwargs) -> pd.DataFrame:
        """
        Load ET data for specified gauges.

        Args:
            gauge_ids: List of gauge IDs to load (8-digit format, e.g., "01013500")
            **kwargs: Additional parameters including:
                - huc2: HUC2 code (optional, can be inferred)
                - start_date: Start date for filtering (optional)
                - end_date: End date for filtering (optional)

        Returns:
            DataFrame with ET data
        """

        if not gauge_ids:
            logger.warning("No gauge IDs provided")
            return pd.DataFrame()

        all_data = []

        for gauge_id in gauge_ids:
            try:
                gauge_data = self._load_single_gauge(gauge_id, **kwargs)
                if gauge_data is not None and not gauge_data.empty:
                    all_data.append(gauge_data)
                    logger.debug(f"Loaded ET data for gauge {gauge_id}")
                else:
                    logger.warning(f"No ET data found for gauge {gauge_id}")
            except Exception as e:
                logger.error(
                    f"Error loading ET data for gauge {gauge_id}: {e}")

        if not all_data:
            return pd.DataFrame()

        # Combine all gauge data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Apply date filtering if specified
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        if start_date or end_date:
            combined_df = self._filter_by_dates(
                combined_df, start_date, end_date)

        # Store and return
        self.data = combined_df
        return combined_df

    def _load_single_gauge(self, gauge_id: str, **
                           kwargs) -> Optional[pd.DataFrame]:
        """Load ET data for a single gauge."""
        huc2 = kwargs.get('huc2')

        # Try to determine HUC2 if not provided
        if not huc2:
            huc2 = self._find_huc2_for_gauge(gauge_id)
            if not huc2:
                logger.warning(f"Cannot determine HUC2 for gauge {gauge_id}")
                return None

        # Build file path
        file_path = self._build_file_path(gauge_id, huc2)
        if not file_path or not file_path.exists():
            # Try alternative path patterns
            file_path = self._find_alternative_path(gauge_id, huc2)
            if not file_path or not file_path.exists():
                logger.debug(
                    f"ET file not found for gauge {gauge_id} in HUC2 {huc2}")
                return None

        # Read and process the file
        try:
            return self._read_and_process_file(file_path, gauge_id)
        except Exception as e:
            logger.error(f"Failed to process ET file {file_path}: {e}")
            return None

    def _build_file_path(self, gauge_id: str, huc2: str) -> Optional[Path]:
        """Build file path from configuration."""
        if not self.config.data_source_path.exists():
            logger.warning(
                f"Data source path does not exist: {self.config.data_source_path}")
            return None

        # Use the configuration's get_file_path method if available
        if hasattr(self.config, 'get_file_path'):
            return self.config.get_file_path(gauge_id, huc2)

        # Otherwise build manually
        path = self.config.data_source_path

        if self.config.subdirectory:
            subdir = self.config.subdirectory.format(huc2=huc2)
            path = path / subdir

        if self.config.file_pattern:
            # Use the original 8-digit gauge_id in file pattern
            # Replace both {basin_id} and {gauge_id} with the original 8-digit gauge_id
            filename = self.config.file_pattern.replace('{basin_id}', gauge_id)
            filename = filename.replace('{gauge_id}', gauge_id)
            path = path / filename

        return path

    def _find_huc2_for_gauge(self, gauge_id: str) -> Optional[str]:
        """Try to find HUC2 code for a gauge by scanning directories."""
        if not self.config.data_source_path.exists():
            return None

        # Look for HUC2 directories
        huc2_dirs = [d for d in self.config.data_source_path.iterdir()
                     if d.is_dir() and d.name.isdigit() and len(d.name) == 2]

        for huc2_dir in huc2_dirs:
            # Check if file exists in this HUC2 directory
            test_path = self._build_file_path(gauge_id, huc2_dir.name)
            if test_path and test_path.exists():
                return huc2_dir.name

        return None

    def _find_alternative_path(
            self,
            gauge_id: str,
            huc2: str) -> Optional[Path]:
        """Try alternative path patterns if default doesn't exist."""
        # Use the original 8-digit gauge_id for file path
        
        # Try standard CAMELS pattern
        alt_path = self.config.data_source_path / "basin_mean_forcing" / \
            huc2 / f"{gauge_id}_lump_modis16a2v006_et.txt"
        if alt_path.exists():
            return alt_path

        return None

    def _read_and_process_file(
            self,
            file_path: Path,
            gauge_id: str) -> pd.DataFrame:
        """Read and process ET data file."""
        # Read ET data (comma-separated with header)
        df = pd.read_csv(file_path, sep=',', header=0)

        # Standardize column names
        df = df.rename(columns={
            'Year': 'year',
            'Mnth': 'month',
            'Day': 'day',
            'Hr': 'hour',
            'ET(kg/m^2/8day)': 'et',
            'LE(J/m^2/day)': 'le',
            'PET(kg/m^2/8day)': 'pet',
            'PLE(J/m^2/day)': 'ple',
            'ET_QC': 'et_qc'
        })

        # Convert numeric columns
        df['year'] = df['year'].astype(int)
        df['month'] = df['month'].astype(int)
        df['day'] = df['day'].astype(int)

        # Create date column
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

        # Convert data columns to numeric
        numeric_cols = ['et', 'le', 'pet', 'ple', 'et_qc']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Add gauge_id (original 8-digit format)
        df['gauge_id'] = gauge_id

        # Drop year/month/day/hour columns
        cols_to_drop = ['year', 'month', 'day', 'hour']
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(columns=cols_to_drop)

        # Reorder columns
        col_order = ['date', 'gauge_id'] + \
            [c for c in df.columns if c not in ['date', 'gauge_id']]
        df = df[col_order]

        logger.debug(f"Loaded ET data from {file_path}: {len(df)} records")
        return df

    def _filter_by_dates(self, df: pd.DataFrame, start_date: Optional[str],
                         end_date: Optional[str]) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        if df.empty or 'date' not in df.columns:
            return df

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        if start_date:
            start_dt = pd.Timestamp(start_date)
            df = df[df['date'] >= start_dt]

        if end_date:
            end_dt = pd.Timestamp(end_date)
            df = df[df['date'] <= end_dt]

        return df.reset_index(drop=True)