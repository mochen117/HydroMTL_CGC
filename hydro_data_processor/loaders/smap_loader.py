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
    """Loader for soil moisture data."""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config, "smap")
        logger.info(f"SMAPLoader initialized for {config.data_source_path}")

    def load(self, gage_ids: List[str], **kwargs) -> pd.DataFrame:
        """
        Load SMAP data for specified gauges.

        Args:
            gage_ids: List of gage IDs to load (8-digit format, e.g., "01013500")
            **kwargs: Additional parameters including:
                - huc2: HUC2 code (optional, can be inferred)
                - start_date: Start date for filtering (optional)
                - end_date: End date for filtering (optional)

        Returns:
            DataFrame with SMAP data (no interpolation for missing values)
        """
        if not gage_ids:
            logger.warning("No gage IDs provided")
            return pd.DataFrame()

        all_data = []

        for gage_id in gage_ids:
            try:
                gage_data = self._load_single_gauge(gage_id, **kwargs)
                if gage_data is not None and not gage_data.empty:
                    all_data.append(gage_data)
                    logger.debug(f"Loaded SMAP data for gage {gage_id}")
                else:
                    logger.warning(f"No SMAP data found for gage {gage_id}")
            except Exception as e:
                logger.error(f"Error loading SMAP data for gage {gage_id}: {e}")

        if not all_data:
            return pd.DataFrame()

        # Combine all gage data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Apply date filtering if specified
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        if start_date or end_date:
            combined_df = self._filter_by_dates(combined_df, start_date, end_date)

        # Store and return
        self.data = combined_df
        return combined_df

    def _load_single_gauge(self, gage_id: str, **kwargs) -> Optional[pd.DataFrame]:
        """Load SMAP data for a single gage."""
        huc2 = kwargs.get('huc2')

        # Try to determine HUC2 if not provided
        if not huc2:
            huc2 = self._find_huc2_for_gauge(gage_id)
            if not huc2:
                logger.warning(f"Cannot determine HUC2 for gage {gage_id}")
                return None

        # Build file path
        file_path = self._build_file_path(gage_id, huc2)
        if not file_path or not file_path.exists():
            # Try alternative path patterns
            file_path = self._find_alternative_path(gage_id, huc2)
            if not file_path or not file_path.exists():
                logger.debug(f"SMAP file not found for gage {gage_id} in HUC2 {huc2}")
                return None

        # Read and process the file
        try:
            return self._read_and_process_file(file_path, gage_id)
        except Exception as e:
            logger.error(f"Failed to process SMAP file {file_path}: {e}")
            return None

    def _build_file_path(self, gage_id: str, huc2: str) -> Optional[Path]:
        """Build file path from configuration."""
        if not self.config.data_source_path.exists():
            logger.warning(f"Data source path does not exist: {self.config.data_source_path}")
            return None

        # Use the configuration's get_file_path method if available
        if hasattr(self.config, 'get_file_path'):
            return self.config.get_file_path(gage_id, huc2)

        # Otherwise build manually
        path = self.config.data_source_path

        if self.config.subdirectory:
            subdir = self.config.subdirectory.format(huc2=huc2)
            path = path / subdir

        if self.config.file_pattern:
            # Use the original 8-digit gage_id in file pattern
            filename = self.config.file_pattern.replace('{basin_id}', gage_id)
            filename = filename.replace('{gage_id}', gage_id)
            path = path / filename

        return path

    def _find_huc2_for_gauge(self, gage_id: str) -> Optional[str]:
        """Try to find HUC2 code for a gage by scanning directories."""
        if not self.config.data_source_path.exists():
            return None

        # Look for HUC2 directories
        huc2_dirs = [d for d in self.config.data_source_path.iterdir()
                     if d.is_dir() and d.name.isdigit() and len(d.name) == 2]

        for huc2_dir in huc2_dirs:
            # Check if file exists in this HUC2 directory
            test_path = self._build_file_path(gage_id, huc2_dir.name)
            if test_path and test_path.exists():
                return huc2_dir.name

        return None

    def _find_alternative_path(self, gage_id: str, huc2: str) -> Optional[Path]:
        """Try alternative path patterns if default doesn't exist."""
        # Try standard CAMELS pattern
        alt_path = self.config.data_source_path / "NASA_USDA_SMAP_CAMELS" / huc2 / f"{gage_id}_lump_nasa_usda_smap.txt"
        if alt_path.exists():
            return alt_path

        return None

    def _read_and_process_file(self, file_path: Path, gage_id: str) -> pd.DataFrame:
        """Read and process SMAP data file."""
        try:
            # Read SMAP data (comma-separated with header)
            df = pd.read_csv(file_path, sep=',', header=0)

            # Standardize column names
            df = df.rename(columns={
                'Year': 'year',
                'Mnth': 'month',
                'Day': 'day',
                'Hr': 'hour',
                'ssm(mm)': 'ssm',
                'susm(mm)': 'susm',
                'smp(-)': 'smp',
                'ssma(-)': 'ssma',
                'susma(-)': 'susma'
            })

            # Convert numeric columns
            df['year'] = df['year'].astype(int)
            df['month'] = df['month'].astype(int)
            df['day'] = df['day'].astype(int)

            # Create date column
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

            # Convert data columns to numeric
            numeric_cols = ['ssm', 'susm', 'smp', 'ssma', 'susma']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Add gage_id (original 8-digit format)
            df['gage_id'] = gage_id

            # Drop year/month/day/hour columns
            cols_to_drop = ['year', 'month', 'day', 'hour']
            cols_to_drop = [col for col in cols_to_drop if col in df.columns]
            df = df.drop(columns=cols_to_drop)

            # Reorder columns
            col_order = ['date', 'gage_id'] + [c for c in df.columns if c not in ['date', 'gage_id']]
            df = df[col_order]

            logger.debug(f"Loaded SMAP data from {file_path}: {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Error reading SMAP file {file_path}: {e}")
            # Try with different separators if comma fails
            try:
                df = pd.read_csv(file_path, sep=r'\s+', header=0)
                # Apply same column renaming and processing
                return self._process_dataframe(df, gage_id)
            except Exception as e2:
                logger.error(f"Failed to read SMAP file with whitespace separator: {e2}")
                raise

    def _process_dataframe(self, df: pd.DataFrame, gage_id: str) -> pd.DataFrame:
        """Process SMAP dataframe after reading with alternative separator."""
        # Standardize column names (may have different format)
        # Find the actual column names
        actual_columns = list(df.columns)
        
        # Map expected column names based on position or partial matching
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
            elif 'susm' in col_lower:
                column_mapping[col] = 'susm'
            elif 'smp' in col_lower:
                column_mapping[col] = 'smp'
            elif 'ssma' in col_lower:
                column_mapping[col] = 'ssma'
            elif 'susma' in col_lower:
                column_mapping[col] = 'susma'
        
        df = df.rename(columns=column_mapping)
        
        # Continue with processing
        df['year'] = df['year'].astype(int)
        df['month'] = df['month'].astype(int)
        df['day'] = df['day'].astype(int)
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        
        numeric_cols = ['ssm', 'susm', 'smp', 'ssma', 'susma']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['gage_id'] = gage_id
        
        cols_to_drop = ['year', 'month', 'day', 'hour']
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        col_order = ['date', 'gage_id'] + [c for c in df.columns if c not in ['date', 'gage_id']]
        df = df[col_order]
        
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