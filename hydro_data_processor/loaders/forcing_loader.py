"""
Forcing data loader for NLDAS dataset.
Fixed to handle CAMELS forcing format with year, month, day columns.
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

    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.data_source_path = config.data_source_path
        logger.debug(f"ForcingLoader initialized for {self.data_source_path}")

    def load(self, gauge_ids: List[str], **kwargs) -> Optional[pd.DataFrame]:
        """Load forcing data for given gauge IDs."""
        all_data = []
        successful_gauges = []

        for gauge_id in gauge_ids:
            try:
                # Get HUC2 from kwargs or find it
                huc2 = kwargs.get('huc2')
                if not huc2:
                    huc2 = self._find_huc_by_scanning(gauge_id)
                
                if not huc2:
                    logger.warning(f"No HUC2 found for gauge {gauge_id}")
                    continue
                
                # Ensure HUC2 is 2-digit format
                huc2 = str(huc2).zfill(2)
                
                gauge_data = self._load_single_gauge(gauge_id, huc2)
                if gauge_data is not None and not gauge_data.empty:
                    all_data.append(gauge_data)
                    successful_gauges.append(gauge_id)
            except Exception as e:
                logger.warning(
                    f"Failed to load forcing for gauge {gauge_id}: {e}")

        if not all_data:
            logger.error("No forcing data loaded")
            return None

        logger.info(f"Loaded forcing for {len(successful_gauges)} gauges")
        return pd.concat(all_data, ignore_index=True)

    def _load_single_gauge(self, gauge_id: str, huc2: str) -> Optional[pd.DataFrame]:
        """Load forcing data for a single gauge."""
        # Build file path
        if hasattr(self.config, 'get_file_path'):
            file_path = self.config.get_file_path(gauge_id, huc2)
        else:
            # Fallback: build path manually
            file_path = self._build_file_path(gauge_id, huc2)
        
        if not file_path.exists():
            # Try alternative path
            file_path = self._find_alternative_path(gauge_id, huc2)
            if not file_path or not file_path.exists():
                logger.warning(f"Forcing file not found for gauge {gauge_id} in HUC2 {huc2}")
                return None
        
        try:
            return self._read_forcing_file(file_path, gauge_id)
        except Exception as e:
            logger.error(
                f"Error reading forcing file for gauge {gauge_id}: {e}")
            return None

    def _build_file_path(self, gauge_id: str, huc2: str) -> Path:
        """Build file path from configuration."""
        path = self.data_source_path
        
        if self.config.subdirectory:
            subdir = self.config.subdirectory.format(huc2=huc2)
            path = path / subdir
        
        if self.config.file_pattern:
            filename = self.config.file_pattern.format(basin_id=gauge_id, huc2=huc2)
            path = path / filename
        
        return path

    def _find_alternative_path(self, gauge_id: str, huc2: str) -> Optional[Path]:
        """Try alternative path patterns if default doesn't exist."""
        # Ensure HUC2 is 2-digit
        huc2_2digit = str(huc2).zfill(2)
        
        # Try multiple path patterns
        possible_paths = [
            self.data_source_path / "basin_mean_forcing" / huc2_2digit / f"{gauge_id}_lump_nldas_forcing_leap.txt",
            self.data_source_path / huc2_2digit / f"{gauge_id}_lump_nldas_forcing_leap.txt",
            self.data_source_path / "basin_mean_forcing" / huc2 / f"{gauge_id}_lump_nldas_forcing_leap.txt",
            self.data_source_path / huc2 / f"{gauge_id}_lump_nldas_forcing_leap.txt"
        ]
        
        for alt_path in possible_paths:
            if alt_path.exists():
                return alt_path
        
        return None

    def _find_huc_by_scanning(self, gauge_id: str) -> Optional[str]:
        """Scan HUC2 directories to find the forcing file."""
        if not self.data_source_path.exists():
            return None

        # Check if we're in the correct directory structure
        if (self.data_source_path / "basin_mean_forcing").exists():
            base_dir = self.data_source_path / "basin_mean_forcing"
        else:
            base_dir = self.data_source_path

        # Scan all HUC2 directories
        for huc_dir in base_dir.iterdir():
            if huc_dir.is_dir() and huc_dir.name.isdigit():
                # Ensure HUC2 is 2-digit for matching
                huc2_name = huc_dir.name.zfill(2)
                # Check if file exists in this HUC2 directory
                possible_file = huc_dir / f"{gauge_id}_lump_nldas_forcing_leap.txt"
                if possible_file.exists():
                    logger.debug(
                        f"Found gauge {gauge_id} in HUC2 {huc2_name}")
                    return huc2_name

        return None

    def _read_forcing_file(self, file_path: Path, gauge_id: str) -> pd.DataFrame:
        """Read and parse a NLDAS forcing file."""
        logger.debug(f"Reading forcing file: {file_path}")
        
        try:
            # First: Determine if file has header by reading first line
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
            
            # Check if first line contains 'Year' (indicating header)
            has_header = 'Year' in first_line
            
            if has_header:
                # Read with header and space separator
                df = pd.read_csv(file_path, sep=r'\s+', header=0)
                logger.debug(f"Forcing file has header with columns: {df.columns.tolist()}")
                
                # Standard column name mapping for NLDAS forcing data
                column_mapping = {
                    'Year': 'year',
                    'Mnth': 'month', 
                    'Day': 'day',
                    'Hr': 'hour',
                    'temperature(C)': 'temperature',
                    'specific_humidity(kg/kg)': 'specific_humidity',
                    'pressure(Pa)': 'pressure',
                    'wind_u(m/s)': 'wind_u',
                    'wind_v(m/s)': 'wind_v',
                    'longwave_radiation(W/m^2)': 'longwave_radiation',
                    'convective_fraction(-)': 'convective_fraction',
                    'shortwave_radiation(W/m^2)': 'shortwave_radiation',
                    'potential_energy(J/kg)': 'potential_energy',
                    'potential_evaporation(kg/m^2)': 'potential_evaporation',
                    'total_precipitation(kg/m^2)': 'total_precipitation'
                }
                
                # Rename columns
                df = df.rename(columns=column_mapping)
                
                # If gauge_id column exists, drop it (we'll add it back later)
                if 'gauge_id' in df.columns:
                    df = df.drop(columns=['gauge_id'])
            else:
                # Read without header (original logic for old format)
                df = pd.read_csv(file_path, sep=r'\s+', header=None, dtype=str)
                
                # Determine number of columns and format
                num_cols = df.shape[1]
                
                if num_cols == 11:
                    # Expected format: year month day + 8 forcing variables
                    forcing_vars = ['SWdown', 'LWdown', 'Tair', 'Qair', 'Wind', 'Rainf', 'Snowf', 'Psurf']
                    df.columns = ['year', 'month', 'day'] + forcing_vars
                elif num_cols == 12:
                    # Format with gauge_id: gauge_id year month day + 8 forcing variables
                    forcing_vars = ['SWdown', 'LWdown', 'Tair', 'Qair', 'Wind', 'Rainf', 'Snowf', 'Psurf']
                    df.columns = ['gauge_id', 'year', 'month', 'day'] + forcing_vars
                    df = df.drop(columns=['gauge_id'])
                elif num_cols >= 3:
                    # Unknown format, use generic names
                    df.columns = ['year', 'month', 'day'] + \
                        [f'var_{i}' for i in range(num_cols - 3)]
                    logger.warning(f"Unexpected forcing file format with {num_cols} columns")
                else:
                    raise ValueError(f"Too few columns in forcing file: {num_cols}")
        
        except Exception as e:
            logger.error(f"Failed to read forcing file {file_path}: {e}")
            raise
        
        # Ensure required columns exist
        if not all(col in df.columns for col in ['year', 'month', 'day']):
            # Try to find date columns with different names
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
        
        # Convert date columns to numeric
        for col in ['year', 'month', 'day']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows where date columns are NaN
        df = df.dropna(subset=['year', 'month', 'day'])
        
        # Create date column
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Convert other columns to numeric
        for col in df.columns:
            if col not in ['year', 'month', 'day', 'date', 'gauge_id', 'hour']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add gauge_id
        df['gauge_id'] = gauge_id
        
        # Drop year/month/day/hour columns
        cols_to_drop = ['year', 'month', 'day', 'hour']
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        # Reorder columns, put gauge_id as second column
        cols = ['date', 'gauge_id'] + [c for c in df.columns if c not in ['date', 'gauge_id']]
        df = df[cols]
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.debug(f"Loaded {len(df)} rows of forcing data for gauge {gauge_id}")
        return df

    def get_available_gauges(self) -> List[str]:
        """Get list of available gauge IDs."""
        gauges = []

        if not self.data_source_path.exists():
            return gauges

        # Find the basin_mean_forcing directory
        forcing_dir = self.data_source_path
        if (self.data_source_path / "basin_mean_forcing").exists():
            forcing_dir = self.data_source_path / "basin_mean_forcing"

        # Scan all HUC2 directories
        for huc_dir in forcing_dir.iterdir():
            if huc_dir.is_dir() and huc_dir.name.isdigit():
                for file in huc_dir.glob("*_lump_nldas_forcing_leap.txt"):
                    # Extract gauge_id from filename
                    gauge_id = file.name.split('_')[0]
                    gauges.append(gauge_id)

        return sorted(set(gauges))