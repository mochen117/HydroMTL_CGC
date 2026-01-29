"""
Streamflow data loader for CAMELS dataset.
Handles CAMELS streamflow format with gauge_id, year, month, day columns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from hydro_data_processor.config.settings import DataSourceConfig

logger = logging.getLogger(__name__)


class StreamflowLoader:
    """Loader for streamflow data from CAMELS dataset."""

    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.data_source_path = config.data_source_path
        logger.debug(f"StreamflowLoader initialized for {self.data_source_path}")

        # Initialize HUC2 mapping
        self.huc_mapping: Dict[str, str] = {}
        self._load_huc_mapping()

    def _load_huc_mapping(self):
        """Load HUC2 mapping from camels_name.txt if available."""
        try:
            # Try multiple possible locations for camels_name.txt
            possible_paths = [
                self.data_source_path.parent.parent.parent.parent / "camels_name.txt",
                self.data_source_path.parent.parent.parent / "camels_name.txt",
                self.data_source_path.parent.parent / "camels_name.txt",
                self.data_source_path.parent / "camels_name.txt",
                Path("/home/mochen/hydro_data/camels/camels_us/camels_name.txt"),
            ]
            
            name_file = None
            for path in possible_paths:
                if path.exists():
                    name_file = path
                    break
            
            if name_file and name_file.exists():
                df = pd.read_csv(name_file, delimiter=';')
                
                # Ensure gauge_id is 8-digit format
                def ensure_8_digits(gauge_id):
                    if gauge_id is None:
                        return None
                    gauge_str = str(gauge_id)
                    if len(gauge_str) == 7:
                        return '0' + gauge_str
                    elif len(gauge_str) < 8:
                        return gauge_str.zfill(8)
                    return gauge_str
                
                df['gauge_id'] = df['gauge_id'].astype(str).apply(ensure_8_digits)
                
                # Ensure huc_02 is 2-digit format
                if 'huc_02' in df.columns:
                    df['huc_02'] = df['huc_02'].astype(str).str.zfill(2)
                    self.huc_mapping = dict(zip(df['gauge_id'], df['huc_02']))
                    logger.debug(f"Loaded HUC mapping: {len(self.huc_mapping)} gauges")
                else:
                    logger.debug(f"camels_name.txt does not contain huc_02 column")
            else:
                logger.debug("camels_name.txt not found in expected locations")
                
        except Exception as e:
            logger.debug(f"Failed to load HUC mapping: {e}")

    def _get_huc_for_gauge(self, gauge_id: str) -> Optional[str]:
        """Get HUC2 code for a gauge_id."""
        # Ensure gauge_id is 8-digit format
        gauge_id_8 = gauge_id
        if len(gauge_id) == 7:
            gauge_id_8 = '0' + gauge_id
        elif len(gauge_id) < 8:
            gauge_id_8 = gauge_id.zfill(8)

        # Lookup in mapping
        if gauge_id_8 in self.huc_mapping:
            return self.huc_mapping[gauge_id_8]

        # Try original gauge_id
        if gauge_id in self.huc_mapping:
            return self.huc_mapping[gauge_id]

        return None

    def load(self, gauge_ids: List[str]) -> Optional[pd.DataFrame]:
        """Load streamflow data for given gauge IDs."""
        all_data = []
        successful_gauges = []

        for gauge_id in gauge_ids:
            try:
                gauge_data = self._load_single_gauge(gauge_id)
                if gauge_data is not None and not gauge_data.empty:
                    all_data.append(gauge_data)
                    successful_gauges.append(gauge_id)
            except Exception as e:
                logger.warning(f"Failed to load streamflow for gauge {gauge_id}: {e}")

        if not all_data:
            logger.error("No valid streamflow data loaded")
            return None

        logger.info(f"Loaded streamflow for {len(successful_gauges)} gauges")
        return pd.concat(all_data, ignore_index=True)

    def _load_single_gauge(self, gauge_id: str) -> Optional[pd.DataFrame]:
        """Load streamflow data for a single gauge."""
        # Get HUC2 code
        huc2 = self._get_huc_for_gauge(gauge_id)

        if not huc2:
            logger.debug(f"No HUC2 mapping found for gauge {gauge_id}")
            # Try to find by scanning HUC2 directories
            huc2 = self._find_huc_by_scanning(gauge_id)

        if not huc2:
            logger.warning(f"Could not find streamflow data for gauge {gauge_id}")
            return None

        # Build file path
        file_path = self.config.get_file_path(gauge_id, huc2)

        if not file_path.exists():
            logger.warning(f"Streamflow file not found: {file_path}")
            return None

        try:
            return self._read_streamflow_file(file_path, gauge_id)
        except Exception as e:
            logger.error(f"Error reading streamflow file for gauge {gauge_id}: {e}")
            return None

    def _find_huc_by_scanning(self, gauge_id: str) -> Optional[str]:
        """Scan HUC2 directories to find the gauge file."""
        if not self.data_source_path.exists():
            return None

        # Ensure gauge_id is 8-digit for filename matching
        gauge_id_8 = gauge_id
        if len(gauge_id) == 7:
            gauge_id_8 = '0' + gauge_id
        elif len(gauge_id) < 8:
            gauge_id_8 = gauge_id.zfill(8)

        # Scan all HUC2 directories
        for huc_dir in self.data_source_path.iterdir():
            if huc_dir.is_dir() and huc_dir.name.isdigit():
                # Check if file exists in this HUC2 directory
                possible_file = huc_dir / f"{gauge_id_8}_streamflow_qc.txt"
                if possible_file.exists():
                    logger.debug(f"Found gauge {gauge_id} in HUC2 {huc_dir.name}")
                    return huc_dir.name.zfill(2)  # Ensure 2-digit format

        return None

    def _read_streamflow_file(self, file_path: Path, gauge_id: str) -> pd.DataFrame:
        """Read and parse a CAMELS streamflow file."""
        logger.debug(f"Reading streamflow file: {file_path}")

        # Read the file with whitespace separator, no header
        df = pd.read_csv(file_path, sep=r'\s+', header=None, dtype=str)

        # Check number of columns
        if df.shape[1] == 6:
            # Expected format: gauge_id year month day streamflow qc_flag
            df.columns = [
                'file_gauge_id',
                'year',
                'month',
                'day',
                'streamflow',
                'qc_flag']
        elif df.shape[1] == 5:
            # Format without qc_flag
            df.columns = [
                'file_gauge_id',
                'year',
                'month',
                'day',
                'streamflow']
            df['qc_flag'] = None
        else:
            raise ValueError(f"Unexpected number of columns: {df.shape[1]}")

        # Convert numeric columns
        df['year'] = df['year'].astype(int)
        df['month'] = df['month'].astype(int)
        df['day'] = df['day'].astype(int)
        df['streamflow'] = pd.to_numeric(df['streamflow'], errors='coerce')

        # Parse file_gauge_id to verify
        df['file_gauge_id'] = df['file_gauge_id'].astype(str)

        # Create date column
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

        # Verify gauge_id matches
        file_gauge_id = df['file_gauge_id'].iloc[0]
        
        # Ensure gauge_id is 8-digit for comparison
        gauge_id_8 = gauge_id
        if len(gauge_id) == 7:
            gauge_id_8 = '0' + gauge_id
        elif len(gauge_id) < 8:
            gauge_id_8 = gauge_id.zfill(8)

        if file_gauge_id != gauge_id_8:
            logger.debug(f"Gauge ID mismatch: file has {file_gauge_id}, expected {gauge_id_8}")

        # Filter and rename columns
        result_df = df[['date', 'streamflow', 'qc_flag']].copy()
        result_df['gauge_id'] = gauge_id_8  # Use 8-digit gauge_id

        # Sort by date
        result_df = result_df.sort_values('date').reset_index(drop=True)

        logger.debug(f"Loaded {len(result_df)} rows for gauge {gauge_id}")
        return result_df

    def get_available_gauges(self) -> List[str]:
        """Get list of available gauge IDs."""
        gauges = []

        if not self.data_source_path.exists():
            return gauges

        # Scan all HUC2 directories
        for huc_dir in self.data_source_path.iterdir():
            if huc_dir.is_dir() and huc_dir.name.isdigit():
                for file in huc_dir.glob("*_streamflow_qc.txt"):
                    # Extract gauge_id from filename
                    gauge_id = file.name.split('_')[0]
                    gauges.append(gauge_id)

        return sorted(set(gauges))