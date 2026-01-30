"""
Streamflow data loader for CAMELS dataset.
Handles both CAMELS (with header) and USGS (without header) streamflow formats.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from hydro_data_processor.config.settings import DataSourceConfig

logger = logging.getLogger(__name__)


class StreamflowLoader:
    """Loader for streamflow data from CAMELS and USGS datasets."""

    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.data_source_path = config.data_source_path
        self.data_source_type = "camels"
        logger.debug(f"StreamflowLoader initialized for {self.data_source_path}")

        self.huc_mapping: Dict[str, str] = {}
        self._load_huc_mapping()

    def _load_huc_mapping(self):
        """Load HUC2 mapping from camels_name.txt if available."""
        try:
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
                
                if 'gauge_id' in df.columns and 'gage_id' not in df.columns:
                    df = df.rename(columns={'gauge_id': 'gage_id'})
                    logger.debug("Renamed 'gauge_id' column to 'gage_id' in HUC mapping")
                
                df['gage_id'] = df['gage_id'].astype(str).str.strip()
                
                if 'huc_02' in df.columns:
                    df['huc_02'] = df['huc_02'].astype(str).str.zfill(2)
                    self.huc_mapping = dict(zip(df['gage_id'], df['huc_02']))
                    logger.debug(f"Loaded HUC mapping: {len(self.huc_mapping)} gages")
                else:
                    logger.debug(f"camels_name.txt does not contain huc_02 column")
            else:
                logger.debug("camels_name.txt not found in expected locations")
                
        except Exception as e:
            logger.debug(f"Failed to load HUC mapping: {e}")

    def _get_huc_for_gage(self, gage_id: str) -> Optional[str]:
        """Get HUC2 code for a gage_id."""
        gage_id_str = gage_id.strip()
        
        if gage_id_str in self.huc_mapping:
            return self.huc_mapping[gage_id_str]

        return None

    def load(self, gage_ids: List[str], huc2: str = None) -> Optional[pd.DataFrame]:
        """Load streamflow data for given gage IDs."""
        all_data = []
        successful_gages = []

        for gage_id in gage_ids:
            try:
                gage_data = self._load_single_gage(gage_id, huc2)
                if gage_data is not None and not gage_data.empty:
                    all_data.append(gage_data)
                    successful_gages.append(gage_id)
            except Exception as e:
                logger.warning(f"Failed to load streamflow for gage {gage_id}: {e}")

        if not all_data:
            logger.error("No valid streamflow data loaded")
            return None

        logger.info(f"Loaded streamflow for {len(successful_gages)} gages")
        return pd.concat(all_data, ignore_index=True)

    def _load_single_gage(self, gage_id: str, huc2: str = None) -> Optional[pd.DataFrame]:
        """Load streamflow data for a single gage."""
        if not huc2:
            huc2 = self._get_huc_for_gage(gage_id)

        if not huc2:
            logger.debug(f"No HUC2 mapping found for gage {gage_id}")
            huc2 = self._find_huc_by_scanning(gage_id)

        if not huc2:
            logger.warning(f"Could not find streamflow data for gage {gage_id}")
            return None

        file_path = self._build_file_path(gage_id, huc2)

        if not file_path.exists():
            logger.warning(f"Streamflow file not found: {file_path}")
            return None

        try:
            return self._read_streamflow_file(file_path, gage_id)
        except Exception as e:
            logger.error(f"Error reading streamflow file for gage {gage_id}: {e}")
            return None

    def _build_file_path(self, gage_id: str, huc2: str) -> Path:
        """Build the file path based on data source type."""
        gage_id_str = gage_id.strip()
        huc2_str = str(huc2).zfill(2)
        
        filename = f"{gage_id_str}_streamflow_qc.txt"
        
        if hasattr(self, 'data_source_type') and self.data_source_type == "usgs":
            return self.data_source_path / huc2_str / filename
        else:
            return self.data_source_path / huc2_str / filename

    def _find_huc_by_scanning(self, gage_id: str) -> Optional[str]:
        """Scan HUC2 directories to find the gage file."""
        if not self.data_source_path.exists():
            return None

        gage_id_str = gage_id.strip()

        for huc_dir in self.data_source_path.iterdir():
            if huc_dir.is_dir() and huc_dir.name.isdigit():
                filename = f"{gage_id_str}_streamflow_qc.txt"
                possible_file = huc_dir / filename
                if possible_file.exists():
                    logger.debug(f"Found gage {gage_id} in HUC2 {huc_dir.name}")
                    return huc_dir.name.zfill(2)

        return None

    def _read_streamflow_file(self, file_path: Path, gage_id: str) -> pd.DataFrame:
        """Read and parse a streamflow file (CAMELS with header or USGS without header)."""
        logger.debug(f"Reading streamflow file: {file_path}")

        is_usgs_data = False
        if hasattr(self, 'data_source_type') and self.data_source_type == "usgs":
            is_usgs_data = True
        elif "usgs_streamflow" in str(file_path):
            is_usgs_data = True

        if is_usgs_data:
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                
                has_header = ('GAGE_ID' in first_line.upper() or 
                             'gage_id' in first_line.lower() or 
                             'Year' in first_line or 
                             'year' in first_line.lower())
                
                is_comma_separated = ',' in first_line
                
                if has_header and is_comma_separated:
                    df = pd.read_csv(file_path, sep=',', header=0, dtype=str)
                    logger.debug(f"USGS data with header, comma-separated: {df.columns.tolist()}")
                    
                    column_mapping = {}
                    for col in df.columns:
                        col_lower = col.lower()
                        if 'gage' in col_lower or 'gauge' in col_lower:
                            column_mapping[col] = 'file_gage_id'
                        elif 'year' in col_lower:
                            column_mapping[col] = 'year'
                        elif 'month' in col_lower or 'mnth' in col_lower:
                            column_mapping[col] = 'month'
                        elif 'day' in col_lower:
                            column_mapping[col] = 'day'
                        elif 'streamflow' in col_lower:
                            column_mapping[col] = 'streamflow'
                        elif 'qc' in col_lower or 'flag' in col_lower:
                            column_mapping[col] = 'qc_flag'
                    
                    if column_mapping:
                        df = df.rename(columns=column_mapping)
                    
                    if 'file_gage_id' not in df.columns and 'gage_id' in df.columns:
                        df = df.rename(columns={'gage_id': 'file_gage_id'})
                    
                    if 'qc_flag' not in df.columns:
                        df['qc_flag'] = None
                        
                else:
                    df = pd.read_csv(file_path, sep=r'\s+', header=None, dtype=str)
                    
                    if df.shape[1] == 6:
                        df.columns = ['file_gage_id', 'year', 'month', 'day', 'streamflow', 'qc_flag']
                    elif df.shape[1] == 5:
                        df.columns = ['file_gage_id', 'year', 'month', 'day', 'streamflow']
                        df['qc_flag'] = None
                    else:
                        raise ValueError(f"Unexpected number of columns in USGS file: {df.shape[1]}")
                        
            except Exception as e:
                logger.error(f"Failed to read USGS file {file_path}: {e}")
                try:
                    df = pd.read_csv(file_path, sep=r'\s+', header=None, dtype=str)
                    if df.shape[1] == 6:
                        df.columns = ['file_gage_id', 'year', 'month', 'day', 'streamflow', 'qc_flag']
                    elif df.shape[1] == 5:
                        df.columns = ['file_gage_id', 'year', 'month', 'day', 'streamflow']
                        df['qc_flag'] = None
                    else:
                        raise ValueError(f"Unexpected number of columns in USGS file: {df.shape[1]}")
                except Exception as e2:
                    logger.error(f"Failed to read USGS file with whitespace separator: {e2}")
                    raise
        else:
            try:
                df = pd.read_csv(file_path, sep=r'\s+', header=0, dtype=str)
                
                expected_cols = ['gage_id', 'year', 'month', 'day', 'streamflow', 'qc_flag']
                if not all(col in df.columns for col in expected_cols):
                    df = pd.read_csv(file_path, sep=r'\s+', header=None, dtype=str)
                    if df.shape[1] == 6:
                        df.columns = expected_cols
                    elif df.shape[1] == 5:
                        df.columns = ['gage_id', 'year', 'month', 'day', 'streamflow']
                        df['qc_flag'] = None
                    else:
                        raise ValueError(f"Unexpected number of columns in CAMELS file: {df.shape[1]}")
            except Exception as e:
                logger.warning(f"Error reading CAMELS file with header, trying without: {e}")
                df = pd.read_csv(file_path, sep=r'\s+', header=None, dtype=str)
                if df.shape[1] == 6:
                    df.columns = ['gage_id', 'year', 'month', 'day', 'streamflow', 'qc_flag']
                elif df.shape[1] == 5:
                    df.columns = ['gage_id', 'year', 'month', 'day', 'streamflow']
                    df['qc_flag'] = None
                else:
                    raise ValueError(f"Unexpected number of columns: {df.shape[1]}")

        if 'file_gage_id' in df.columns:
            df['file_gage_id'] = df['file_gage_id'].astype(str).str.strip()
        elif 'gage_id' in df.columns:
            df['gage_id'] = df['gage_id'].astype(str).str.strip()
        
        df['year'] = df['year'].astype(int)
        df['month'] = df['month'].astype(int)
        df['day'] = df['day'].astype(int)
        df['streamflow'] = pd.to_numeric(df['streamflow'], errors='coerce')
        df['streamflow'] = df['streamflow'].replace(-999.0, np.nan)

        if 'file_gage_id' in df.columns:
            df['gage_id_from_file'] = df['file_gage_id']
        elif 'gage_id' in df.columns:
            df['gage_id_from_file'] = df['gage_id']

        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

        gage_id_str = gage_id.strip()

        if 'gage_id_from_file' in df.columns and not df.empty:
            file_gage_id = df['gage_id_from_file'].iloc[0]
            if file_gage_id != gage_id_str:
                logger.debug(f"Gage ID mismatch: file has {file_gage_id}, expected {gage_id_str}")

        if 'qc_flag' in df.columns:
            result_df = df[['date', 'streamflow', 'qc_flag']].copy()
        else:
            result_df = df[['date', 'streamflow']].copy()
        
        result_df['gage_id'] = gage_id_str
        result_df = result_df.sort_values('date').reset_index(drop=True)

        logger.debug(f"Loaded {len(result_df)} rows for gage {gage_id}")
        return result_df

    def get_available_gages(self) -> List[str]:
        """Get list of available gage IDs."""
        gages = []

        if not self.data_source_path.exists():
            return gages

        for huc_dir in self.data_source_path.iterdir():
            if huc_dir.is_dir() and huc_dir.name.isdigit():
                for file in huc_dir.glob("*_streamflow_qc.txt"):
                    gage_id = file.name.split('_')[0]
                    gages.append(gage_id)

        return sorted(set(gages))