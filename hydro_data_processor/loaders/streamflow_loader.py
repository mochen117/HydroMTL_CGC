"""
Streamflow loader for Hydro Data Processor
Loads streamflow data from CAMELS and USGS
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
import numpy as np
from datetime import datetime
import re

from .base_loader import BaseDataLoader
from hydro_data_processor.config.settings import DataSourceConfig

logger = logging.getLogger(__name__)


class StreamflowLoader(BaseDataLoader):
    """Loader for streamflow data."""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config, "streamflow")
        
        # Streamflow quality threshold (5% missing data)
        self.max_missing_rate = 0.05
        
        # Streamflow column names in order of priority
        self.streamflow_columns = ['streamflow', 'q_obs', 'discharge', 'Q', 'flow']
        
        # Cache for file paths to avoid repeated directory scanning
        self._file_path_cache = {}
        
        logger.debug(f"StreamflowLoader initialized for source: {config.data_source_path}")
    
    def load(self, basin_ids: Optional[List[str]] = None,
             start_date: Optional[str] = None,
             end_date: Optional[str] = None,
             **kwargs) -> pd.DataFrame:
        """
        Load streamflow data for specified basins.
        
        Args:
            basin_ids: List of basin IDs to load
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with streamflow data
        """
        if not basin_ids:
            logger.error("basin_ids must be provided for streamflow data")
            return pd.DataFrame()
        
        logger.info(f"Loading streamflow data for {len(basin_ids)} basins")
        
        # Use study period if dates not specified
        if start_date is None:
            start_date = self.study_start_date
        if end_date is None:
            end_date = self.study_end_date
        
        logger.debug(f"Date range: {start_date} to {end_date}")
        
        all_data = []
        valid_basins = []
        failed_basins = []
        
        for i, basin_id in enumerate(basin_ids):
            if i % 100 == 0 and i > 0:
                logger.info(f"Processed {i}/{len(basin_ids)} basins")
            
            # Ensure basin_id is in correct format
            formatted_basin_id = self._format_basin_id(basin_id)
            if not formatted_basin_id:
                logger.warning(f"Invalid basin_id format: {basin_id}")
                failed_basins.append(basin_id)
                continue
            
            logger.debug(f"Loading streamflow for basin {formatted_basin_id}")
            
            # Load basin data
            basin_data = self._load_basin_streamflow(formatted_basin_id, start_date, end_date)
            
            if basin_data is not None and not basin_data.empty:
                # Check missing data rate
                missing_rate = self._calculate_missing_rate(basin_data)
                
                if missing_rate <= self.max_missing_rate:
                    all_data.append(basin_data)
                    valid_basins.append(formatted_basin_id)
                    logger.debug(f"Basin {formatted_basin_id}: {len(basin_data)} records, missing rate: {missing_rate:.2%}")
                else:
                    logger.warning(f"Basin {formatted_basin_id}: missing rate {missing_rate:.2%} > {self.max_missing_rate:.0%} - REJECTED")
                    failed_basins.append(formatted_basin_id)
            else:
                logger.warning(f"No data loaded for basin {formatted_basin_id}")
                failed_basins.append(formatted_basin_id)
        
        if not all_data:
            logger.error(f"No valid streamflow data found for any basin. Failed: {len(failed_basins)} basins")
            return pd.DataFrame()
        
        # Combine data
        if len(all_data) == 1:
            df = all_data[0]
        else:
            df = pd.concat(all_data, ignore_index=True)
        
        # Filter by date range (redundant but safe)
        if 'date' in df.columns and not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            mask = (df['date'] >= start_ts) & (df['date'] <= end_ts)
            df = df[mask].copy()
        
        self.data = df
        self.metadata = {
            'source': str(self.config.data_source_path),
            'valid_basins': valid_basins,
            'basin_count': len(valid_basins),
            'failed_basins': failed_basins,
            'failed_count': len(failed_basins),
            'period': f"{start_date} to {end_date}",
            'missing_rate_threshold': self.max_missing_rate,
            'total_records': len(df)
        }
        
        logger.info(f"Loaded streamflow data: {len(valid_basins)} basins, {len(df)} records")
        logger.info(f"Failed basins: {len(failed_basins)}")
        
        return df
    
    def _format_basin_id(self, basin_id: str) -> Optional[str]:
        """Format basin ID to 8-digit string."""
        if basin_id is None or pd.isna(basin_id):
            return None
        
        basin_str = str(basin_id).strip()
        
        # Remove quotes if present
        basin_str = basin_str.strip("'\"")
        
        # Extract digits
        digits = re.findall(r'\d+', basin_str)
        if not digits:
            return None
        
        # Take the longest digit sequence
        longest_digits = max(digits, key=len)
        
        # Ensure it's 8 digits
        if len(longest_digits) != 8:
            # Try padding with zeros
            if longest_digits.isdigit():
                padded = longest_digits.zfill(8)
                if len(padded) == 8:
                    return padded
            return None
        
        return longest_digits
    
    def _find_basin_file(self, basin_id: str) -> Optional[Path]:
        """Find streamflow file for a basin ID."""
        # Check cache first
        if basin_id in self._file_path_cache:
            return self._file_path_cache[basin_id]
        
        data_dir = self.config.data_source_path
        
        # Check if it's a directory or file
        if not data_dir.exists():
            logger.error(f"Data directory does not exist: {data_dir}")
            return None
        
        # Look for files with basin_id in name
        possible_patterns = [
            f"{basin_id}.txt",
            f"{basin_id}.csv",
            f"{basin_id}_streamflow.txt",
            f"{basin_id}_streamflow.csv",
            f"{basin_id}_Q.txt",
            f"{basin_id}_Q.csv",
        ]
        
        for pattern in possible_patterns:
            file_path = data_dir / pattern
            if file_path.exists():
                logger.debug(f"Found streamflow file: {file_path}")
                self._file_path_cache[basin_id] = file_path
                return file_path
        
        # If not found by exact name, search for files containing basin_id
        if data_dir.is_dir():
            for file_path in data_dir.glob("*"):
                if file_path.is_file() and basin_id in file_path.stem:
                    logger.debug(f"Found streamflow file by partial match: {file_path}")
                    self._file_path_cache[basin_id] = file_path
                    return file_path
        
        logger.warning(f"No streamflow file found for basin {basin_id} in {data_dir}")
        return None
    
    def _load_basin_streamflow(self, basin_id: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load streamflow data for a single basin."""
        # Find file
        file_path = self._find_basin_file(basin_id)
        if file_path is None:
            return None
        
        try:
            # Determine file format and load
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                # Assume text file with space/tab separation
                df = pd.read_csv(file_path, sep='\s+', header=None)
            
            if df.empty:
                logger.warning(f"Empty streamflow file for basin {basin_id}: {file_path}")
                return None
            
            # Parse columns
            df = self._parse_streamflow_columns(df, basin_id)
            
            if df is None or df.empty:
                return None
            
            # Ensure date column exists
            if 'date' not in df.columns:
                df = self._create_date_column(df, basin_id)
            
            if df is None or 'date' not in df.columns:
                logger.error(f"Cannot create date column for basin {basin_id}")
                return None
            
            # Filter by date range
            df['date'] = pd.to_datetime(df['date'])
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            mask = (df['date'] >= start_ts) & (df['date'] <= end_ts)
            df = df[mask].copy()
            
            if df.empty:
                logger.warning(f"No data in date range for basin {basin_id}")
                return None
            
            # Add basin_id if not present
            if 'basin_id' not in df.columns:
                df['basin_id'] = basin_id
            
            return df[['basin_id', 'date', 'streamflow']]
            
        except Exception as e:
            logger.error(f"Error loading streamflow for basin {basin_id} from {file_path}: {e}")
            return None
    
    def _parse_streamflow_columns(self, df: pd.DataFrame, basin_id: str) -> Optional[pd.DataFrame]:
        """Parse streamflow columns from raw data."""
        df = df.copy()
        
        # Check for column names (case-insensitive)
        df.columns = [str(col).lower().strip() for col in df.columns]
        
        # Try to find streamflow column
        streamflow_col = None
        for col_pattern in self.streamflow_columns:
            for col in df.columns:
                if col_pattern in col:
                    streamflow_col = col
                    break
            if streamflow_col:
                break
        
        # If no named streamflow column, check column content
        if streamflow_col is None:
            # Look for numeric columns with reasonable range
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Check if values look like streamflow (positive values)
                    non_null = df[col].dropna()
                    if len(non_null) > 0:
                        if non_null.min() >= 0 and non_null.max() < 10000:  # Reasonable range
                            streamflow_col = col
                            break
        
        # If still no streamflow column, assume 4th column (CAMELS format)
        if streamflow_col is None and len(df.columns) >= 4:
            streamflow_col = df.columns[3]  # 4th column (0-indexed)
            logger.debug(f"Assuming column {streamflow_col} as streamflow for basin {basin_id}")
        
        if streamflow_col is None:
            logger.error(f"Cannot identify streamflow column for basin {basin_id}. Columns: {list(df.columns)}")
            return None
        
        # Rename to 'streamflow'
        df = df.rename(columns={streamflow_col: 'streamflow'})
        
        # Find date columns
        date_cols = []
        for col in ['date', 'time', 'datetime', 'year', 'month', 'day']:
            if col in df.columns:
                date_cols.append(col)
        
        # If no date columns, check for year/month/day pattern
        if not date_cols and len(df.columns) >= 3:
            # Check if first three columns could be year, month, day
            potential_date_cols = df.columns[:3]
            if all(pd.api.types.is_numeric_dtype(df[col]) for col in potential_date_cols):
                # Check ranges
                if (df[potential_date_cols[0]].min() >= 1900 and 
                    df[potential_date_cols[0]].max() <= 2100 and
                    df[potential_date_cols[1]].min() >= 1 and 
                    df[potential_date_cols[1]].max() <= 12):
                    
                    df = df.rename(columns={
                        potential_date_cols[0]: 'year',
                        potential_date_cols[1]: 'month',
                        potential_date_cols[2]: 'day'
                    })
                    date_cols = ['year', 'month', 'day']
        
        # If we have date components, store them
        if 'year' in df.columns and 'month' in df.columns and 'day' in df.columns:
            # Already have year/month/day columns
            pass
        elif date_cols:
            # We have some date columns, keep them
            pass
        
        return df
    
    def _create_date_column(self, df: pd.DataFrame, basin_id: str) -> Optional[pd.DataFrame]:
        """Create date column from year, month, day columns."""
        if 'year' in df.columns and 'month' in df.columns and 'day' in df.columns:
            try:
                # Create date column
                df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
                return df
            except Exception as e:
                logger.error(f"Cannot create date from year/month/day for basin {basin_id}: {e}")
        
        # Check if there's a datetime column
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df['date'] = pd.to_datetime(df[col])
                    return df
                except:
                    continue
        
        logger.error(f"Cannot create date column for basin {basin_id}")
        return None
    
    def _calculate_missing_rate(self, df: pd.DataFrame) -> float:
        """Calculate missing data rate for streamflow."""
        if df.empty or 'streamflow' not in df.columns:
            return 1.0
        
        total_records = len(df)
        missing_records = df['streamflow'].isna().sum()
        
        return missing_records / total_records if total_records > 0 else 1.0
    
    def calculate_statistics(self) -> Dict[str, float]:
        """Calculate streamflow statistics."""
        if self.data is None or self.data.empty:
            return {}
        
        stats = {}
        
        if 'streamflow' in self.data.columns:
            streamflow = self.data['streamflow'].dropna()
            if len(streamflow) > 0:
                stats['mean'] = float(streamflow.mean())
                stats['std'] = float(streamflow.std())
                stats['min'] = float(streamflow.min())
                stats['max'] = float(streamflow.max())
                stats['median'] = float(streamflow.median())
                stats['q25'] = float(streamflow.quantile(0.25))
                stats['q75'] = float(streamflow.quantile(0.75))
                stats['missing_rate'] = float(self.data['streamflow'].isna().mean())
                stats['count'] = int(len(streamflow))
                stats['total_records'] = int(len(self.data))
        
        return stats
    
    def get_file_paths(self) -> Dict[str, str]:
        """Get file paths for all loaded basins."""
        return {k: str(v) for k, v in self._file_path_cache.items()}