"""
SMAP loader for Hydro Data Processor
Loads soil moisture data from NASA-USDA Enhanced SMAP
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import numpy as np
from datetime import datetime

from .base_loader import BaseDataLoader
from hydro_data_processor.config.settings import DataSourceConfig

logger = logging.getLogger(__name__)


class SMAPLoader(BaseDataLoader):
    """Loader for soil moisture data from SMAP."""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config, "smap")
        
        # SMAP characteristics
        self.temporal_resolution = 3  # 3-day timestep with gaps
        
    def load(self, basin_ids: Optional[List[str]] = None,
             start_date: Optional[str] = None,
             end_date: Optional[str] = None,
             **kwargs) -> pd.DataFrame:
        """
        Load SMAP soil moisture data for specified basins.
        
        Args:
            basin_ids: List of basin IDs to load
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with SMAP data
        """
        logger.info(f"Loading SMAP data for {len(basin_ids) if basin_ids else 'all'} basins")
        
        # SMAP data starts from 2015
        if start_date is None:
            start_date = self.study_start_date
        if end_date is None:
            end_date = self.study_end_date
        
        if basin_ids is None:
            logger.error("basin_ids must be provided for SMAP data")
            return pd.DataFrame()
        
        all_data = []
        
        for basin_id in basin_ids:
            basin_data = self._load_basin_smap(basin_id, start_date, end_date)
            if basin_data is not None and not basin_data.empty:
                all_data.append(basin_data)
            else:
                logger.warning(f"Failed to load SMAP data for basin {basin_id}")
        
        if not all_data:
            logger.error("No SMAP data loaded for any basin")
            return pd.DataFrame()
        
        # Combine data
        df = pd.concat(all_data, ignore_index=True)
        
        # Filter by date range
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
            df = df[mask].copy()
        
        self.data = df
        self.metadata = {
            'source': 'NASA-USDA Enhanced SMAP',
            'temporal_pattern': '3-day timestep with gaps',
            'basin_count': len(basin_ids),
            'period': f"{start_date} to {end_date}"
        }
        
        logger.info(f"Loaded SMAP data for {len(all_data)} basins")
        return df
    
    def _load_basin_smap(self, basin_id: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load SMAP data for a single basin."""
        # Load from file
        df = self._load_single_file(basin_id, header=None)
        
        if df is None or df.empty:
            logger.warning(f"No SMAP file found for basin {basin_id}")
            return None
        
        # Check if we have enough columns
        if len(df.columns) < 2:
            logger.error(f"SMAP file for basin {basin_id} has insufficient columns: {len(df.columns)}")
            return None
        
        # Assign column names
        df.columns = ['date', 'ssm']
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        
        # Add basin_id if not present
        if 'basin_id' not in df.columns:
            df['basin_id'] = basin_id
        
        return df