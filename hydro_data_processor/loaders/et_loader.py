"""
ET loader for Hydro Data Processor
Loads evapotranspiration data from MOD16A2
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import numpy as np
from datetime import datetime, timedelta

from .base_loader import BaseDataLoader
from hydro_data_processor.config.settings import DataSourceConfig

logger = logging.getLogger(__name__)


class ETLoader(BaseDataLoader):
    """Loader for evapotranspiration data from MOD16A2."""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config, "et")
        
        # MOD16A2 characteristics
        self.temporal_resolution = 8  # 8-day composite
    
    def load(self, basin_ids: Optional[List[str]] = None,
             start_date: Optional[str] = None,
             end_date: Optional[str] = None,
             **kwargs) -> pd.DataFrame:
        """
        Load ET data for specified basins.
        
        Args:
            basin_ids: List of basin IDs to load
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with ET data
        """
        logger.info(f"Loading ET data for {len(basin_ids) if basin_ids else 'all'} basins")
        
        # MODIS data starts from 2001
        if start_date is None:
            start_date = self.study_start_date
        if end_date is None:
            end_date = self.study_end_date
        
        if basin_ids is None:
            logger.error("basin_ids must be provided for ET data")
            return pd.DataFrame()
        
        all_data = []
        
        for basin_id in basin_ids:
            basin_data = self._load_basin_et(basin_id, start_date, end_date)
            if basin_data is not None and not basin_data.empty:
                all_data.append(basin_data)
            else:
                logger.warning(f"Failed to load ET data for basin {basin_id}")
        
        if not all_data:
            logger.error("No ET data loaded for any basin")
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
            'source': 'MOD16A2 v006',
            'temporal_resolution': '8-day composite',
            'basin_count': len(basin_ids),
            'period': f"{start_date} to {end_date}"
        }
        
        logger.info(f"Loaded ET data for {len(all_data)} basins")
        return df
    
    def _load_basin_et(self, basin_id: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load ET data for a single basin."""
        # Load from file
        df = self._load_single_file(basin_id, header=None)
        
        if df is None or df.empty:
            logger.warning(f"No ET file found for basin {basin_id}")
            return None
        
        # Check if we have enough columns
        if len(df.columns) < 2:
            logger.error(f"ET file for basin {basin_id} has insufficient columns: {len(df.columns)}")
            return None
        
        # Assign column names
        df.columns = ['date', 'et']
        
        # Convert date column (MOD16A2 uses 8-day composite dates)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        
        # Add basin_id if not present
        if 'basin_id' not in df.columns:
            df['basin_id'] = basin_id
        
        return df
    
    def interpolate_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate 8-day composite ET data to daily resolution.
        
        Args:
            df: DataFrame with 8-day composite ET data
            
        Returns:
            DataFrame with daily ET values
        """
        if df.empty:
            return df
        
        daily_data = []
        
        for _, row in df.iterrows():
            start_date = row['date']
            basin_id = row['basin_id']
            et_value = row['et']
            
            # MOD16A2 values are sum over 8 days, convert to daily average
            daily_et = et_value / 8
            
            for day in range(8):
                current_date = start_date + timedelta(days=day)
                daily_data.append({
                    'basin_id': basin_id,
                    'date': current_date,
                    'et_daily': daily_et
                })
        
        return pd.DataFrame(daily_data)