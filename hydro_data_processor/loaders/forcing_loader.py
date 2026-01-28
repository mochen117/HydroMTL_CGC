"""
Forcing loader for Hydro Data Processor
Loads meteorological forcing data from NLDAS-2
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


class ForcingLoader(BaseDataLoader):
    """Loader for meteorological forcing data from NLDAS-2."""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config, "forcing")
        
        # Forcing variables from NLDAS-2
        self.forcing_variables = [
            'dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp'
        ]
        
        # Variable descriptions
        self.variable_descriptions = {
            'dayl': 'Day length (s)',
            'prcp': 'Precipitation (mm/day)',
            'srad': 'Shortwave radiation (W/m²)',
            'swe': 'Snow water equivalent (mm)',
            'tmax': 'Maximum temperature (C)',
            'tmin': 'Minimum temperature (C)',
            'vp': 'Vapor pressure (Pa)'
        }
    
    def load(self, basin_ids: Optional[List[str]] = None,
             start_date: Optional[str] = None,
             end_date: Optional[str] = None,
             **kwargs) -> pd.DataFrame:
        """
        Load forcing data for specified basins.
        
        Args:
            basin_ids: List of basin IDs to load
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with forcing data
        """
        logger.info(f"Loading forcing data for {len(basin_ids) if basin_ids else 'all'} basins")
        
        # Use study period if dates not specified
        if start_date is None:
            start_date = self.study_start_date
        if end_date is None:
            end_date = self.study_end_date
        
        if basin_ids is None:
            logger.error("basin_ids must be provided for forcing data")
            return pd.DataFrame()
        
        all_data = []
        
        for basin_id in basin_ids:
            basin_data = self._load_basin_forcing(basin_id, start_date, end_date)
            if basin_data is not None and not basin_data.empty:
                all_data.append(basin_data)
            else:
                logger.warning(f"Failed to load forcing data for basin {basin_id}")
        
        if not all_data:
            logger.error("No forcing data loaded for any basin")
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
            'source': 'NLDAS-2',
            'basin_count': len(basin_ids),
            'period': f"{start_date} to {end_date}",
            'variables': self.forcing_variables
        }
        
        logger.info(f"Loaded forcing data for {len(all_data)} basins")
        return df
    
    def _load_basin_forcing(self, basin_id: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load forcing data for a single basin."""
        # Load from file
        df = self._load_single_file(basin_id, header=None)
        
        if df is None or df.empty:
            logger.warning(f"No forcing file found for basin {basin_id}")
            return None
        
        # Check if we have enough columns
        if len(df.columns) < 8:
            logger.error(f"Forcing file for basin {basin_id} has insufficient columns: {len(df.columns)}")
            return None
        
        # Assign column names based on CAMELS NLDAS forcing format
        column_names = ['date', 'dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
        if len(df.columns) > 8:
            column_names += [f'extra_{i}' for i in range(len(df.columns) - 8)]
        
        df.columns = column_names
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        
        # Add basin_id if not present
        if 'basin_id' not in df.columns:
            df['basin_id'] = basin_id
        
        return df