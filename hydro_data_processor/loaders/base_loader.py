"""
Base data loader class for Hydro Data Processor
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime

from hydro_data_processor.config.settings import DataSourceConfig

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """Base class for all data loaders."""
    
    def __init__(self, config: DataSourceConfig, data_type: str):
        """
        Initialize base data loader.
        
        Args:
            config: Data source configuration
            data_type: Type of data being loaded
        """
        self.config = config
        self.data_type = data_type
        self.data = None
        self.metadata = {}
        
        # Study period from the paper
        self.study_start_date = pd.Timestamp("2001-01-01")
        self.study_end_date = pd.Timestamp("2021-09-30")
        
        # Cache for loaded files
        self.file_cache = {}
    
    @abstractmethod
    def load(self, basin_ids: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
        """Load data for specified basins."""
        pass
    
    def validate(self) -> bool:
        """Validate loaded data."""
        return self.data is not None
    
    def get_info(self) -> Dict:
        """Get information about loaded data."""
        return {
            "data_type": self.data_type,
            "has_data": self.data is not None,
            "basin_count": len(self.data) if self.data is not None else 0,
            "metadata": self.metadata
        }
    
    def filter_by_study_period(self, df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """Filter data to study period (2001-01-01 to 2021-09-30)."""
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            mask = (df[date_column] >= self.study_start_date) & (df[date_column] <= self.study_end_date)
            return df[mask].copy()
        return df
    
    def calculate_missing_rate(self, df: pd.DataFrame, value_column: str) -> float:
        """Calculate missing data rate for a specific column."""
        if df.empty or value_column not in df.columns:
            return 1.0
        return df[value_column].isna().sum() / len(df)
    
    def _build_file_path(self, basin_id: str) -> Path:
        """
        Build file path for a basin based on configuration.
        
        Args:
            basin_id: 8-digit basin ID (e.g., '01013500')
            
        Returns:
            Path object for the data file
        """
        # Extract HUC2 from basin ID (first 2 digits)
        huc2 = basin_id[:2]
        
        # Build subdirectory path with HUC2 substitution
        subdirectory = self.config.subdirectory.format(huc2=huc2) if self.config.subdirectory else ""
        
        # Build filename with basin_id substitution
        filename = self.config.file_pattern.format(basin_id=basin_id)
        
        # Construct full path
        if subdirectory:
            file_path = self.config.data_source_path / subdirectory / filename
        else:
            file_path = self.config.data_source_path / filename
        
        return file_path
    
    def _load_single_file(self, basin_id: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Load data from a single file for a basin.
        
        Args:
            basin_id: 8-digit basin ID
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame with loaded data or None if file not found
        """
        file_path = self._build_file_path(basin_id)
        
        if not file_path.exists():
            logger.debug(f"File not found: {file_path}")
            return None
        
        # Use cached data if available
        if basin_id in self.file_cache:
            logger.debug(f"Using cached data for basin {basin_id}")
            return self.file_cache[basin_id].copy()
        
        try:
            # Default delimiter from config
            delimiter = kwargs.get('sep', self.config.delimiter)
            
            df = pd.read_csv(
                file_path,
                sep=delimiter,
                engine='python',
                **{k: v for k, v in kwargs.items() if k != 'sep'}
            )
            
            # Add basin_id if not present
            if 'basin_id' not in df.columns:
                df['basin_id'] = basin_id
            
            # Cache the result
            self.file_cache[basin_id] = df.copy()
            
            logger.debug(f"Successfully loaded data from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return None