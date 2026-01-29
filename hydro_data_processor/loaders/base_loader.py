"""
Base data loader class for Hydro Data Processor.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

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

        logger.debug(f"{self.__class__.__name__} initialized for {data_type}")

    @abstractmethod
    def load(self,
             basin_ids: Optional[List[str]] = None,
             **kwargs) -> pd.DataFrame:
        """Load data for specified basins."""
        pass

    def validate(self) -> bool:
        """Validate loaded data."""
        return self.data is not None

    def get_info(self) -> Dict:
        """Get information about loaded data."""
        info = {
            "data_type": self.data_type,
            "has_data": self.data is not None,
            "metadata": self.metadata
        }

        if self.data is not None:
            info["basin_count"] = len(
                self.data['basin_id'].unique()) if 'basin_id' in self.data.columns else 0
            info["total_records"] = len(self.data)

        return info

    def filter_by_date(
            self,
            df: pd.DataFrame,
            date_column: str = 'date') -> pd.DataFrame:
        """Filter data to study period (2001-01-01 to 2021-09-30)."""
        if df.empty or date_column not in df.columns:
            return df

        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        mask = (df[date_column] >= self.study_start_date) & (
            df[date_column] <= self.study_end_date)
        return df[mask].reset_index(drop=True)

    def calculate_missing_rate(
            self,
            df: pd.DataFrame,
            value_column: str) -> float:
        """Calculate missing data rate for a specific column."""
        if df.empty or value_column not in df.columns:
            return 1.0
        missing = df[value_column].isna().sum()
        return missing / len(df) if len(df) > 0 else 1.0
