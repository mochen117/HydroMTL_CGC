# attribute_loader.py
"""
Attribute loader for Hydro Data Processor
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import numpy as np

from .base_loader import BaseDataLoader
from hydro_data_processor.config.settings import DataSourceConfig

logger = logging.getLogger(__name__)


class AttributeLoader(BaseDataLoader):
    """Loader for basin attributes."""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config, "attributes")
        
    def load(self, max_basins: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """
        Load basin attributes.
        
        Args:
            max_basins: Maximum number of basins to load
            
        Returns:
            DataFrame with basin attributes
        """
        logger.info(f"Loading basin attributes from {self.config.data_source_path}")
        
        # Find attribute file
        attribute_file = self._find_attribute_file()
        if attribute_file is None:
            logger.error(f"No attribute file found in {self.config.data_source_path}")
            return pd.DataFrame()
        
        # Load the file
        df = self._load_attribute_file(attribute_file)
        
        if df.empty:
            logger.error(f"Attribute file is empty: {attribute_file}")
            return df
        
        # Ensure 'basin_id' column exists
        if 'basin_id' not in df.columns:
            logger.warning("'basin_id' column not found in attributes. Trying to identify...")
            df = self._identify_basin_id_column(df)
        
        # Limit number of basins if specified
        if max_basins is not None and len(df) > max_basins:
            logger.info(f"Limiting to {max_basins} basins")
            df = df.head(max_basins)
        
        logger.info(f"Loaded attributes for {len(df)} basins")
        return df
    
    def _find_attribute_file(self) -> Optional[Path]:
        """Find the attribute file."""
        data_dir = self.config.data_source_path
        
        if not data_dir.exists():
            logger.error(f"Attribute directory does not exist: {data_dir}")
            return None
        
        # Look for common attribute file names
        possible_files = [
            "basin_attributes.txt",
            "basin_attributes.csv",
            "attributes.txt", 
            "attributes.csv",
            "camels_attributes.txt",
            "camels_attributes.csv",
        ]
        
        for fname in possible_files:
            file_path = data_dir / fname
            if file_path.exists():
                logger.debug(f"Found attribute file: {file_path}")
                return file_path
        
        # Try to find any text or CSV file
        for ext in ['.txt', '.csv']:
            for file_path in data_dir.glob(f"*{ext}"):
                if file_path.is_file():
                    logger.debug(f"Found potential attribute file: {file_path}")
                    return file_path
        
        logger.error(f"No attribute file found in {data_dir}")
        return None
    
    def _load_attribute_file(self, file_path: Path) -> pd.DataFrame:
        """Load attribute file with appropriate format."""
        try:
            # Try to detect the format
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
            
            # Check if it's semicolon separated (based on your logs)
            if ';' in first_line:
                logger.info(f"Loading semicolon-separated attributes from {file_path}")
                df = pd.read_csv(file_path, sep=';', header=None)
                
                # If the file has no header, name columns
                if df.shape[1] > 0:
                    # First column is basin_id
                    df = df.rename(columns={0: 'basin_id'})
                    
                    # Name remaining columns
                    for i in range(1, df.shape[1]):
                        df = df.rename(columns={i: f'attr_{i}'})
                
                return df
            
            # Try comma-separated
            elif ',' in first_line:
                logger.info(f"Loading comma-separated attributes from {file_path}")
                return pd.read_csv(file_path)
            
            # Try tab or space separated
            else:
                logger.info(f"Loading space/tab separated attributes from {file_path}")
                try:
                    return pd.read_csv(file_path, sep='\s+')
                except:
                    # Last resort: read with default separator
                    return pd.read_csv(file_path)
                    
        except Exception as e:
            logger.error(f"Error loading attribute file {file_path}: {e}")
            return pd.DataFrame()
    
    def _identify_basin_id_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Try to identify which column contains basin IDs."""
        df = df.copy()
        
        # Look for column with 8-digit numbers
        for col in df.columns:
            if df[col].dtype in [object, str]:
                # Check if values look like basin IDs
                sample = df[col].head(10).astype(str)
                if sample.str.match(r'^\d{8}').all():
                    logger.info(f"Identified basin_id column: {col}")
                    df = df.rename(columns={col: 'basin_id'})
                    return df
        
        # If no column found, create a dummy basin_id column
        logger.warning("Could not identify basin_id column. Creating dummy IDs.")
        df['basin_id'] = [f"basin_{i:04d}" for i in range(len(df))]
        
        return df