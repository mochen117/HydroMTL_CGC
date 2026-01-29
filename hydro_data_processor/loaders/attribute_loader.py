"""
Attribute loader for CAMELS basin attributes.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from .base_loader import BaseDataLoader
from hydro_data_processor.config.settings import DataSourceConfig
from hydro_data_processor.utils.huc_mapper import HucMapper

logger = logging.getLogger(__name__)


class AttributeLoader(BaseDataLoader):
    """Loader for CAMELS basin attributes."""

    ATTRIBUTE_FILES = {
        "name": "camels_name.txt",
        "hydro": "camels_hydro.txt",
        "clim": "camels_clim.txt",
        "topo": "camels_topo.txt",
        "soil": "camels_soil.txt",
        "vege": "camels_vege.txt",
        "geol": "camels_geol.txt",
    }

    def __init__(self, config: DataSourceConfig):
        super().__init__(config, "attributes")

        self.huc_mapper = HucMapper(self.config.data_source_path.parent.parent)
        self._attributes_cache: Dict[str, pd.DataFrame] = {}

        logger.debug(f"AttributeLoader initialized")

    def load(self, max_basins: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """
        Load basin attributes from all CAMELS attribute files.

        Args:
            max_basins: Maximum number of basins to load

        Returns:
            Merged DataFrame with all attributes
        """
        logger.info("Loading CAMELS basin attributes")

        name_df = self._load_attribute_file("name", self.ATTRIBUTE_FILES["name"])
        if name_df is None or name_df.empty:
            logger.error("Failed to load basin names")
            return pd.DataFrame()

        merged_df = name_df

        for attr_type, filename in self.ATTRIBUTE_FILES.items():
            if attr_type == "name":
                continue

            try:
                attr_df = self._load_attribute_file(attr_type, filename)
                if attr_df is not None and not attr_df.empty:
                    merged_df = pd.merge(merged_df, attr_df, on="gauge_id", how="left")
                    logger.debug(f"Merged {attr_type} attributes: {len(attr_df)} basins")
            except Exception as e:
                logger.error(f"Failed to load {attr_type} from {filename}: {e}")

        valid_mask = merged_df['gauge_id'].apply(self.huc_mapper.validate_basin)
        invalid_count = (~valid_mask).sum()

        if invalid_count > 0:
            logger.warning(f"Removing {invalid_count} invalid basin IDs")
            merged_df = merged_df[valid_mask].copy()

        if 'huc_02' not in merged_df.columns:
            logger.debug("huc_02 column not found in attributes")
            merged_df['huc_02'] = merged_df['gauge_id'].apply(
                lambda x: str(x)[:2] if len(str(x)) >= 2 else '01'
            )
        
        merged_df['huc_02'] = merged_df['huc_02'].astype(str).str.zfill(2)

        huc_mapping = self._extract_huc_from_camels_name()
        if huc_mapping is not None and not huc_mapping.empty:
            for idx, row in merged_df.iterrows():
                gauge_id = row['gauge_id']
                if gauge_id in huc_mapping.index:
                    merged_df.at[idx, 'huc_02'] = huc_mapping.loc[gauge_id]

        if max_basins and max_basins < len(merged_df):
            merged_df = merged_df.head(max_basins)
            logger.info(f"Limited to {max_basins} basins")

        self.metadata = {
            "total_basins": len(merged_df),
            "attribute_files": list(self.ATTRIBUTE_FILES.keys()),
            "columns": list(merged_df.columns)
        }

        self.data = merged_df
        logger.info(f"Loaded attributes for {len(merged_df)} basins")
        logger.debug(f"Available columns: {list(merged_df.columns)}")
        if 'huc_02' in merged_df.columns:
            logger.debug(f"HUC2 values sample: {merged_df['huc_02'].head().tolist()}")
        return merged_df

    def _load_attribute_file(self, attr_type: str,
                             filename: str) -> Optional[pd.DataFrame]:
        """Load a single attribute file."""
        file_path = self.config.data_source_path / filename

        if not file_path.exists():
            logger.debug(f"Attribute file not found: {file_path}")
            return None

        try:
            df = pd.read_csv(file_path, delimiter=';')

            if 'gauge_id' not in df.columns:
                logger.error(f"No gauge_id column in {filename}")
                return None

            df['gauge_id'] = df['gauge_id'].astype(str).str.strip()

            return df

        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return None

    def _extract_huc_from_camels_name(self) -> Optional[pd.Series]:
        """Extract HUC2 mapping from camels_name.txt if available."""
        possible_paths = [
            self.config.data_source_path / "camels_name.txt",
            self.config.data_source_path.parent / "camels_name.txt",
            Path("/home/mochen/hydro_data/camels/camels_us/camels_name.txt"),
        ]
        
        for name_file_path in possible_paths:
            if name_file_path.exists():
                try:
                    name_df = pd.read_csv(name_file_path, delimiter=';')
                    logger.info(f"Loaded camels_name.txt from {name_file_path}")
                    
                    if 'huc_02' not in name_df.columns:
                        logger.debug("camels_name.txt does not contain huc_02 column")
                        return None
                        
                    huc_series = pd.Series(
                        name_df['huc_02'].values, 
                        index=name_df['gauge_id'].astype(str).str.strip()
                    )
                    huc_series = huc_series.astype(str).str.zfill(2)
                    
                    logger.info(f"Extracted HUC2 mapping from camels_name.txt: {len(huc_series)} records")
                    return huc_series
                    
                except Exception as e:
                    logger.debug(f"Failed to extract HUC from camels_name.txt: {e}")
        
        logger.debug("camels_name.txt not found in any expected location")
        return None

    def get_basin_attributes(self, gauge_id: str) -> Dict[str, Any]:
        """Get all attributes for a specific gauge."""
        if not self.data or self.data.empty:
            logger.warning("Attributes not loaded yet")
            return {}

        gauge_row = self.data[self.data['gauge_id'] == gauge_id]
        if gauge_row.empty:
            logger.warning(f"No attributes found for gauge {gauge_id}")
            return {}

        return gauge_row.iloc[0].to_dict()