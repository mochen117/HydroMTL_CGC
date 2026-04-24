"""
Attribute loader for CAMELS basin attributes.
Categorical variables are label-encoded (missing -> -1). No one-hot encoding.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from .base_loader import BaseDataLoader
from hydro_data_processor.config.settings import DataSourceConfig
from hydro_data_processor.utils.huc_mapper import HucMapper

logger = logging.getLogger(__name__)


class AttributeLoader(BaseDataLoader):
    """Loader for CAMELS basin attributes with detailed English annotations."""

    ATTRIBUTE_FILES = {
        "name": "camels_name.txt",
        "vege": "camels_vege.txt",
        "soil": "camels_soil.txt",
        "geol": "camels_geol.txt",
        "topo": "camels_topo.txt",
        "clim": "camels_clim.txt",
    }
    
    # Mapping and English Descriptions
    STATIC_VARIABLE_MAPPING = {
        # --- Topographic & Geometric Attributes (camels_topo.txt) ---
        'elev_mean': 'elev_mean',            # Mean elevation [m]
        'slope_mean': 'slope_mean',          # Mean slope [m/km]
        'area_gages2': 'area_gages2',        # Basin area [km2]
        
        # --- Vegetation Attributes (camels_vege.txt) ---
        'frac_forest': 'pct_forest',         # Forest fraction [0-1]
        'lai_max': 'lai_max',                # Maximum Leaf Area Index [-]
        'lai_diff': 'lai_diff',              # Difference between max and min LAI [-]
        'gvf_max': 'gvf_max',                # Maximum Green Vegetation Fraction [0-1]
        'dom_land_cover': 'dom_land_cover',  # Dominant land cover type (Categorical)
        'dom_land_cover_frac': 'dom_land_cover_frac', # Fraction of dominant land cover [0-1]
        'root_depth_50': 'root_depth_50',    # Depth above which 50% of roots are located [m]
        
        # --- Soil Attributes (camels_soil.txt) ---
        'soil_depth_statgso': 'soil_depth_statsgo', # Soil depth to bedrock [mm] (STATSGO)
        'soil_porosity': 'soil_porosity',    # Soil porosity [-]
        'soil_conductivity': 'soil_conductivity', # Saturated hydraulic conductivity [cm/hr]
        'max_water_content': 'max_water_content', # Maximum soil water content [m]
        'sand_frac': 'sand_frac',            # Sand fraction in soil [%]
        'clay_frac': 'clay_frac',            # Clay fraction in soil [%]
        'organic_frac': 'organic_frac',      # Organic matter fraction [%]
        
        # --- Geologic Attributes (camels_geol.txt) ---
        'geol_class_1st': 'geol_1st_class',  # Primary geological class (Categorical)
        'geol_class_2nd': 'geol_2nd_class',  # Secondary geological class (Categorical)
        'geol_porosity': 'geol_porostiy',    # Geological porosity [-]
        'geol_permeability': 'geol_permeability', # Geological permeability [m2]
        'carbonate_rocks_frac': 'carbonate_rocks_frac', # Fraction of carbonate rocks [0-1]
        
        # --- Climatic Attributes (camels_clim.txt) ---
        'p_mean': 'p_mean',                  # Mean daily precipitation [mm/day]
        'pet_mean': 'pet_mean',              # Mean daily potential evapotranspiration [mm/day]
        'aridity': 'aridity',                # Aridity index (PET/P) [-]
        'p_seasonality': 'p_seasonality',    # Precipitation seasonality index [-]
        'frac_snow': 'frac_snow',            # Fraction of precipitation falling as snow [0-1]
    }
    
    # List of variables required for model input
    REQUIRED_STATIC_VARS = list(STATIC_VARIABLE_MAPPING.keys())
    
    # Variables that need Label Encoding
    CATEGORICAL_VARS = ['dom_land_cover', 'geol_class_1st', 'geol_class_2nd']
    
    HUC2_COL = 'huc_02' # 2-digit Hydrologic Unit Code for regional grouping

    def __init__(self, config: DataSourceConfig):
        super().__init__(config, "attributes")
        # Ensure path safety (no hardcoded absolute paths)
        self.huc_mapper = HucMapper(self.config.data_source_path.parent.parent)
        self.cat_mappings = {}
        logger.info("AttributeLoader initialized with expanded feature set.")

    def load(self, max_basins: Optional[int] = None, selected_basins: Optional[List[str]] = None,
             skip_validation: bool = False, **kwargs) -> pd.DataFrame:
        """
        Loads and processes CAMELS attributes.
        1. Loads data from multiple .txt files.
        2. Standardizes column names and units.
        3. Encodes categorical variables (Missing -> -1).
        4. Derives HUC2 codes for regional context.
        """
        logger.info("Merging CAMELS attribute files...")
        
        merged_df: Optional[pd.DataFrame] = None

        # Process each defined attribute file
        for attr_type, filename in self.ATTRIBUTE_FILES.items():
            df = self._load_attribute_file(attr_type, filename)
            if df is None or df.empty:
                continue
            
            df = self._format_gage_ids_batch(df, f"{attr_type} attributes")
            
            if merged_df is None:
                merged_df = df
            else:
                # Merge logic: Outer join on gage_id, ensuring no column duplication
                cols_to_use = df.columns.difference(merged_df.columns).tolist() + ['gage_id']
                merged_df = pd.merge(merged_df, df[cols_to_use], on="gage_id", how="outer")

        if merged_df is None or merged_df.empty:
            logger.error("Critical Failure: No attribute files were loaded.")
            return pd.DataFrame()

        # Step: Selection and Validation
        if selected_basins:
            formatted_selected = [self._format_gage_id(gid) for gid in selected_basins]
            merged_df = merged_df[merged_df['gage_id'].isin(formatted_selected)].copy()

        if not skip_validation:
            valid_mask = merged_df['gage_id'].apply(self.huc_mapper.validate_basin)
            merged_df = merged_df[valid_mask].copy()

        # Step: Feature Engineering & Standardization
        merged_df = self._standardize_column_names(merged_df)
        merged_df = self._ensure_required_variables(merged_df)

        # Step: Regional Categorization (HUC 2-digit)
        if self.HUC2_COL not in merged_df.columns:
            merged_df[self.HUC2_COL] = merged_df['gage_id'].str[:2].str.zfill(2)

        # Step: Stable Label Encoding for Categorical Data
        for var in self.CATEGORICAL_VARS:
            series = merged_df[var].fillna('missing').astype(str)
            # Sort unique values to ensure consistent mapping across different sessions
            unique_vals = sorted(series.unique())
            mapping = {val: (i if val != 'missing' else -1) for i, val in enumerate(unique_vals) if val != 'missing'}
            mapping['missing'] = -1
            
            merged_df[var] = series.map(mapping).astype(int)
            self.cat_mappings[var] = mapping

        # Step: Final Filtering of columns
        final_cols = ['gage_id', self.HUC2_COL] + self.REQUIRED_STATIC_VARS
        merged_df = merged_df[final_cols]

        if max_basins:
            merged_df = merged_df.head(max_basins)

        self.data = merged_df
        logger.info(f"Successfully loaded {len(merged_df)} basins with {len(self.REQUIRED_STATIC_VARS)} attributes.")
        return merged_df

    def _load_attribute_file(self, attr_type: str, filename: str) -> Optional[pd.DataFrame]:
        file_path = self.config.data_source_path / filename
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None
        try:
            # Note: CAMELS files usually use ';' as delimiter
            df = pd.read_csv(
                file_path, delimiter=';', dtype={'gauge_id': str, 'gage_id': str},
                na_values=['NA', 'NaN', 'nan', 'None', '-999', ''],
                keep_default_na=True, engine='python'
            )
            # Normalize ID column name
            if 'gauge_id' in df.columns:
                df = df.rename(columns={'gauge_id': 'gage_id'})
            return df
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")
            return None

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies name mapping and unit conversions (e.g., % to fraction)."""
        df = df.copy()
        if 'pct_forest' in df.columns:
            df['frac_forest'] = df['pct_forest'] / 100.0
            
        for target, source in self.STATIC_VARIABLE_MAPPING.items():
            if source in df.columns and target not in df.columns:
                df[target] = df[source]
        return df

    def _ensure_required_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing required columns with NaN to maintain consistent schema."""
        for var in self.REQUIRED_STATIC_VARS:
            if var not in df.columns:
                logger.debug(f"Missing variable {var}, filling with NaN.")
                df[var] = np.nan
        return df

    def _format_gage_id(self, gage_id_str: str) -> str:
        """Ensures 8-digit string format for USGS gage IDs."""
        clean_id = ''.join(filter(str.isdigit, str(gage_id_str)))
        return clean_id.zfill(8)

    def _format_gage_ids_batch(self, df: pd.DataFrame, context: str = "") -> pd.DataFrame:
        if 'gage_id' not in df.columns:
            return df
        df = df.copy()
        df['gage_id'] = df['gage_id'].apply(self._format_gage_id)
        return df

    def get_basin_attributes(self, gage_id: str) -> Dict[str, Any]:
        """Returns a clean dictionary of attributes for a specific basin."""
        if self.data is None or self.data.empty:
            return {}
        
        row = self.data[self.data['gage_id'] == self._format_gage_id(gage_id)]
        if row.empty:
            return {}
            
        attrs = row.iloc[0].to_dict()
        # Clean numeric types for JSON/API compatibility
        return {k: (int(v) if k in self.CATEGORICAL_VARS else (float(v) if pd.notna(v) else np.nan)) 
                for k, v in attrs.items()}