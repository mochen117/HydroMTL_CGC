"""
Attribute loader for CAMELS basin attributes.
Categorical variables are label-encoded (missing -> 0, valid -> 1, 2, ...).
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
    """Loader for CAMELS basin attributes with detailed annotations."""

    ATTRIBUTE_FILES = {
        "name": "camels_name.txt",
        "vege": "camels_vege.txt",
        "soil": "camels_soil.txt",
        "geol": "camels_geol.txt",
        "topo": "camels_topo.txt",
        "clim": "camels_clim.txt",
    }
    
    STATIC_VARIABLE_MAPPING = {
        'elev_mean': 'elev_mean',
        'slope_mean': 'slope_mean',
        'area_gages2': 'area_gages2',
        
        'frac_forest': 'pct_forest',
        'lai_max': 'lai_max',
        'lai_diff': 'lai_diff',
        'gvf_max': 'gvf_max',
        'dom_land_cover': 'dom_land_cover',
        'dom_land_cover_frac': 'dom_land_cover_frac',
        'root_depth_50': 'root_depth_50',
        
        'soil_depth_statgso': 'soil_depth_statsgo',
        'soil_porosity': 'soil_porosity',
        'soil_conductivity': 'soil_conductivity',
        'max_water_content': 'max_water_content',
        'sand_frac': 'sand_frac',
        'clay_frac': 'clay_frac',
        'organic_frac': 'organic_frac',
        
        'geol_class_1st': 'geol_1st_class',
        'geol_class_2nd': 'geol_2nd_class',
        'geol_porosity': 'geol_porostiy',
        'geol_permeability': 'geol_permeability',
        'carbonate_rocks_frac': 'carbonate_rocks_frac',
        
        'p_mean': 'p_mean',
        'pet_mean': 'pet_mean',
        'aridity': 'aridity',
        'p_seasonality': 'p_seasonality',
        'frac_snow': 'frac_snow',
    }
    
    REQUIRED_STATIC_VARS = list(STATIC_VARIABLE_MAPPING.keys())
    CATEGORICAL_VARS = ['dom_land_cover', 'geol_class_1st', 'geol_class_2nd']
    HUC2_COL = 'huc_02' 

    def __init__(self, config: DataSourceConfig):
        super().__init__(config, "attributes")
        self.huc_mapper = HucMapper(self.config.data_source_path.parent.parent)
        self.cat_mappings = {}
        logger.info("AttributeLoader initialized.")

    def load(self, max_basins: Optional[int] = None, selected_basins: Optional[List[str]] = None,
             skip_validation: bool = False, **kwargs) -> pd.DataFrame:
        
        logger.info("Merging CAMELS attribute files...")
        merged_df: Optional[pd.DataFrame] = None

        for attr_type, filename in self.ATTRIBUTE_FILES.items():
            df = self._load_attribute_file(attr_type, filename)
            if df is None or df.empty:
                continue
            
            df = self._format_gage_ids_batch(df, f"{attr_type} attributes")
            
            if merged_df is None:
                merged_df = df
            else:
                cols_to_use = df.columns.difference(merged_df.columns).tolist() + ['gage_id']
                merged_df = pd.merge(merged_df, df[cols_to_use], on="gage_id", how="outer")

        if merged_df is None or merged_df.empty:
            logger.error("Critical Failure: No attribute files were loaded.")
            return pd.DataFrame()

        if selected_basins:
            formatted_selected = [self._format_gage_id(gid) for gid in selected_basins]
            merged_df = merged_df[merged_df['gage_id'].isin(formatted_selected)].copy()

        if not skip_validation:
            valid_mask = merged_df['gage_id'].apply(self.huc_mapper.validate_basin)
            merged_df = merged_df[valid_mask].copy()

        merged_df = self._standardize_column_names(merged_df)
        merged_df = self._ensure_required_variables(merged_df)

        if self.HUC2_COL not in merged_df.columns:
            merged_df[self.HUC2_COL] = merged_df['gage_id'].str[:2].str.zfill(2)

        # Ensure missing is 0, and valid categories start from 1
        for var in self.CATEGORICAL_VARS:
            series = merged_df[var].fillna('missing').astype(str)
            unique_vals = sorted(series.unique())
            
            mapping = {val: (i + 1) for i, val in enumerate(unique_vals) if val != 'missing'}
            mapping['missing'] = 0
            
            merged_df[var] = series.map(mapping).astype(int)
            self.cat_mappings[var] = mapping

        final_cols = ['gage_id', self.HUC2_COL] + self.REQUIRED_STATIC_VARS
        merged_df = merged_df[final_cols]

        if max_basins:
            merged_df = merged_df.head(max_basins)

        self.data = merged_df
        logger.info(f"Loaded {len(merged_df)} basins with attributes.")
        return merged_df

    def _load_attribute_file(self, attr_type: str, filename: str) -> Optional[pd.DataFrame]:
        file_path = self.config.data_source_path / filename
        if not file_path.exists():
            return None
        try:
            df = pd.read_csv(
                file_path, delimiter=';', dtype={'gauge_id': str, 'gage_id': str},
                na_values=['NA', 'NaN', 'nan', 'None', '-999', ''],
                keep_default_na=True, engine='python'
            )
            if 'gauge_id' in df.columns:
                df = df.rename(columns={'gauge_id': 'gage_id'})
            return df
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")
            return None

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'pct_forest' in df.columns:
            df['frac_forest'] = df['pct_forest'] / 100.0
            
        for target, source in self.STATIC_VARIABLE_MAPPING.items():
            if source in df.columns and target not in df.columns:
                df[target] = df[source]
        return df

    def _ensure_required_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        for var in self.REQUIRED_STATIC_VARS:
            if var not in df.columns:
                df[var] = np.nan
        return df

    def _format_gage_id(self, gage_id_str: str) -> str:
        clean_id = ''.join(filter(str.isdigit, str(gage_id_str)))
        return clean_id.zfill(8)

    def _format_gage_ids_batch(self, df: pd.DataFrame, context: str = "") -> pd.DataFrame:
        if 'gage_id' not in df.columns:
            return df
        df = df.copy()
        df['gage_id'] = df['gage_id'].apply(self._format_gage_id)
        return df

    def get_basin_attributes(self, gage_id: str) -> Dict[str, Any]:
        if self.data is None or self.data.empty:
            return {}
        
        row = self.data[self.data['gage_id'] == self._format_gage_id(gage_id)]
        if row.empty:
            return {}
            
        attrs = row.iloc[0].to_dict()
        return {k: (int(v) if k in self.CATEGORICAL_VARS else (float(v) if pd.notna(v) else np.nan)) 
                for k, v in attrs.items()}