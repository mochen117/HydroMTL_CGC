"""
Attribute loader for CAMELS basin attributes.
Modified to ensure required static attributes are loaded,
and to add one-hot encoded columns for categorical variables (without dropping originals).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import re

from .base_loader import BaseDataLoader
from hydro_data_processor.config.settings import DataSourceConfig
from hydro_data_processor.utils.huc_mapper import HucMapper

logger = logging.getLogger(__name__)


class AttributeLoader(BaseDataLoader):
    """Loader for CAMELS basin attributes and one-hot encoding."""

    ATTRIBUTE_FILES = {
        "name": "camels_name.txt",
        "vege": "camels_vege.txt",
        "soil": "camels_soil.txt",
        "geol": "camels_geol.txt",
        "topo": "camels_topo.txt",
    }
    
    # Static variable mapping with CAMELS column names
    STATIC_VARIABLE_MAPPING = {
        # Terrain attributes (3)
        'elev_mean': 'elev_mean',
        'slope_mean': 'slope_mean',
        'area_gages2': 'area_gages2',
        
        # Land cover attributes (5)
        'frac_forest': 'pct_forest',
        'lai_max': 'lai_max',
        'lai_diff': 'lai_diff',
        'dom_land_cover_frac': 'dom_land_cover_frac',
        'dom_land_cover': 'dom_land_cover',
        
        # Soil attributes (5)
        'root_depth_50': 'root_depth_50',
        'soil_depth_statgso': 'soil_depth_statsgo',
        'soil_porosity': 'soil_porosity',
        'soil_conductivity': 'soil_conductivity',
        'max_water_content': 'max_water_content',
        
        # Geology attributes (4)
        'geol_class_1st': 'geol_1st_class',
        'geol_class_2nd': 'geol_2nd_class',
        'geol_porosity': 'geol_porostiy',
        'geol_permeability': 'geol_permeability',
    }
    
    # Required static variables (17 in total)
    REQUIRED_STATIC_VARS = [
        # Terrain
        'elev_mean', 'slope_mean', 'area_gages2',
        # Land cover
        'frac_forest', 'lai_max', 'lai_diff',
        'dom_land_cover_frac', 'dom_land_cover',
        # Soil
        'root_depth_50', 'soil_depth_statgso', 'soil_porosity',
        'soil_conductivity', 'max_water_content',
        # Geology
        'geol_class_1st', 'geol_class_2nd',
        'geol_porosity', 'geol_permeability'
    ]
    
    # Categorical variables to one-hot encode (original columns are kept)
    CATEGORICAL_VARS = ['dom_land_cover', 'geol_class_1st', 'geol_class_2nd']

    # Whitelist of allowed static variables (paper Table 1)
    ALLOWED_STATIC = [
        'elev_mean', 'slope_mean', 'area_gages2',
        'frac_forest', 'lai_max', 'lai_diff', 'dom_land_cover_frac',
        'root_depth_50', 'soil_depth_statgso', 'soil_porosity',
        'soil_conductivity', 'max_water_content',
        'geol_porosity', 'geol_permeability'
    ]

    def __init__(self, config: DataSourceConfig):
        super().__init__(config, "attributes")

        self.huc_mapper = HucMapper(self.config.data_source_path.parent.parent)
        self._attributes_cache: Dict[str, pd.DataFrame] = {}
        self._debug_counter = 0
        self._max_debug_samples = 5

        logger.debug(f"AttributeLoader initialized for one-hot encoding")

    def load(self, max_basins: Optional[int] = None, selected_basins: Optional[List[str]] = None,
             skip_validation: bool = False, **kwargs) -> pd.DataFrame:
        """
        Load basin attributes from all CAMELS attribute files, then add one-hot encoded columns
        for categorical variables (original columns are kept).

        Args:
            max_basins: Maximum number of basins to load, None means all basins
            selected_basins: Optional list of gage_ids to process exclusively.
            skip_validation: If True, skip the basin ID validation step (huc_mapper.validate_basin).
                            Use this when you need to load all basins for custom filtering.

        Returns:
            Merged DataFrame with all required static attributes plus one-hot encoded columns.
        """
        logger.info("Loading CAMELS basin attributes")
        
        if max_basins is None:
            logger.info("max_basins is None - will load ALL available basins")
        else:
            logger.info(f"max_basins is {max_basins} - will load up to {max_basins} basins")

        # -----------------------------------------------------------------
        # Step 1: Load name file (full dataset)
        # -----------------------------------------------------------------
        name_df = self._load_attribute_file("name", self.ATTRIBUTE_FILES["name"])
        if name_df is None or name_df.empty:
            logger.error("Failed to load basin names")
            return pd.DataFrame()

        name_df = self._format_gage_ids_batch(name_df, "name dataframe")

        # -----------------------------------------------------------------
        # Step 2: Apply selected_basins filter (if provided)
        # -----------------------------------------------------------------
        if selected_basins:
            formatted_selected = [self._format_gage_id(gid) for gid in selected_basins]
            original_count = len(name_df)
            name_df = name_df[name_df['gage_id'].isin(formatted_selected)].copy()
            logger.info(f"Filtered name file to {len(name_df)} basins from selected_basins list (original: {original_count})")
            
            missing = set(formatted_selected) - set(name_df['gage_id'])
            if missing:
                logger.warning(f"Selected basins not found in camels_name.txt: {sorted(missing)}")
                if len(name_df) == 0:
                    logger.error("No basins remain after filtering by selected_basins")
                    return pd.DataFrame()

        merged_df = name_df

        # -----------------------------------------------------------------
        # Step 3: Merge other attribute files (only keep basins already in merged_df)
        # -----------------------------------------------------------------
        for attr_type, filename in self.ATTRIBUTE_FILES.items():
            if attr_type == "name":
                continue

            try:
                attr_df = self._load_attribute_file(attr_type, filename)
                if attr_df is not None and not attr_df.empty:
                    attr_df = self._format_gage_ids_batch(attr_df, f"{attr_type} attributes")
                    
                    # Only keep basins that are already in merged_df
                    attr_df = attr_df[attr_df['gage_id'].isin(merged_df['gage_id'])]
                    
                    overlap = set(attr_df['gage_id']).intersection(set(merged_df['gage_id']))
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Overlap between name and {attr_type}: {len(overlap)} IDs")
                    
                    merged_df = pd.merge(merged_df, attr_df, on="gage_id", how="left")
                    
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Merged {attr_type} attributes: {len(attr_df)} basins")
                        
                    if attr_type == "vege" and "root_depth_50" in attr_df.columns:
                        if "root_depth_50" in merged_df.columns and logger.isEnabledFor(logging.DEBUG):
                            non_na_rows = merged_df[merged_df["root_depth_50"].notna()]
                            if len(non_na_rows) > 0:
                                sample = non_na_rows[["gage_id", "root_depth_50"]].head(3).to_dict(orient="records")
                                logger.debug(f"root_depth_50 actual values (non-NA) after merging: {sample}")
            except Exception as e:
                logger.error(f"Failed to load {attr_type} from {filename}: {e}")

        # -----------------------------------------------------------------
        # Step 4: Validate basin IDs (skip if skip_validation=True)
        # -----------------------------------------------------------------
        if not skip_validation:
            valid_mask = merged_df['gage_id'].apply(self.huc_mapper.validate_basin)
            invalid_count = (~valid_mask).sum()
            if invalid_count > 0:
                invalid_ids = merged_df.loc[~valid_mask, 'gage_id'].tolist()
                logger.warning(f"Removing {invalid_count} invalid basin IDs: {invalid_ids}")
                merged_df = merged_df[valid_mask].copy()
            else:
                logger.info("All basins passed validation")
        else:
            logger.info("Skipping basin validation as requested (skip_validation=True)")

        # -----------------------------------------------------------------
        # Step 5: Ensure huc_02 column exists
        # -----------------------------------------------------------------
        if 'huc_02' not in merged_df.columns:
            logger.debug("huc_02 column not found in attributes, deriving from gage_id")
            merged_df['huc_02'] = merged_df['gage_id'].apply(
                lambda x: str(x)[:2] if len(str(x)) >= 2 else '01'
            )
        
        merged_df['huc_02'] = merged_df['huc_02'].astype(str).str.zfill(2)

        # -----------------------------------------------------------------
        # Step 6: Try to extract HUC mapping from camels_name.txt (optional)
        # -----------------------------------------------------------------
        huc_mapping = self._extract_huc_from_camels_name()
        if huc_mapping is not None and not huc_mapping.empty:
            for idx, row in merged_df.iterrows():
                gage_id = row['gage_id']
                if gage_id in huc_mapping.index:
                    merged_df.at[idx, 'huc_02'] = huc_mapping.loc[gage_id]

        # -----------------------------------------------------------------
        # Step 7: Standardize column names to match required static variables
        # -----------------------------------------------------------------
        merged_df = self._standardize_column_names(merged_df)
        
        # -----------------------------------------------------------------
        # Step 8: Add one-hot encoded columns for categorical variables
        # (original columns are kept)
        # -----------------------------------------------------------------
        merged_df = self._encode_categorical_variables(merged_df)
        
        # -----------------------------------------------------------------
        # Step 9: Keep only allowed static variables (paper Table 1) and one-hot columns
        # -----------------------------------------------------------------
        allowed_cols = set(self.ALLOWED_STATIC)
        # Add one-hot encoded columns
        for col in merged_df.columns:
            if (col.startswith('dom_land_cover_') or 
                col.startswith('geol_class_1st_') or 
                col.startswith('geol_class_2nd_')):
                allowed_cols.add(col)
        # Keep gage_id and huc_02 as well
        keep_cols = ['gage_id'] + [c for c in allowed_cols if c in merged_df.columns]
        if 'huc_02' in merged_df.columns:
            keep_cols.append('huc_02')
        merged_df = merged_df[keep_cols]
        
        # -----------------------------------------------------------------
        # Step 10: Ensure all required static variables exist (numeric ones only)
        # -----------------------------------------------------------------
        merged_df = self._ensure_required_variables(merged_df)
        
        # -----------------------------------------------------------------
        # Step 11: Validate variable completeness
        # -----------------------------------------------------------------
        validation = self._validate_variable_completeness(merged_df)
        self._log_completeness(validation)

        # -----------------------------------------------------------------
        # Step 12: Apply max_basins limit (only if specified)
        # -----------------------------------------------------------------
        if max_basins is not None and max_basins < len(merged_df):
            logger.info(f"Limiting to {max_basins} basins (as requested)")
            merged_df = merged_df.head(max_basins)
        else:
            logger.info(f"Loaded ALL {len(merged_df)} basins")

        # -----------------------------------------------------------------
        # Step 13: Store metadata and data
        # -----------------------------------------------------------------
        self.metadata = {
            "total_basins": len(merged_df),
            "required_variables_count": len(self.REQUIRED_STATIC_VARS),
            "variables_loaded": validation['present_count'],
            "variables_missing": validation['missing_count'],
            "attribute_files": list(self.ATTRIBUTE_FILES.keys()),
            "columns": list(merged_df.columns)
        }

        self.data = merged_df
        logger.info(f"Loaded attributes for {len(merged_df)} basins with {validation['present_count']}/{len(self.REQUIRED_STATIC_VARS)} required variables")
        
        if logger.isEnabledFor(logging.DEBUG):
            critical_vars = ['root_depth_50', 'frac_forest', 'soil_depth_statgso']
            for var in critical_vars:
                if var in merged_df.columns:
                    non_na_count = merged_df[var].notna().sum()
                    total_count = len(merged_df)
                    coverage = non_na_count / total_count if total_count > 0 else 0
                    
                    if non_na_count > 0:
                        sample_vals = merged_df.loc[merged_df[var].notna(), ['gage_id', var]].head(3).to_dict(orient='records')
                        logger.debug(f"Variable {var}: coverage={coverage:.1%}, sample values: {sample_vals}")
        
        if 'huc_02' in merged_df.columns and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"HUC2 values sample: {merged_df['huc_02'].head().tolist()}")
        
        return merged_df

    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add one-hot encoded columns for categorical variables and drop original columns.
        """
        if df.empty:
            return df

        df = df.copy()
        for var in self.CATEGORICAL_VARS:
            if var not in df.columns:
                logger.warning(f"Categorical variable {var} not found, skipping encoding")
                continue

            col = df[var].fillna('missing').astype(str)
            dummies = pd.get_dummies(col, prefix=var, dummy_na=False)

            def sanitize_name(name: str) -> str:
                name = name.replace(' ', '_')
                name = re.sub(r'[\(\)\[\]\{\}/\\]', '', name)
                if name and name[0].isdigit():
                    name = '_' + name
                return name

            for dummy_col in dummies.columns:
                safe_col = sanitize_name(dummy_col)
                df[safe_col] = dummies[dummy_col].astype(int)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Added one-hot encoded column: {safe_col}")

            # Drop the original categorical column (now encoded)
            df = df.drop(columns=[var])

            logger.info(f"One-hot encoded {var} into {len(dummies.columns)} binary features (original dropped)")

        return df

    # -------------------------------------------------------------------------
    # Helper methods (unchanged)
    # -------------------------------------------------------------------------
    def _format_gage_id(self, gage_id_str: str) -> str:
        try:
            clean_id = ''.join(filter(str.isdigit, str(gage_id_str)))
            if len(clean_id) == 7:
                return '0' + clean_id
            elif len(clean_id) == 8:
                return clean_id
            else:
                return clean_id.zfill(8)
        except Exception:
            return str(gage_id_str).zfill(8)

    def _format_gage_ids_batch(self, df: pd.DataFrame, context: str = "") -> pd.DataFrame:
        if 'gage_id' not in df.columns:
            return df
        
        original_lengths = df['gage_id'].astype(str).str.len().value_counts().to_dict()
        df = df.copy()
        df['gage_id'] = df['gage_id'].apply(self._format_gage_id)
        formatted_lengths = df['gage_id'].str.len().value_counts().to_dict()
        
        if logger.isEnabledFor(logging.DEBUG) and original_lengths != formatted_lengths:
            logger.debug(f"Gage ID formatting for {context}:")
            logger.debug(f"  Original length distribution: {original_lengths}")
            logger.debug(f"  Formatted length distribution: {formatted_lengths}")
            
            if self._debug_counter < self._max_debug_samples:
                sample_size = min(3, len(df))
                for i in range(sample_size):
                    original_id = df.index[i] if not df.empty else ""
                    gage_id = df['gage_id'].iloc[i] if i < len(df) else ""
                    logger.debug(f"  Sample: {original_id} -> {gage_id}")
                self._debug_counter += 1
        
        return df

    def _load_attribute_file(self, attr_type: str, filename: str) -> Optional[pd.DataFrame]:
        file_path = self.config.data_source_path / filename
        if not file_path.exists():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Attribute file not found: {file_path}")
            return None

        try:
            df = pd.read_csv(
                file_path, 
                delimiter=';',
                dtype={'gauge_id': str},
                na_values=['NA', 'NaN', 'nan', '', ' ', 'None', 'none', 'NULL', 'null'],
                keep_default_na=True,
                engine='python'
            )
            
            if 'gauge_id' in df.columns and 'gage_id' not in df.columns:
                df = df.rename(columns={'gauge_id': 'gage_id'})
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Renamed 'gauge_id' column to 'gage_id' in {filename}")

            if 'gage_id' not in df.columns:
                logger.error(f"No gage_id column in {filename}")
                return None

            df['gage_id'] = df['gage_id'].astype(str).str.strip()
            
            if attr_type == "vege" and 'root_depth_50' in df.columns:
                df['root_depth_50'] = pd.to_numeric(df['root_depth_50'], errors='coerce')
                if logger.isEnabledFor(logging.DEBUG):
                    non_na_count = df['root_depth_50'].notna().sum()
                    total_count = len(df)
                    coverage = non_na_count / total_count if total_count > 0 else 0
                    logger.debug(f"root_depth_50 in {filename}: dtype={df['root_depth_50'].dtype}, coverage={coverage:.1%}")

            return df

        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            if logger.isEnabledFor(logging.DEBUG):
                import traceback
                logger.debug(traceback.format_exc())
            return None

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'pct_forest' in df.columns and 'frac_forest' not in df.columns:
            df['frac_forest'] = df['pct_forest'] / 100.0
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Converted pct_forest to frac_forest (percentage to fraction)")
        
        if 'soil_depth_statsgo' in df.columns and 'soil_depth_statgso' not in df.columns:
            df['soil_depth_statgso'] = df['soil_depth_statsgo']
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Renamed soil_depth_statsgo to soil_depth_statgso")
        
        if 'geol_porostiy' in df.columns and 'geol_porosity' not in df.columns:
            df['geol_porosity'] = df['geol_porostiy']
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Renamed geol_porostiy to geol_porosity")
        
        for target_name, source_name in self.STATIC_VARIABLE_MAPPING.items():
            if source_name in df.columns and target_name not in df.columns:
                df[target_name] = df[source_name]
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Mapped {source_name} to {target_name}")
        
        if 'root_depth_50' in df.columns and logger.isEnabledFor(logging.DEBUG):
            non_na = df['root_depth_50'].notna().sum()
            total = len(df)
            logger.debug(f"root_depth_50 column found after standardization: {non_na}/{total} non-NA values")
        
        return df

    def _ensure_required_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required static variables (numeric) exist in DataFrame."""
        required_numeric = [v for v in self.REQUIRED_STATIC_VARS if v not in self.CATEGORICAL_VARS]
        
        for var in required_numeric:
            if var not in df.columns:
                logger.warning(f"Required numeric variable {var} not found, filling with NaN")
                df[var] = np.nan
        
        if 'root_depth_50' in df.columns and df['root_depth_50'].isna().all():
            logger.warning("root_depth_50 is NaN for all basins, attempting to estimate from soil data")
            if 'soil_depth_statgso' in df.columns and df['soil_depth_statgso'].notna().any():
                df['root_depth_50'] = df['soil_depth_statgso'] * 0.5
                logger.info("Estimated root_depth_50 as 50% of soil_depth_statgso")
            elif 'root_depth_99' in df.columns and df['root_depth_99'].notna().any():
                df['root_depth_50'] = df['root_depth_99'] * 0.5
                logger.info("Estimated root_depth_50 as 50% of root_depth_99")
        
        return df

    def _validate_variable_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate that all required static variables (numeric) are present."""
        required_numeric = [v for v in self.REQUIRED_STATIC_VARS if v not in self.CATEGORICAL_VARS]
        validation_result = {
            'total_required': len(required_numeric),
            'present': [],
            'missing': [],
            'coverage': {},
            'all_present': True
        }
        
        for var in required_numeric:
            if var in df.columns:
                validation_result['present'].append(var)
                non_nan = df[var].notna().sum()
                total = len(df)
                validation_result['coverage'][var] = non_nan / total if total > 0 else 0.0
                if non_nan == 0:
                    validation_result['missing'].append(f"{var} (all NaN)")
                    validation_result['all_present'] = False
            else:
                validation_result['missing'].append(var)
                validation_result['coverage'][var] = 0.0
                validation_result['all_present'] = False
        
        validation_result['present_count'] = len(validation_result['present'])
        validation_result['missing_count'] = len(validation_result['missing'])
        return validation_result

    def _log_completeness(self, validation: Dict[str, Any]) -> None:
        """Log variable completeness validation results."""
        if validation['all_present']:
            logger.info(f"Static variables completeness: PASS - All {validation['total_required']} numeric variables present")
        else:
            logger.warning(f"Static variables completeness: FAIL - {validation['missing_count']}/{validation['total_required']} variables missing: {validation['missing']}")
        
        low_coverage = {var: cov for var, cov in validation['coverage'].items() if cov < 0.8}
        if low_coverage:
            logger.warning(f"Static variables with low coverage (<80%): {low_coverage}")

    def _extract_huc_from_camels_name(self) -> Optional[pd.Series]:
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
                    
                    if 'gauge_id' in name_df.columns and 'gage_id' not in name_df.columns:
                        name_df = name_df.rename(columns={'gauge_id': 'gage_id'})
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug("Renamed 'gauge_id' column to 'gage_id' in camels_name.txt")
                    
                    if 'huc_02' not in name_df.columns:
                        logger.debug("camels_name.txt does not contain huc_02 column")
                        return None
                    
                    name_df = self._format_gage_ids_batch(name_df, "camels_name.txt")
                    huc_series = pd.Series(name_df['huc_02'].values, index=name_df['gage_id'])
                    huc_series = huc_series.astype(str).str.zfill(2)
                    logger.info(f"Extracted HUC2 mapping from camels_name.txt: {len(huc_series)} records")
                    return huc_series
                    
                except Exception as e:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Failed to extract HUC from camels_name.txt: {e}")
        
        logger.debug("camels_name.txt not found in any expected location")
        return None

    def get_basin_attributes(self, gage_id: str) -> Dict[str, Any]:
        """Get all attributes for a specific gage."""
        if not self.data or self.data.empty:
            logger.warning("Attributes not loaded yet")
            return {}

        formatted_gage_id = self._format_gage_id(gage_id)
        gage_row = self.data[self.data['gage_id'] == formatted_gage_id]
        if gage_row.empty:
            logger.warning(f"No attributes found for gage {formatted_gage_id} (original: {gage_id})")
            return {}

        attrs = gage_row.iloc[0].to_dict()
        clean_attrs = {}
        
        required_numeric = [v for v in self.REQUIRED_STATIC_VARS if v not in self.CATEGORICAL_VARS]
        for var in required_numeric:
            if var in attrs:
                value = attrs[var]
                if isinstance(value, (np.integer, np.int64)):
                    clean_attrs[var] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    clean_attrs[var] = float(value)
                elif isinstance(value, np.ndarray):
                    clean_attrs[var] = value.tolist()
                elif pd.isna(value):
                    clean_attrs[var] = np.nan
                else:
                    clean_attrs[var] = value
            else:
                logger.warning(f"Required variable {var} missing for gage {formatted_gage_id}")
                clean_attrs[var] = np.nan
        
        # Also include categorical original columns and one-hot encoded columns
        for col in self.data.columns:
            if col not in clean_attrs and col != 'gage_id' and col not in required_numeric:
                clean_attrs[col] = attrs.get(col, np.nan)
        
        return clean_attrs

    def get_variables_summary(self) -> Dict[str, Any]:
        """Get summary of required variable coverage (numeric only)."""
        if self.data is None or self.data.empty:
            return {}
        
        required_numeric = [v for v in self.REQUIRED_STATIC_VARS if v not in self.CATEGORICAL_VARS]
        summary = {}
        for var in required_numeric:
            if var in self.data.columns:
                coverage = self.data[var].notna().sum() / len(self.data)
                summary[var] = {
                    'present': True,
                    'coverage': float(coverage),
                    'min': float(self.data[var].min()) if self.data[var].notna().any() else None,
                    'max': float(self.data[var].max()) if self.data[var].notna().any() else None,
                    'mean': float(self.data[var].mean()) if self.data[var].notna().any() else None
                }
            else:
                summary[var] = {
                    'present': False,
                    'coverage': 0.0,
                    'min': None,
                    'max': None,
                    'mean': None
                }
        
        return summary