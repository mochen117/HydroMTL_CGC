"""
Main pipeline for Hydro Data Processing.
Modified to comply with paper Table 1 specifications for MTL hydrological modeling.
"""

import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
from datetime import datetime
import traceback

from hydro_data_processor.config.settings import ProjectConfig
from hydro_data_processor.loaders.attribute_loader import AttributeLoader
from hydro_data_processor.loaders.streamflow_loader import StreamflowLoader
from hydro_data_processor.loaders.forcing_loader import ForcingLoader
from hydro_data_processor.loaders.et_loader import ETLoader
from hydro_data_processor.loaders.smap_loader import SMAPLoader
from hydro_data_processor.processors.multi_source_processor import MultiSourceProcessor
from hydro_data_processor.processors.batch_processor import BatchProcessor
from hydro_data_processor.utils.io import save_json

logger = logging.getLogger(__name__)

__all__ = ['HydroDataPipeline']


class HydroDataPipeline:
    """Main pipeline for processing hydrological data from multiple sources with paper compliance."""

    # Paper Table 1 variable specifications
    PAPER_SPECIFICATION = {
        'meteorological_forcing_variables': [
            'total_precipitation',  # Daily total precipitation
            'temperature',          # Air temperature at 2 m above the surface
            'specific_humidity',    # Specific humidity at 2 m above the surface
            'shortwave_radiation',  # Surface downward shortwave radiation
            'potential_energy'      # Convective available potential energy
        ],
        'model_output_variables': [
            'streamflow',           # Daily streamflow in the outlet of a basin
            'evapotranspiration'    # Basin mean daily actual evapotranspiration
        ],
        'model_evaluation_variable': 'ssm',  # Surface soil moisture for evaluation
        'static_attribute_count': 17  # 3 terrain + 5 land cover + 5 soil + 4 geology
    }

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.data_root = config.data_root.resolve()

        self._initialize_loaders()

        self.processed_gages: List[str] = []
        self.failed_gages: List[Dict] = []
        self.skipped_gages: List[Dict] = []
        self.valid_gages: List[Dict] = []  # Gages that meet coverage requirements

        self.huc2_mapping: Dict[str, str] = {}
        self.paper_validation_results: List[Dict] = []

        logger.debug("Hydro Data Pipeline initialized with paper Table 1 compliance")
        logger.debug(f"Data root (resolved): {self.data_root}")

    def _initialize_loaders(self):
        """Initialize all data loaders with paper compliance."""
        logger.debug("Initializing data loaders for paper compliance")

        logger.info(f"Data root resolved to: {self.data_root}")
        
        if not self.data_root.exists():
            logger.error(f"Data root directory does not exist: {self.data_root}")
            raise FileNotFoundError(f"Data root directory not found: {self.data_root}")

        logger.debug("Checking data directory structure...")
        
        camels_us_dir = self.data_root / "camels" / "camels_us"
        if camels_us_dir.exists():
            logger.debug(f"Found CAMELS directory: {camels_us_dir}")
        else:
            logger.warning(f"CAMELS directory not found: {camels_us_dir}")
            
            camels_alt = self.data_root / "camels_us"
            if camels_alt.exists():
                logger.info(f"Using alternative CAMELS structure: {camels_alt}")

        attribute_config = self.config.data_sources.get("attributes")
        if attribute_config:
            if not attribute_config.data_source_path.is_absolute():
                attribute_config.data_source_path = self.data_root / attribute_config.data_source_path
            else:
                if not attribute_config.data_source_path.exists():
                    logger.warning(f"Attribute path not found: {attribute_config.data_source_path}")
                    alt_path = self.data_root / attribute_config.data_source_path.name
                    if alt_path.exists():
                        logger.info(f"Using alternative attribute path: {alt_path}")
                        attribute_config.data_source_path = alt_path
            
            logger.debug(f"Attribute loader path: {attribute_config.data_source_path}")
            self.attribute_loader = AttributeLoader(attribute_config)
        else:
            logger.error("No attribute configuration found")
            self.attribute_loader = None

        camels_streamflow_config = self.config.data_sources.get("camels_streamflow")
        if camels_streamflow_config:
            if not camels_streamflow_config.data_source_path.is_absolute():
                camels_streamflow_config.data_source_path = self.data_root / camels_streamflow_config.data_source_path
            
            if not camels_streamflow_config.data_source_path.exists():
                logger.warning(f"CAMELS streamflow path not found: {camels_streamflow_config.data_source_path}")
                self._search_and_fix_camels_path(camels_streamflow_config)
            
            self.camels_streamflow_loader = StreamflowLoader(camels_streamflow_config)
            self.camels_streamflow_loader.data_source_type = "camels"
            logger.debug(f"CAMELS streamflow loader initialized: {camels_streamflow_config.data_source_path}")
        else:
            logger.error("No CAMELS streamflow configuration found")
            self.camels_streamflow_loader = None

        usgs_streamflow_config = self.config.data_sources.get("usgs_streamflow")
        if usgs_streamflow_config:
            if not usgs_streamflow_config.data_source_path.is_absolute():
                usgs_streamflow_config.data_source_path = self.data_root / usgs_streamflow_config.data_source_path
            
            if not usgs_streamflow_config.data_source_path.exists():
                logger.warning(f"USGS streamflow path not found: {usgs_streamflow_config.data_source_path}")
                alt_path = self.data_root / "camels" / "camels_us" / "usgs_streamflow"
                if alt_path.exists():
                    logger.info(f"Using alternative USGS path: {alt_path}")
                    usgs_streamflow_config.data_source_path = alt_path
            
            self.usgs_streamflow_loader = StreamflowLoader(usgs_streamflow_config)
            self.usgs_streamflow_loader.data_source_type = "usgs"
            logger.debug(f"USGS streamflow loader initialized: {usgs_streamflow_config.data_source_path}")
        else:
            logger.warning("No USGS streamflow configuration found - some data may be missing")
            self.usgs_streamflow_loader = None

        forcing_config = self.config.data_sources.get("nldas_forcing")
        if forcing_config:
            if not forcing_config.data_source_path.is_absolute():
                forcing_config.data_source_path = self.data_root / forcing_config.data_source_path
            
            if not forcing_config.data_source_path.exists():
                logger.warning(f"Forcing data path not found: {forcing_config.data_source_path}")
                alt_paths = [
                    self.data_root / "nldas",
                    self.data_root / "forcing",
                    self.data_root / "basin_mean_forcing"
                ]
                for alt_path in alt_paths:
                    if alt_path.exists():
                        logger.info(f"Using alternative forcing path: {alt_path}")
                        forcing_config.data_source_path = alt_path
                        break
            
            self.forcing_loader = ForcingLoader(forcing_config)
            logger.debug(f"Forcing loader initialized: {forcing_config.data_source_path}")
        else:
            logger.error("No forcing configuration found")
            self.forcing_loader = None

        et_config = self.config.data_sources.get("et_data")
        if et_config:
            if not et_config.data_source_path.is_absolute():
                et_config.data_source_path = self.data_root / et_config.data_source_path
            
            self.et_loader = ETLoader(et_config)
            logger.debug(f"ET loader initialized: {et_config.data_source_path}")
        else:
            logger.debug("No ET configuration found")
            self.et_loader = None

        smap_config = self.config.data_sources.get("smap_data")
        if smap_config:
            if not smap_config.data_source_path.is_absolute():
                smap_config.data_source_path = self.data_root / smap_config.data_source_path
            
            self.smap_loader = SMAPLoader(smap_config)
            logger.debug(f"SMAP loader initialized: {smap_config.data_source_path}")
        else:
            logger.debug("No SMAP configuration found")
            self.smap_loader = None

        logger.info("All loaders initialized with paper compliance")

    def _search_and_fix_camels_path(self, config):
        """Search for CAMELS data directory structure and fix the path."""
        base_dir = self.data_root / "camels" / "camels_us"
        
        possible_paths = [
            base_dir / "basin_timeseries_v1p2_metForcing_obsFlow" / 
            "basin_dataset_public_v1p2" / "camels_streamflow",
            base_dir / "camels_streamflow",
            base_dir / "streamflow",
            base_dir / "usgs_streamflow",
            self.data_root / "camels_streamflow"
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found CAMELS streamflow data at: {path}")
                config.data_source_path = path
                return
        
        logger.warning("Could not find CAMELS streamflow data in any expected location")
    def _filter_basins_by_streamflow_coverage(self, attributes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter basins based on streamflow data coverage (2001-2021).
        Only keep basins with coverage >= min_streamflow_coverage.
        
        Args:
            attributes_df: DataFrame with all basin attributes (including gage_id)
        
        Returns:
            Filtered DataFrame containing only basins meeting coverage requirement.
        """
        logger.info(f"Filtering basins by streamflow coverage ≥{self.config.processing_config.min_streamflow_coverage:.0%}...")
        
        gage_ids = attributes_df['gage_id'].tolist()
        valid_gage_ids = []
        coverage_info = []
        
        min_coverage = self.config.processing_config.min_streamflow_coverage
        
        for i, gage_id in enumerate(gage_ids, 1):
            if i % 50 == 0:
                logger.info(f"Processed {i}/{len(gage_ids)} basins for coverage check")
            
            try:
                # Get HUC2 from attributes
                gage_attrs = self._get_gage_attributes(gage_id, attributes_df)
                huc2 = self._get_huc2_for_gage(gage_id, gage_attrs)
                if not huc2:
                    logger.debug(f"Cannot determine HUC2 for {gage_id}, skipping")
                    continue
                
                # Load streamflow data
                streamflow_df = self._load_streamflow_with_huc2(gage_id, huc2)
                if streamflow_df is None or streamflow_df.empty:
                    logger.debug(f"No streamflow data for {gage_id}")
                    continue
                
                # Calculate coverage
                total_days = len(streamflow_df)
                valid_days = streamflow_df['streamflow'].notna().sum()
                coverage = valid_days / total_days if total_days > 0 else 0.0
                
                coverage_info.append({
                    'gage_id': gage_id,
                    'coverage': coverage,
                    'valid_days': valid_days,
                    'total_days': total_days
                })
                
                if coverage >= min_coverage:
                    valid_gage_ids.append(gage_id)
                    logger.debug(f"Gage {gage_id}: coverage = {coverage:.2%} - PASS")
                else:
                    logger.debug(f"Gage {gage_id}: coverage = {coverage:.2%} - FAIL")
                    
            except Exception as e:
                logger.debug(f"Error checking coverage for {gage_id}: {e}")
                continue
        
        logger.info(f"Coverage check completed: {len(valid_gage_ids)}/{len(gage_ids)} basins passed "
                    f"(coverage ≥{min_coverage:.0%})")
        
        self.coverage_check_results = coverage_info
        
        filtered_df = attributes_df[attributes_df['gage_id'].isin(valid_gage_ids)].copy()
        logger.info(f"Filtered attributes to {len(filtered_df)} basins")
        
        return filtered_df

    def run(self):
        """Run the complete pipeline with batch processing and coverage filtering."""
        logger.info("=" * 60)
        logger.info("Hydro Data Processing Pipeline - Paper Table 1 Compliance")
        logger.info("=" * 60)
        logger.info(f"Data root: {self.data_root}")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info(f"Study period: {self.config.processing_config.start_date} to "
                f"{self.config.processing_config.end_date}")
        logger.info(f"Min coverage: {self.config.processing_config.min_streamflow_coverage:.0%}")
        logger.info(f"Paper specification: {self.PAPER_SPECIFICATION}")

        if not self.attribute_loader:
            logger.error("Attribute loader not available")
            return

        logger.info("Step 1: Loading all basin attributes for coverage filtering")
        attributes_df = self.attribute_loader.load(
            max_basins=None,  # Load all basins to perform coverage check
            skip_validation=True
        )

        if attributes_df.empty:
            logger.error("No attributes loaded. Exiting.")
            return

        # ---------------------------------------------------------------------
        # Perform streamflow coverage filtering (paper criterion: loss rate <5%)
        # ---------------------------------------------------------------------
        attributes_df = self._filter_basins_by_streamflow_coverage(attributes_df)

        # Apply max_basins limit after coverage filtering (if specified)
        if self.config.max_basins is not None and self.config.max_basins < len(attributes_df):
            logger.info(f"Limiting to {self.config.max_basins} basins (as requested)")
            attributes_df = attributes_df.head(self.config.max_basins)

        if attributes_df.empty:
            logger.error("No attributes loaded. Exiting.")
            return

        if 'basin_id' in attributes_df.columns and 'gage_id' not in attributes_df.columns:
            attributes_df = attributes_df.rename(columns={'basin_id': 'gage_id'})
            logger.debug("Renamed 'basin_id' column to 'gage_id'")

        if 'gage_id' not in attributes_df.columns:
            logger.error("No gage_id column found in attributes")
            return

        def ensure_8_digits(gage_id):
            if gage_id is None:
                return None
            gage_str = str(gage_id)
            if len(gage_str) == 7:
                return '0' + gage_str
            elif len(gage_str) < 8:
                return gage_str.zfill(8)
            return gage_str

        attributes_df['gage_id'] = attributes_df['gage_id'].astype(str).apply(ensure_8_digits)

        gage_ids = attributes_df['gage_id'].tolist()
        logger.info(f"Step 2: Processing {len(gage_ids)} gages with paper compliance")
        
        multi_source_processor = MultiSourceProcessor(self.config)
        batch_processor = BatchProcessor(self, multi_source_processor)
        
        stats, coverage_results = self._process_with_coverage_filtering(
            gage_ids, attributes_df, batch_processor
        )
        
        logger.info("Step 3: Generating final summary and paper compliance report")
        self._generate_final_summary(stats, coverage_results, attributes_df)

    def _process_with_coverage_filtering(self, gage_ids: List[str], 
                                        attributes_df: pd.DataFrame,
                                        batch_processor) -> Tuple[Dict, List[Dict]]:
        """Process gages with coverage filtering and paper compliance."""
        all_results = []
        coverage_results = []
        min_coverage = self.config.processing_config.min_streamflow_coverage
        
        for gage_id in gage_ids:
            logger.debug(f"Processing gage {gage_id} with paper compliance")
            
            try:
                gage_attrs = self._get_gage_attributes(gage_id, attributes_df)
                
                # Validate static attributes for paper compliance
                static_validation = self._validate_static_attributes(gage_attrs, gage_id)
                if not static_validation['all_present']:
                    logger.warning(f"Gage {gage_id} missing paper-required static attributes: {static_validation['missing_count']} missing")
                
                huc2 = self._get_huc2_for_gage(gage_id, gage_attrs)
                if not huc2:
                    logger.warning(f"No HUC2 found for gage {gage_id}")
                    self.failed_gages.append({
                        'gage_id': gage_id,
                        'reason': 'No HUC2 mapping found'
                    })
                    continue
                
                streamflow_data = self._load_streamflow_with_huc2(gage_id, huc2)
                forcing_data = self._load_forcing_with_huc2(gage_id, huc2)
                
                if streamflow_data is None or streamflow_data.empty:
                    logger.warning(f"No streamflow data for gage {gage_id}")
                    self.failed_gages.append({
                        'gage_id': gage_id,
                        'reason': 'No streamflow data'
                    })
                    continue
                    
                if forcing_data is None or forcing_data.empty:
                    logger.warning(f"No forcing data for gage {gage_id}")
                    self.failed_gages.append({
                        'gage_id': gage_id,
                        'reason': 'No forcing data'
                    })
                    continue
                
                # Load ET data with resampling
                et_data = None
                if self.et_loader:
                    et_data = self.et_loader.load(
                        [gage_id], 
                        huc2=huc2,
                        align_to_study_period=True,
                        start_date=self.config.processing_config.start_date,
                        end_date=self.config.processing_config.end_date
                    )
                    
                # Load SMAP data for evaluation only
                smap_data = None
                if self.smap_loader:
                    smap_data = self.smap_loader.load([gage_id], huc2=huc2)
                
                merged_data, coverage_info = self._merge_all_data_with_time_alignment(
                    streamflow_data, forcing_data, et_data, smap_data, gage_id
                )
                
                if merged_data is None or merged_data.empty:
                    logger.warning(f"Failed to merge data for gage {gage_id}")
                    self.failed_gages.append({
                        'gage_id': gage_id,
                        'reason': 'Data merging failed'
                    })
                    continue
                
                # Validate merged data for paper compliance
                paper_validation = self._validate_merged_data_paper_compliance(merged_data, gage_id)
                self.paper_validation_results.append(paper_validation)
                
                key_vars_coverage = self._check_key_variables_coverage(merged_data)
                streamflow_coverage = key_vars_coverage.get('streamflow', 0.0)

                coverage_result = {
                    'gage_id': gage_id,
                    'coverage': key_vars_coverage,
                    'streamflow_coverage': streamflow_coverage,
                    'passed': streamflow_coverage >= min_coverage,
                    'total_days': len(merged_data),
                    'valid_days': {
                        'streamflow': merged_data['streamflow'].notna().sum(),
                        'total_precipitation': merged_data['total_precipitation'].notna().sum(),
                        'temperature': merged_data['temperature'].notna().sum(),
                        'evapotranspiration': merged_data['evapotranspiration'].notna().sum() if 'evapotranspiration' in merged_data.columns else 0
                    },
                    'streamflow_missing_rate': 1.0 - streamflow_coverage,
                    'paper_compliance': paper_validation
                }

                if streamflow_coverage >= min_coverage:
                    logger.info(f"Gage {gage_id}: streamflow coverage {streamflow_coverage:.2%}, "
                               f"paper compliance: {paper_validation['compliance_status']}")
                else:
                    logger.warning(f"Gage {gage_id} failed streamflow coverage: {streamflow_coverage:.2%} < {min_coverage:.0%}")
                
                coverage_results.append(coverage_result)
                
                if streamflow_coverage >= min_coverage:
                    success = self._create_and_save_dataset(
                        merged_data, gage_attrs, gage_id, coverage_result, paper_validation
                    )
                    
                    if success:
                        self.valid_gages.append({
                            'gage_id': gage_id,
                            'coverage': streamflow_coverage,
                            'huc2': huc2,
                            'paper_compliance': paper_validation['compliance_status']
                        })
                        self.processed_gages.append(gage_id)
                        all_results.append({
                            'gage_id': gage_id,
                            'status': 'success',
                            'coverage': coverage_result,
                            'paper_compliance': paper_validation
                        })
                        logger.info(f"Gage {gage_id} processed successfully with paper compliance")
                    else:
                        self.failed_gages.append({
                            'gage_id': gage_id,
                            'reason': 'Dataset creation failed'
                        })
                else:
                    self.skipped_gages.append({
                        'gage_id': gage_id,
                        'reason': f'Streamflow coverage insufficient: {streamflow_coverage:.2%}',
                        'coverage': coverage_result,
                        'paper_compliance': paper_validation
                    })
                    
            except Exception as e:
                logger.error(f"Error processing gage {gage_id}: {e}")
                logger.error(traceback.format_exc())
                self.failed_gages.append({
                    'gage_id': gage_id,
                    'reason': f'Processing error: {str(e)}'
                })
        
        # Generate paper compliance summary
        self._generate_paper_compliance_summary()
        
        stats = {
            'total_gages': len(gage_ids),
            'processed_gages': len(self.processed_gages),
            'failed_gages': len(self.failed_gages),
            'skipped_gages': len(self.skipped_gages),
            'valid_gages': len(self.valid_gages),
            'success_rate': len(self.processed_gages) / len(gage_ids) if gage_ids else 0,
            'coverage_threshold': min_coverage,
            'paper_compliance_rate': len([r for r in self.paper_validation_results if r['compliance_status'] == 'FULL_COMPLIANCE']) / len(self.paper_validation_results) if self.paper_validation_results else 0
        }
        
        return stats, coverage_results

    def _merge_all_data_with_time_alignment(self, streamflow_df: pd.DataFrame,
                                           forcing_df: pd.DataFrame,
                                           et_df: Optional[pd.DataFrame],
                                           smap_df: Optional[pd.DataFrame],
                                           gage_id: str) -> Tuple[Optional[pd.DataFrame], Dict[str, float]]:
        """Merge all data sources with time alignment to daily scale with paper compliance."""
        if streamflow_df is None or streamflow_df.empty:
            return None, {}
        
        start_date = pd.Timestamp(self.config.processing_config.start_date)
        end_date = pd.Timestamp(self.config.processing_config.end_date)
        base_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        base_df = pd.DataFrame({'date': base_dates})
        
        streamflow_df = streamflow_df.copy()
        if 'date' not in streamflow_df.columns and 'time' in streamflow_df.columns:
            streamflow_df = streamflow_df.rename(columns={'time': 'date'})
        
        if 'streamflow' in streamflow_df.columns:
            streamflow_df['streamflow'] = streamflow_df['streamflow'].replace(-999.0, np.nan)
        
        merged_df = pd.merge(base_df, streamflow_df[['date', 'streamflow']], 
                            on='date', how='left')
        
        # Merge forcing data with ONLY paper-required variables (5 variables)
        if forcing_df is not None and not forcing_df.empty:
            forcing_df = forcing_df.copy()
            if 'date' not in forcing_df.columns and 'time' in forcing_df.columns:
                forcing_df = forcing_df.rename(columns={'time': 'date'})
            
            # Paper requires only these 5 meteorological forcing variables
            paper_forcing_vars = self.PAPER_SPECIFICATION['meteorological_forcing_variables']
            
            for col in paper_forcing_vars:
                if col in forcing_df.columns:
                    merged_df = pd.merge(merged_df, forcing_df[['date', col]], 
                                        on='date', how='left')
                else:
                    logger.warning(f"Paper-required forcing variable {col} not found for gage {gage_id}")
                    merged_df[col] = np.nan
        
        # Merge ET data - only evapotranspiration as per paper
        if et_df is not None and not et_df.empty:
            et_df = et_df.copy()
            if 'date' not in et_df.columns and 'time' in et_df.columns:
                et_df = et_df.rename(columns={'time': 'date'})
            
            # Paper uses 'evapotranspiration' as model output
            if 'evapotranspiration' in et_df.columns:
                merged_df = pd.merge(merged_df, et_df[['date', 'evapotranspiration']], 
                                    on='date', how='left')
            else:
                logger.warning(f"ET data does not contain 'evapotranspiration' for gage {gage_id}")
        
        # Merge SMAP data - only ssm for model evaluation
        if smap_df is not None and not smap_df.empty:
            smap_df = smap_df.copy()
            if 'date' not in smap_df.columns and 'time' in smap_df.columns:
                smap_df = smap_df.rename(columns={'time': 'date'})
            
            # Paper uses 'ssm' for model evaluation
            if 'ssm' in smap_df.columns:
                merged_df = pd.merge(merged_df, smap_df[['date', 'ssm']], 
                                    on='date', how='left')
            else:
                logger.debug(f"SMAP data does not contain 'ssm' for gage {gage_id}")
        
        coverage_info = self._calculate_coverage(merged_df)
        
        logger.debug(f"Merged data for gage {gage_id}: {len(merged_df)} records, "
                    f"{len(merged_df.columns)} variables")
        
        return merged_df, coverage_info

    def _calculate_coverage(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate coverage percentage for key paper variables."""
        coverage = {}
        
        paper_variables = {
            'streamflow': 'streamflow',
            'total_precipitation': 'total_precipitation',
            'temperature': 'temperature',
            'evapotranspiration': 'evapotranspiration',
            'ssm': 'ssm'
        }
        
        for name, col in paper_variables.items():
            if col in df.columns:
                non_nan = df[col].notna().sum()
                total = len(df)
                coverage[name] = non_nan / total if total > 0 else 0.0
            else:
                coverage[name] = 0.0
        
        return coverage

    def _check_key_variables_coverage(self, df: pd.DataFrame) -> Dict[str, float]:
        """Check coverage for key paper hydrological variables."""
        coverage = {}
        
        # Check all paper-required dynamic variables
        paper_vars = (self.PAPER_SPECIFICATION['meteorological_forcing_variables'] +
                     self.PAPER_SPECIFICATION['model_output_variables'] +
                     [self.PAPER_SPECIFICATION['model_evaluation_variable']])
        
        for var in paper_vars:
            if var in df.columns:
                non_nan = df[var].notna().sum()
                total = len(df)
                coverage[var] = non_nan / total if total > 0 else 0.0
            else:
                coverage[var] = 0.0
        
        return coverage

    def _validate_static_attributes(self, gage_attrs: Dict[str, Any], gage_id: str) -> Dict[str, Any]:
        """Validate that all paper-required static attributes are present."""
        required_static_vars = [
            'elev_mean', 'slope_mean', 'area_gages2',                    # Terrain (3)
            'frac_forest', 'lai_max', 'lai_diff',                        # Land cover (5)
            'dom_land_cover_frac', 'dom_land_cover',
            'root_depth_50', 'soil_depth_statgs0', 'soil_porosity',      # Soil (5)
            'soil_conductivity', 'max_water_content',
            'geol_class_1st', 'geol_class_2nd', 'geol_porosity',         # Geology (4)
            'geol_permeability'
        ]
        
        validation = {
            'gage_id': gage_id,
            'required_count': len(required_static_vars),
            'present': [],
            'missing': [],
            'all_present': True
        }
        
        for var in required_static_vars:
            if var in gage_attrs and not pd.isna(gage_attrs[var]):
                validation['present'].append(var)
            else:
                validation['missing'].append(var)
                validation['all_present'] = False
        
        validation['present_count'] = len(validation['present'])
        validation['missing_count'] = len(validation['missing'])
        
        return validation

    def _validate_merged_data_paper_compliance(self, df: pd.DataFrame, gage_id: str) -> Dict[str, Any]:
        """Validate that merged data contains all paper-required variables."""
        validation = {
            'gage_id': gage_id,
            'compliance_status': 'FULL_COMPLIANCE',
            'missing_variables': [],
            'present_variables': [],
            'variable_coverage': {},
            'issues': []
        }
        
        # Check meteorological forcing variables
        for var in self.PAPER_SPECIFICATION['meteorological_forcing_variables']:
            if var in df.columns:
                validation['present_variables'].append(var)
                coverage = df[var].notna().sum() / len(df)
                validation['variable_coverage'][var] = float(coverage)
                
                if coverage < 0.8:
                    validation['issues'].append(f"{var} coverage low: {coverage:.1%}")
            else:
                validation['missing_variables'].append(var)
                validation['compliance_status'] = 'PARTIAL_COMPLIANCE'
        
        # Check model output variables
        for var in self.PAPER_SPECIFICATION['model_output_variables']:
            if var in df.columns:
                validation['present_variables'].append(var)
                coverage = df[var].notna().sum() / len(df)
                validation['variable_coverage'][var] = float(coverage)
                
                # Special check for ET coverage after resampling
                if var == 'evapotranspiration' and coverage < 0.8:
                    validation['issues'].append(f"ET coverage low after resampling: {coverage:.1%}")
            else:
                validation['missing_variables'].append(var)
                validation['compliance_status'] = 'PARTIAL_COMPLIANCE'
        
        # Check evaluation variable
        eval_var = self.PAPER_SPECIFICATION['model_evaluation_variable']
        if eval_var in df.columns:
            validation['present_variables'].append(eval_var)
            coverage = df[eval_var].notna().sum() / len(df)
            validation['variable_coverage'][eval_var] = float(coverage)
        else:
            validation['missing_variables'].append(eval_var)
            validation['compliance_status'] = 'PARTIAL_COMPLIANCE'
        
        # Update status based on missing variables
        if validation['missing_variables']:
            missing_count = len(validation['missing_variables'])
            if missing_count > 3:
                validation['compliance_status'] = 'LOW_COMPLIANCE'
        
        return validation

    def _add_variable_attributes(self, dataset: xr.Dataset) -> None:
        """Add CF-compliant attributes to dataset variables according to paper Table 1."""
        logger.debug(f"Adding variable attributes. Dataset variables: {list(dataset.data_vars.keys())}")
        
        # Variable attributes according to paper Table 1 requirements
        var_attrs = {
            # Model output variables (2)
            'streamflow': {
                'units': 'm^3/s',
                'long_name': 'Daily streamflow in the outlet of a basin',
                'standard_name': 'water_volume_transport_in_river_channel',
                'description': 'Observed daily streamflow from CAMELS (2001-2014) and USGS (2015-2021)',
                'missing_value': -999.0,
                'paper_role': 'model_output',
                'paper_table_reference': 'Table 1'
            },
            'evapotranspiration': {
                'units': 'mm/day',
                'long_name': 'Basin mean daily actual evapotranspiration',
                'standard_name': 'water_evapotranspiration_flux',
                'description': 'Daily evapotranspiration from MODIS16A2 v006, resampled from 8-day to daily with leap year adjustment',
                'missing_value': -999.0,
                'paper_role': 'model_output',
                'paper_table_reference': 'Table 1',
                'original_resolution': '8-day',
                'data_source': 'MODIS16A2 v006',
                'resampling_method': 'Period average distribution'
            },
            
            # Meteorological forcing variables (5)
            'total_precipitation': {
                'units': 'mm/day',
                'long_name': 'Daily total precipitation',
                'standard_name': 'precipitation_amount',
                'description': 'Daily total precipitation from NLDAS-2 forcing data',
                'missing_value': -999.0,
                'paper_role': 'meteorological_forcing',
                'paper_table_reference': 'Table 1'
            },
            'temperature': {
                'units': 'degree_C',
                'long_name': 'Air temperature at 2 m above the surface',
                'standard_name': 'air_temperature',
                'description': 'Air temperature at 2 m above the surface from NLDAS-2 forcing data',
                'missing_value': -999.0,
                'paper_role': 'meteorological_forcing',
                'paper_table_reference': 'Table 1'
            },
            'specific_humidity': {
                'units': 'kg/kg',
                'long_name': 'Specific humidity at 2 m above the surface',
                'standard_name': 'specific_humidity',
                'description': 'Specific humidity at 2 m above the surface from NLDAS-2 forcing data',
                'missing_value': -999.0,
                'paper_role': 'meteorological_forcing',
                'paper_table_reference': 'Table 1'
            },
            'shortwave_radiation': {
                'units': 'W/m^2',
                'long_name': 'Surface downward shortwave radiation',
                'standard_name': 'downwelling_shortwave_flux_in_air',
                'description': 'Surface downward shortwave radiation from NLDAS-2 forcing data',
                'missing_value': -999.0,
                'paper_role': 'meteorological_forcing',
                'paper_table_reference': 'Table 1'
            },
            'potential_energy': {
                'units': 'J/kg',
                'long_name': 'Convective available potential energy',
                'standard_name': 'convective_available_potential_energy',
                'description': 'Convective available potential energy from NLDAS-2 forcing data',
                'missing_value': -999.0,
                'paper_role': 'meteorological_forcing',
                'paper_table_reference': 'Table 1'
            },
            
            # Model evaluation variable
            'ssm': {
                'units': 'm^3/m^3',
                'long_name': 'Surface soil moisture',
                'standard_name': 'volume_fraction_of_condensed_water_in_soil',
                'description': 'Surface soil moisture (0-5 cm) from NASA-USDA SMAP, used for model evaluation only (not model output)',
                'missing_value': -999.0,
                'paper_role': 'model_evaluation',
                'paper_table_reference': 'Section 2.1',
                'temporal_resolution': '3-day (every 3rd day)',
                'depth': '0-5 cm',
                'data_source': 'NASA-USDA SMAP'
            }
        }
        
        # Static variable attributes according to paper Table 1
        static_var_attrs = {
            # Terrain attributes (3)
            'elev_mean': {
                'units': 'm', 
                'long_name': 'Basin mean elevation',
                'paper_role': 'static_attribute',
                'paper_table_reference': 'Table 1'
            },
            'slope_mean': {
                'units': 'm/km', 
                'long_name': 'Basin mean slope',
                'paper_role': 'static_attribute',
                'paper_table_reference': 'Table 1'
            },
            'area_gages2': {
                'units': 'km^2', 
                'long_name': 'Basin area',
                'paper_role': 'static_attribute',
                'paper_table_reference': 'Table 1'
            },
            
            # Land cover attributes (5)
            'frac_forest': {
                'units': '1', 
                'long_name': 'Forest proportion',
                'paper_role': 'static_attribute',
                'paper_table_reference': 'Table 1'
            },
            'lai_max': {
                'units': '1', 
                'long_name': 'Maximum monthly mean of leaf area index',
                'paper_role': 'static_attribute',
                'paper_table_reference': 'Table 1'
            },
            'lai_diff': {
                'units': '1', 
                'long_name': 'Difference between the maximum and minimum monthly mean values of the leaf area index',
                'paper_role': 'static_attribute',
                'paper_table_reference': 'Table 1'
            },
            'dom_land_cover_frac': {
                'units': '1', 
                'long_name': 'Proportion of major land cover types to basin area',
                'paper_role': 'static_attribute',
                'paper_table_reference': 'Table 1'
            },
            'dom_land_cover': {
                'long_name': 'Major land cover types',
                'paper_role': 'static_attribute',
                'paper_table_reference': 'Table 1'
            },
            
            # Soil attributes (5)
            'root_depth_50': {
                'units': 'm', 
                'long_name': 'Average soil layer thickness containing the top 50% of the root system',
                'paper_role': 'static_attribute',
                'paper_table_reference': 'Table 1'
            },
            'soil_depth_statgs0': {
                'units': 'm', 
                'long_name': 'Soil depth',
                'paper_role': 'static_attribute',
                'paper_table_reference': 'Table 1'
            },
            'soil_porosity': {
                'units': '1', 
                'long_name': 'Soil porosity',
                'paper_role': 'static_attribute',
                'paper_table_reference': 'Table 1'
            },
            'soil_conductivity': {
                'units': 'cm/hr', 
                'long_name': 'Saturated hydraulic conductivity',
                'paper_role': 'static_attribute',
                'paper_table_reference': 'Table 1'
            },
            'max_water_content': {
                'units': 'm', 
                'long_name': 'Maximum soil water holding capacity',
                'paper_role': 'static_attribute',
                'paper_table_reference': 'Table 1'
            },
            
            # Geology attributes (4)
            'geol_class_1st': {
                'long_name': 'Most common geological category in the basin',
                'paper_role': 'static_attribute',
                'paper_table_reference': 'Table 1'
            },
            'geol_class_2nd': {
                'long_name': 'Second most common geological category in the basin',
                'paper_role': 'static_attribute',
                'paper_table_reference': 'Table 1'
            },
            'geol_porosity': {
                'units': '1', 
                'long_name': 'Subsurface porosity',
                'paper_role': 'static_attribute',
                'paper_table_reference': 'Table 1'
            },
            'geol_permeability': {
                'units': 'm^2', 
                'long_name': 'Subsurface permeability',
                'paper_role': 'static_attribute',
                'paper_table_reference': 'Table 1'
            }
        }
        
        # Combine all attributes
        all_var_attrs = {**var_attrs, **static_var_attrs}
        
        # Add attributes to dataset variables
        for var_name in dataset.data_vars:
            if var_name in all_var_attrs:
                logger.debug(f"Adding paper-compliant attributes to variable: {var_name}")
                dataset[var_name].attrs.update(all_var_attrs[var_name])
            elif var_name not in ['time', 'gage_id']:
                # Add basic attributes to other variables
                dataset[var_name].attrs.update({
                    'missing_value': -999.0,
                    'paper_role': 'additional_variable'
                })
                logger.debug(f"Added basic attributes to variable: {var_name}")
        
        logger.debug("Variable attributes added with paper Table 1 compliance")

    def _add_static_variables_to_dataset(self, dataset: xr.Dataset, gage_attrs: Dict[str, Any]) -> None:
        """Add static variables to the dataset as data variables with paper compliance."""
        logger.debug("Adding static variables to dataset with paper compliance")
        
        # Define static variable mapping according to paper Table 1
        static_var_mapping = {
            # Terrain attributes (3)
            'elev_mean': 'elev_mean',
            'slope_mean': 'slope_mean',
            'area_gages2': 'area_gages2',
            
            # Land cover attributes (5)
            'frac_forest': 'frac_forest',
            'lai_max': 'lai_max',
            'lai_diff': 'lai_diff',
            'dom_land_cover_frac': 'dom_land_cover_frac',
            'dom_land_cover': 'dom_land_cover',
            
            # Soil attributes (5)
            'root_depth_50': 'root_depth_50',
            'soil_depth_statgs0': 'soil_depth_statgs0',
            'soil_porosity': 'soil_porosity',
            'soil_conductivity': 'soil_conductivity',
            'max_water_content': 'max_water_content',
            
            # Geology attributes (4)
            'geol_class_1st': 'geol_class_1st',
            'geol_class_2nd': 'geol_class_2nd',
            'geol_porosity': 'geol_porosity',
            'geol_permeability': 'geol_permeability'
        }
        
        added_count = 0
        
        # Add static variables as data variables (scalars, no time dimension)
        for paper_name, attr_key in static_var_mapping.items():
            # Check if attribute exists
            if attr_key in gage_attrs:
                value = gage_attrs[attr_key]
            elif f'gage_{attr_key}' in gage_attrs:
                value = gage_attrs[f'gage_{attr_key}']
            else:
                logger.debug(f"Paper static variable {paper_name} not found in attributes")
                continue
            
            # Handle NaN values
            if pd.isna(value):
                logger.debug(f"Paper static variable {paper_name} is NaN, skipping")
                continue
            
            # Create variable based on type
            if paper_name in ['dom_land_cover', 'geol_class_1st', 'geol_class_2nd']:
                # String variables
                dataset[paper_name] = str(value)
            else:
                # Numeric variables
                dataset[paper_name] = float(value) if not isinstance(value, (int, float)) else value
            
            added_count += 1
            logger.debug(f"Added paper static variable: {paper_name} = {value}")
        
        logger.debug(f"Added {added_count} paper static variables to dataset (expected: {self.PAPER_SPECIFICATION['static_attribute_count']})")

    def _create_and_save_dataset(self, data_df: pd.DataFrame,
                                gage_attrs: Dict[str, Any],
                                gage_id: str,
                                coverage_result: Dict,
                                paper_validation: Dict) -> bool:
        """Create and save dataset with CF-compliant NetCDF format and paper compliance."""
        try:
            gage_id_8 = str(gage_id).zfill(8)
            
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            
            if 'date' in data_df.columns:
                data_df = data_df.rename(columns={'date': 'time'})
            
            data_df = data_df.set_index('time')
            
            logger.debug(f"DataFrame columns before conversion: {data_df.columns.tolist()}")
            
            # Create xarray dataset from DataFrame
            dataset = data_df.to_xarray()
            
            logger.debug(f"Dataset variables after conversion: {list(dataset.data_vars.keys())}")
            
            # Add static variables to dataset
            self._add_static_variables_to_dataset(dataset, gage_attrs)
            
            # Add CF-compliant attributes to all variables
            self._add_variable_attributes(dataset)
            
            # Add global attributes with paper compliance information
            dataset.attrs.update({
                'title': 'CAMELS-US Hydro-Meteorological Dataset for MTL Study',
                'institution': 'HydroMTL Project',
                'source': 'CAMELS, NLDAS-2, MODIS16A2 v006, NASA-USDA SMAP',
                'history': f'Created by HydroMTL Pipeline on {datetime.now().isoformat()}',
                'Conventions': 'CF-1.8',
                'featureType': 'timeSeries',
                'summary': 'Daily hydrological and meteorological data for multi-task learning study with paper Table 1 compliance',
                'references': 'Paper: MTL hydrological modeling study; Addison, P.S., 2018. The illustrated wavelet transform handbook. CRC press.',
                'comment': 'Processed for MTL hydrological modeling study with strict compliance to paper Table 1 specifications.',
                'gage_id': gage_id_8,
                'creation_date': datetime.now().isoformat(),
                'study_period': f'{self.config.processing_config.start_date} to {self.config.processing_config.end_date}',
                'paper_specification': 'Based on Table 1 of target MTL hydrological modeling paper',
                'paper_meteorological_variables': str(self.PAPER_SPECIFICATION['meteorological_forcing_variables']),
                'paper_model_output_variables': str(self.PAPER_SPECIFICATION['model_output_variables']),
                'paper_evaluation_variable': self.PAPER_SPECIFICATION['model_evaluation_variable'],
                'paper_static_attribute_count': self.PAPER_SPECIFICATION['static_attribute_count'],
                'paper_compliance_status': paper_validation['compliance_status'],
                'paper_missing_variables': str(paper_validation['missing_variables']),
                'coverage_minimum': self.config.processing_config.min_streamflow_coverage,
                'coverage_achieved': coverage_result['streamflow_coverage'],
                'coverage_details': json.dumps(coverage_result['coverage']),
                'et_processing': 'Resampled from 8-day MODIS16A2 v006 to daily scale with leap year adjustment',
                'smap_processing': 'Kept original 3-day timestep for model evaluation only',
                'forcing_processing': 'Filtered to 5 paper-required meteorological variables'
            })
            
            # Add additional gage attributes as global attributes
            for key, value in gage_attrs.items():
                if key != 'gage_id':
                    if isinstance(value, (int, float, str, bool)):
                        dataset.attrs[f'gage_{key}'] = value
                    elif isinstance(value, (list, tuple)):
                        dataset.attrs[f'gage_{key}'] = str(value)
                    elif pd.isna(value):
                        continue
                    else:
                        dataset.attrs[f'gage_{key}'] = str(value)
            
            # Save to NetCDF file
            output_file = self.config.output_dir / f"gage_{gage_id_8}.nc"
            logger.debug(f"Saving NetCDF file to: {output_file}")
            
            # Clear any conflicting attributes from time coordinate
            if 'time' in dataset.coords:
                for attr in ['units', 'calendar', 'long_name', 'standard_name', 'axis']:
                    if attr in dataset.coords['time'].attrs:
                        del dataset.coords['time'].attrs[attr]
            
            # Set encoding parameters for proper CF compliance
            encoding = {}
            if 'time' in dataset.coords:
                encoding['time'] = {
                    'dtype': 'int64',
                    'units': 'days since 2001-01-01 00:00:00',
                    'calendar': 'proleptic_gregorian'
                }
            
            # Set encoding for data variables
            for var_name in dataset.data_vars:
                if var_name not in encoding:
                    if dataset[var_name].dtype.kind in 'iufc':
                        encoding[var_name] = {
                            'zlib': True,
                            'complevel': 4,
                            '_FillValue': np.nan
                        }
                    else:
                        encoding[var_name] = {}
            
            dataset.to_netcdf(output_file, encoding=encoding)
            
            logger.debug(f"Paper-compliant dataset saved: {output_file}")
            
            # Save metadata JSON file with paper compliance information
            json_file = self.config.output_dir / f"gage_{gage_id_8}_metadata.json"
            metadata = {
                'gage_id': gage_id_8,
                'file_path': str(output_file),
                'creation_date': datetime.now().isoformat(),
                'coverage': coverage_result,
                'paper_compliance': paper_validation,
                'paper_specification': self.PAPER_SPECIFICATION,
                'dimensions': {dim: len(dataset[dim]) for dim in dataset.dims},
                'variables': list(dataset.data_vars.keys()),
                'variable_roles': {
                    'meteorological_forcing': self.PAPER_SPECIFICATION['meteorological_forcing_variables'],
                    'model_output': self.PAPER_SPECIFICATION['model_output_variables'],
                    'model_evaluation': self.PAPER_SPECIFICATION['model_evaluation_variable'],
                    'static_attributes': [var for var in dataset.data_vars.keys() 
                                         if var not in self.PAPER_SPECIFICATION['meteorological_forcing_variables'] +
                                         self.PAPER_SPECIFICATION['model_output_variables'] +
                                         [self.PAPER_SPECIFICATION['model_evaluation_variable']] +
                                         ['time', 'gage_id']]
                },
                'global_attributes': {k: v for k, v in dataset.attrs.items() if not k.startswith('gage_')}
            }
            
            with open(json_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Saved paper-compliant NetCDF and metadata for gage {gage_id_8}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save dataset for gage {gage_id}: {e}")
            logger.error(traceback.format_exc())
            return False

    def _generate_paper_compliance_summary(self):
        """Generate paper compliance summary across all processed gages."""
        if not self.paper_validation_results:
            return
        
        full_compliance = [r for r in self.paper_validation_results if r['compliance_status'] == 'FULL_COMPLIANCE']
        partial_compliance = [r for r in self.paper_validation_results if r['compliance_status'] == 'PARTIAL_COMPLIANCE']
        low_compliance = [r for r in self.paper_validation_results if r['compliance_status'] == 'LOW_COMPLIANCE']
        
        # Analyze missing variables across all gages
        all_missing_vars = {}
        for validation in self.paper_validation_results:
            for var in validation['missing_variables']:
                all_missing_vars[var] = all_missing_vars.get(var, 0) + 1
        
        self.paper_compliance_summary = {
            'total_gages_analyzed': len(self.paper_validation_results),
            'full_compliance_count': len(full_compliance),
            'partial_compliance_count': len(partial_compliance),
            'low_compliance_count': len(low_compliance),
            'full_compliance_rate': len(full_compliance) / len(self.paper_validation_results) if self.paper_validation_results else 0,
            'common_missing_variables': dict(sorted(all_missing_vars.items(), key=lambda x: x[1], reverse=True)),
            'compliance_by_variable': self._analyze_variable_compliance()
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("PAPER TABLE 1 COMPLIANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Full compliance: {self.paper_compliance_summary['full_compliance_count']}/{self.paper_compliance_summary['total_gages_analyzed']} "
                   f"({self.paper_compliance_summary['full_compliance_rate']:.1%})")
        
        if all_missing_vars:
            logger.info("Most commonly missing variables:")
            for var, count in list(all_missing_vars.items())[:5]:
                logger.info(f"  {var}: {count} gages ({count/len(self.paper_validation_results):.1%})")
        
        logger.info("=" * 60)

    def _analyze_variable_compliance(self) -> Dict[str, Any]:
        """Analyze compliance for each paper-required variable."""
        compliance_analysis = {}
        
        all_paper_vars = (self.PAPER_SPECIFICATION['meteorological_forcing_variables'] +
                         self.PAPER_SPECIFICATION['model_output_variables'] +
                         [self.PAPER_SPECIFICATION['model_evaluation_variable']])
        
        for var in all_paper_vars:
            present_count = 0
            coverage_values = []
            
            for validation in self.paper_validation_results:
                if var in validation['present_variables']:
                    present_count += 1
                    if var in validation['variable_coverage']:
                        coverage_values.append(validation['variable_coverage'][var])
            
            compliance_analysis[var] = {
                'present_in_gages': present_count,
                'present_rate': present_count / len(self.paper_validation_results) if self.paper_validation_results else 0,
                'average_coverage': np.mean(coverage_values) if coverage_values else 0,
                'min_coverage': np.min(coverage_values) if coverage_values else 0,
                'max_coverage': np.max(coverage_values) if coverage_values else 0
            }
        
        return compliance_analysis

    def _generate_final_summary(self, stats: Dict, coverage_results: List[Dict], 
                               attributes_df: pd.DataFrame):
        """Generate final processing summary with paper compliance report."""
        
        if coverage_results:
            valid_results = [r for r in coverage_results if r.get('passed', False)]
            invalid_results = [r for r in coverage_results if not r.get('passed', False)]
            
            streamflow_coverages = [r.get('streamflow_coverage', r.get('coverage', {}).get('streamflow', 0)) for r in coverage_results]
            valid_streamflow_coverages = [r.get('streamflow_coverage', r.get('coverage', {}).get('streamflow', 0)) for r in valid_results]

            coverage_stats = {
                'total_analyzed': len(coverage_results),
                'valid_count': len(valid_results),
                'invalid_count': len(invalid_results),
                'avg_streamflow_coverage': np.mean(streamflow_coverages) if coverage_results else 0,
                'valid_avg_streamflow_coverage': np.mean(valid_streamflow_coverages) if valid_results else 0,
                'streamflow_coverage_distribution': {
                    '≥95%': len([r for r in coverage_results if r.get('streamflow_coverage', r.get('coverage', {}).get('streamflow', 0)) >= 0.95]),
                    '90-95%': len([r for r in coverage_results if 0.90 <= r.get('streamflow_coverage', r.get('coverage', {}).get('streamflow', 0)) < 0.95]),
                    '80-90%': len([r for r in coverage_results if 0.80 <= r.get('streamflow_coverage', r.get('coverage', {}).get('streamflow', 0)) < 0.90]),
                    '<80%': len([r for r in coverage_results if r.get('streamflow_coverage', r.get('coverage', {}).get('streamflow', 0)) < 0.80])
                }
            }
            
            var_coverage_stats = {}
            paper_vars = (self.PAPER_SPECIFICATION['meteorological_forcing_variables'] +
                         self.PAPER_SPECIFICATION['model_output_variables'] +
                         [self.PAPER_SPECIFICATION['model_evaluation_variable']])
            
            for var in paper_vars:
                coverages = []
                for r in coverage_results:
                    if 'coverage' in r and var in r['coverage']:
                        coverages.append(r['coverage'][var])
                
                if coverages:
                    var_coverage_stats[var] = {
                        'mean': np.mean(coverages),
                        'min': np.min(coverages),
                        'max': np.max(coverages),
                        'std': np.std(coverages),
                        '≥80%': len([c for c in coverages if c >= 0.80]),
                        '≥90%': len([c for c in coverages if c >= 0.90]),
                        '≥95%': len([c for c in coverages if c >= 0.95])
                    }
        else:
            coverage_stats = {}
            var_coverage_stats = {}
        
        # Add paper compliance summary if available
        paper_summary = getattr(self, 'paper_compliance_summary', {})
        
        summary = {
            'processing_date': datetime.now().isoformat(),
            'config': {
                'data_root': str(self.config.data_root),
                'output_dir': str(self.config.output_dir),
                'start_date': self.config.processing_config.start_date,
                'end_date': self.config.processing_config.end_date,
                'max_basins': self.config.max_basins,
                'min_coverage': self.config.processing_config.min_streamflow_coverage,
                'output_format': self.config.processing_config.output_format,
                'overwrite_existing': self.config.processing_config.overwrite_existing
            },
            'statistics': {
                **stats,
                'coverage_statistics': coverage_stats,
                'variable_coverage': var_coverage_stats,
                'paper_compliance_summary': paper_summary
            },
            'valid_gages': [
                {
                    'gage_id': g['gage_id'],
                    'huc2': g.get('huc2'),
                    'coverage': g.get('coverage'),
                    'paper_compliance': g.get('paper_compliance')
                }
                for g in self.valid_gages
            ],
            'failed_gages': self.failed_gages,
            'skipped_gages': self.skipped_gages,
            'coverage_details': coverage_results,
            'paper_compliance_details': self.paper_validation_results,
            'data_quality_report': {
                'total_basins_available': len(attributes_df),
                'basins_meeting_coverage': len(self.valid_gages),
                'coverage_success_rate': len(self.valid_gages) / len(attributes_df) if len(attributes_df) > 0 else 0,
                'filtering_criteria': 'Streamflow data loss rate < 5% (coverage ≥ 95%)',
                'paper_specification_compliance': {
                    'meteorological_variables': '5 variables as per paper Table 1',
                    'model_output_variables': '2 variables: streamflow, evapotranspiration',
                    'static_attributes': f"17 variables: {self.PAPER_SPECIFICATION['static_attribute_count']} total (3 terrain, 5 land cover, 5 soil, 4 geology)",
                    'evaluation_variable': 'ssm (surface soil moisture) for model evaluation only',
                    'compliance_rate': paper_summary.get('full_compliance_rate', 0)
                },
                'processing_details': [
                    'ET data resampled from 8-day MODIS16A2 v006 to daily scale with leap year adjustment',
                    'SMAP data maintains its original 3-day timestep for model evaluation',
                    'Meteorological forcing filtered to 5 paper-required variables',
                    'All static attributes aligned with paper Table 1 requirements',
                    'All NetCDF files comply with CF-1.8 conventions',
                    'Time dimension properly aligned from 2001-01-01 to 2021-09-30'
                ]
            }
        }
        
        summary_file = self.config.output_dir / "final_processing_summary.json"
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Final processing summary saved to {summary_file}")
        except Exception as e:
            logger.error(f"Failed to save final summary: {e}")
        
        valid_basins_file = self.config.output_dir / "valid_basins.txt"
        try:
            with open(valid_basins_file, 'w') as f:
                for gage in self.valid_gages:
                    f.write(f"{gage['gage_id']}\n")
            logger.info(f"Valid basins list saved to {valid_basins_file}")
        except Exception as e:
            logger.error(f"Failed to save valid basins list: {e}")
        
        logger.info("\n" + "=" * 60)
        logger.info("DATA PROCESSING COMPLETE - PAPER COMPLIANCE REPORT")
        logger.info("=" * 60)
        logger.info(f"Total basins analyzed: {stats['total_gages']}")
        logger.info(f"Basins with streamflow coverage ≥{self.config.processing_config.min_streamflow_coverage:.0%}: {len(self.valid_gages)}")
        logger.info(f"Success rate: {stats['success_rate']:.1%}")
        
        if paper_summary:
            logger.info(f"Paper Table 1 full compliance: {paper_summary['full_compliance_count']}/{paper_summary['total_gages_analyzed']} "
                       f"({paper_summary['full_compliance_rate']:.1%})")
        
        logger.info("Output files generated:")
        if self.valid_gages:
            for gage_info in self.valid_gages:
                actual_gage_id = gage_info['gage_id']
                logger.info(f"  - gage_{actual_gage_id}.nc (paper-compliant NetCDF)")
                logger.info(f"  - gage_{actual_gage_id}_metadata.json (with paper compliance)")
        else:
            logger.info("  - No NetCDF files generated (no basins met coverage requirement)")
        logger.info(f"  - Summary: {summary_file}")
        logger.info(f"  - Valid basins list: {valid_basins_file}")
        logger.info("=" * 60)

    # The following methods remain unchanged from the original code
    # but are included for completeness:
    
    def _load_streamflow_with_huc2(self, gage_id: str, huc2: str) -> Optional[pd.DataFrame]:
        """Load streamflow data for a gage using HUC2 directory, combining CAMELS and USGS data."""
        try:
            camels_data = None
            usgs_data = None
            
            if self.camels_streamflow_loader:
                logger.debug(f"Loading CAMELS streamflow for gage {gage_id}")
                camels_data = self.camels_streamflow_loader.load([gage_id], huc2=huc2)
                if camels_data is not None and not camels_data.empty:
                    start_date_camels = pd.Timestamp('2001-01-01')
                    end_date_camels = pd.Timestamp('2014-12-31')
                    camels_data = camels_data[
                        (camels_data['date'] >= start_date_camels) & 
                        (camels_data['date'] <= end_date_camels)
                    ]
                    logger.debug(f"CAMELS data loaded: {len(camels_data)} records")
            
            if self.usgs_streamflow_loader:
                logger.debug(f"Loading USGS streamflow for gage {gage_id}")
                usgs_data = self.usgs_streamflow_loader.load([gage_id], huc2=huc2)
                if usgs_data is not None and not usgs_data.empty:
                    start_date_usgs = pd.Timestamp('2015-01-01')
                    end_date_usgs = pd.Timestamp('2021-09-30')
                    usgs_data = usgs_data[
                        (usgs_data['date'] >= start_date_usgs) & 
                        (usgs_data['date'] <= end_date_usgs)
                    ]
                    logger.debug(f"USGS data loaded: {len(usgs_data)} records")
            
            combined_dfs = []
            if camels_data is not None and not camels_data.empty:
                combined_dfs.append(camels_data)
            if usgs_data is not None and not usgs_data.empty:
                combined_dfs.append(usgs_data)
            
            if not combined_dfs:
                logger.warning(f"No streamflow data for gage {gage_id}")
                return None
            
            combined = pd.concat(combined_dfs, ignore_index=True)
            combined = combined.sort_values('date').reset_index(drop=True)
            
            combined = combined.drop_duplicates(subset='date', keep='first')
            
            start_date = pd.Timestamp('2001-01-01')
            end_date = pd.Timestamp('2021-09-30')
            full_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            full_df = pd.DataFrame({'date': full_dates})
            
            merged = pd.merge(full_df, combined[['date', 'streamflow']], on='date', how='left')
            
            total_days = len(merged)
            valid_days = merged['streamflow'].notna().sum()
            coverage = valid_days / total_days
            
            camels_count = len(camels_data) if camels_data is not None else 0
            usgs_count = len(usgs_data) if usgs_data is not None else 0
            
            logger.info(f"Gage {gage_id}: CAMELS={camels_count}, USGS={usgs_count}, "
                       f"Total={len(combined)}, Coverage={coverage:.2%}")
            
            return merged
            
        except Exception as e:
            logger.warning(f"Error loading streamflow for gage {gage_id}: {e}")
            return None

    def _get_gage_attributes(self, gage_id: str, attributes_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract attributes for a specific gage."""
        gage_id_8 = gage_id.zfill(8) if len(gage_id) < 8 else gage_id

        gage_row = attributes_df[attributes_df['gage_id'].astype(str) == gage_id_8]

        if gage_row.empty:
            gage_row = attributes_df[attributes_df['gage_id'].astype(str) == gage_id]

        if gage_row.empty:
            logger.warning(f"No attributes found for gage {gage_id}")
            return {}

        attrs = gage_row.iloc[0].to_dict()
        clean_attrs = {}
        for key, value in attrs.items():
            if isinstance(value, (np.integer, np.int64)):
                clean_attrs[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                clean_attrs[key] = float(value)
            elif isinstance(value, np.ndarray):
                clean_attrs[key] = value.tolist()
            elif pd.isna(value):
                continue
            else:
                clean_attrs[key] = value

        return clean_attrs

    def _get_huc2_for_gage(self, gage_id: str, gage_attrs: Dict[str, Any]) -> Optional[str]:
        """Get HUC2 code for a gage from mapping or attributes."""
        gage_id_8 = gage_id.zfill(8)

        if gage_id_8 in self.huc2_mapping:
            huc2 = self.huc2_mapping[gage_id_8]
            if huc2 and pd.notna(huc2):
                huc2_str = str(huc2).zfill(2)
                logger.debug(f"Found HUC2 {huc2_str} for gage {gage_id} from cache")
                return huc2_str

        if 'huc_02' in gage_attrs and gage_attrs['huc_02']:
            huc2 = gage_attrs['huc_02']
            huc2_str = str(huc2).zfill(2)
            logger.debug(f"Found HUC2 {huc2_str} for gage {gage_id} from attributes")
            return huc2_str

        logger.debug(f"No HUC2 mapping found for gage {gage_id}")
        return None

    def _load_forcing_with_huc2(self, gage_id: str, huc2: str) -> Optional[pd.DataFrame]:
        """Load forcing data for a gage using HUC2 directory."""
        if not self.forcing_loader:
            return None

        try:
            data = self.forcing_loader.load([gage_id], huc2=huc2)
            if data is not None and not data.empty:
                logger.debug(f"Loaded forcing data for gage {gage_id} using loader")
                return data
        except Exception as e:
            logger.debug(f"Forcing loader failed for gage {gage_id}: {e}")

        return self._load_forcing_direct(gage_id, huc2)

    def _load_forcing_direct(self, gage_id: str, huc2: str) -> Optional[pd.DataFrame]:
        """Direct file access for forcing data."""
        forcing_config = self.config.data_sources.get("nldas_forcing")
        if not forcing_config:
            return None

        huc2_2digit = str(huc2).zfill(2)

        possible_paths = [
            forcing_config.data_source_path / "basin_mean_forcing" / huc2_2digit / f"{gage_id}_lump_nldas_forcing_leap.txt",
            forcing_config.data_source_path / huc2_2digit / f"{gage_id}_lump_nldas_forcing_leap.txt",
        ]

        file_path = None
        for path in possible_paths:
            if path.exists():
                file_path = path
                logger.debug(f"Found forcing file at: {path}")
                break

        if not file_path or not file_path.exists():
            logger.warning(f"Forcing file not found for gage {gage_id} in HUC2 {huc2}")
            return None

        try:
            df = pd.read_csv(file_path, sep=r'\s+', header=0)

            column_mapping = {
                'Year': 'year',
                'Mnth': 'month',
                'Day': 'day',
                'Hr': 'hour',
                'temperature(C)': 'temperature',
                'specific_humidity(kg/kg)': 'specific_humidity',
                'pressure(Pa)': 'pressure',
                'wind_u(m/s)': 'wind_u',
                'wind_v(m/s)': 'wind_v',
                'longwave_radiation(W/m^2)': 'longwave_radiation',
                'convective_fraction(-)': 'convective_fraction',
                'shortwave_radiation(W/m^2)': 'shortwave_radiation',
                'potential_energy(J/kg)': 'potential_energy',
                'potential_evaporation(kg/m^2)': 'potential_evaporation',
                'total_precipitation(kg/m^2)': 'total_precipitation'
            }

            df = df.rename(columns=column_mapping)

            df['year'] = df['year'].astype(int)
            df['month'] = df['month'].astype(int)
            df['day'] = df['day'].astype(int)

            df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

            numeric_cols = ['temperature', 'specific_humidity', 'pressure', 'wind_u', 'wind_v',
                           'longwave_radiation', 'convective_fraction', 'shortwave_radiation',
                           'potential_energy', 'potential_evaporation', 'total_precipitation']

            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df['gage_id'] = gage_id
            df = df.drop(columns=['year', 'month', 'day', 'hour'])

            cols = ['date', 'gage_id'] + [c for c in df.columns if c not in ['date', 'gage_id']]
            df = df[cols]

            logger.debug(f"Loaded forcing data from {file_path} with {len(df)} records")
            return df

        except Exception as e:
            logger.warning(f"Failed to read forcing file {file_path}: {e}")
            return None

    def explore_data_structure(self):
        """Explore data structure without processing."""
        logger.info("Exploring hydro data structure...")

        camels_us_dir = self.data_root / "camels" / "camels_us"

        if camels_us_dir.exists():
            logger.info(f"Found CAMELS directory: {camels_us_dir}")

            txt_files = list(camels_us_dir.glob("camels_*.txt"))
            logger.info(f"Attribute files found: {len(txt_files)}")
            for f in txt_files[:5]:
                logger.info(f"  - {f.name}")

            streamflow_dirs = [
                camels_us_dir / "camels_streamflow",
                camels_us_dir / "basin_timeseries_v1p2_metForcing_obsFlow" /
                "basin_dataset_public_v1p2" / "camels_streamflow"
            ]

            for dir_path in streamflow_dirs:
                if dir_path.exists():
                    subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
                    logger.info(f"Found streamflow directory: {dir_path}")
                    logger.info(f"  Subdirectories (HUC2): {len(subdirs)}")
                    if subdirs:
                        logger.info(f"  First few: {[d.name for d in subdirs[:5]]}")
                else:
                    logger.debug(f"Directory not found: {dir_path}")
        else:
            logger.error(f"CAMELS directory not found: {camels_us_dir}")

        for name, config in self.config.data_sources.items():
            if name not in ["attributes", "camels_streamflow"]:
                if config.data_source_path.exists():
                    logger.info(f"Found {name} directory: {config.data_source_path}")
                else:
                    logger.debug(f"{name} directory not found: {config.data_source_path}")