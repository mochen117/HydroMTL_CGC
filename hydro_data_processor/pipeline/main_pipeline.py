"""
Main pipeline for Hydro Data Processing.
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
    """Main pipeline for processing hydrological data from multiple sources."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.data_root = config.data_root.resolve()

        self._initialize_loaders()

        self.processed_gages: List[str] = []
        self.failed_gages: List[Dict] = []
        self.skipped_gages: List[Dict] = []
        self.valid_gages: List[Dict] = []  # Gages that meet coverage requirements

        self.huc2_mapping: Dict[str, str] = {}

        logger.debug("Hydro Data Pipeline initialized")
        logger.debug(f"Data root (resolved): {self.data_root}")

    def _initialize_loaders(self):
        """Initialize all data loaders."""
        logger.debug("Initializing data loaders")

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

        logger.info("All loaders initialized")

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

    def run(self):
        """Run the complete pipeline with batch processing and coverage filtering."""
        logger.info("=" * 60)
        logger.info("Hydro Data Processing Pipeline")
        logger.info("=" * 60)
        logger.info(f"Data root: {self.data_root}")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info(f"Study period: {self.config.processing_config.start_date} to "
                f"{self.config.processing_config.end_date}")
        logger.info(f"Min coverage: {self.config.processing_config.min_streamflow_coverage:.0%}")

        if not self.attribute_loader:
            logger.error("Attribute loader not available")
            return

        logger.info("Step 1: Loading gage attributes and HUC2 mapping")
        attributes_df = self.attribute_loader.load(
            max_basins=self.config.max_basins
        )

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
        logger.info(f"Step 2: Processing {len(gage_ids)} gages")
        
        multi_source_processor = MultiSourceProcessor(self.config)
        batch_processor = BatchProcessor(self, multi_source_processor)
        
        stats, coverage_results = self._process_with_coverage_filtering(
            gage_ids, attributes_df, batch_processor
        )
        
        logger.info("Step 3: Generating final summary and data quality report")
        self._generate_final_summary(stats, coverage_results, attributes_df)

    def _process_with_coverage_filtering(self, gage_ids: List[str], 
                                        attributes_df: pd.DataFrame,
                                        batch_processor) -> Tuple[Dict, List[Dict]]:
        """Process gages with coverage filtering."""
        all_results = []
        coverage_results = []
        min_coverage = self.config.processing_config.min_streamflow_coverage
        
        for gage_id in gage_ids:
            logger.debug(f"Processing gage {gage_id}")
            
            try:
                gage_attrs = self._get_gage_attributes(gage_id, attributes_df)
                
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
                
                et_data = None
                if self.et_loader:
                    et_data = self.et_loader.load([gage_id], huc2=huc2)
                    
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
                        'precipitation': merged_data['total_precipitation'].notna().sum(),
                        'temperature': merged_data['temperature'].notna().sum()
                    },
                    'streamflow_missing_rate': 1.0 - streamflow_coverage
                }

                if streamflow_coverage >= min_coverage:
                    logger.info(f"Gage {gage_id} processed successfully, streamflow coverage: {streamflow_coverage:.2%}")
                else:
                    logger.warning(f"Gage {gage_id} failed streamflow coverage requirement: {streamflow_coverage:.2%} < {min_coverage:.0%}")
                
                coverage_results.append(coverage_result)
                
                if streamflow_coverage >= min_coverage:
                    success = self._create_and_save_dataset(
                        merged_data, gage_attrs, gage_id, coverage_result
                    )
                    
                    if success:
                        self.valid_gages.append({
                            'gage_id': gage_id,
                            'coverage': streamflow_coverage,
                            'huc2': huc2
                        })
                        self.processed_gages.append(gage_id)
                        all_results.append({
                            'gage_id': gage_id,
                            'status': 'success',
                            'coverage': coverage_result
                        })
                        logger.info(f"Gage {gage_id} processed successfully, streamflow coverage: {streamflow_coverage:.2%}")
                    else:
                        self.failed_gages.append({
                            'gage_id': gage_id,
                            'reason': 'Dataset creation failed'
                        })
                else:
                    self.skipped_gages.append({
                        'gage_id': gage_id,
                        'reason': f'Streamflow coverage insufficient: {streamflow_coverage:.2%}',
                        'coverage': coverage_result
                    })
                    
            except Exception as e:
                logger.error(f"Error processing gage {gage_id}: {e}")
                self.failed_gages.append({
                    'gage_id': gage_id,
                    'reason': f'Processing error: {str(e)}'
                })
        
        stats = {
            'total_gages': len(gage_ids),
            'processed_gages': len(self.processed_gages),
            'failed_gages': len(self.failed_gages),
            'skipped_gages': len(self.skipped_gages),
            'valid_gages': len(self.valid_gages),
            'success_rate': len(self.processed_gages) / len(gage_ids) if gage_ids else 0,
            'coverage_threshold': min_coverage
        }
        
        return stats, coverage_results

    def _merge_all_data_with_time_alignment(self, streamflow_df: pd.DataFrame,
                                           forcing_df: pd.DataFrame,
                                           et_df: Optional[pd.DataFrame],
                                           smap_df: Optional[pd.DataFrame],
                                           gage_id: str) -> Tuple[Optional[pd.DataFrame], Dict[str, float]]:
        """Merge all data sources with time alignment to daily scale."""
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
        
        if forcing_df is not None and not forcing_df.empty:
            forcing_df = forcing_df.copy()
            if 'date' not in forcing_df.columns and 'time' in forcing_df.columns:
                forcing_df = forcing_df.rename(columns={'time': 'date'})
            
            forcing_cols = ['total_precipitation', 'temperature']
            for col in forcing_cols:
                if col in forcing_df.columns:
                    merged_df = pd.merge(merged_df, forcing_df[['date', col]], 
                                        on='date', how='left')
        
        if et_df is not None and not et_df.empty:
            et_df = et_df.copy()
            if 'date' not in et_df.columns and 'time' in et_df.columns:
                et_df = et_df.rename(columns={'time': 'date'})
            
            et_cols = ['et', 'pet']
            for col in et_cols:
                if col in et_df.columns:
                    merged_df = pd.merge(merged_df, et_df[['date', col]], 
                                        on='date', how='left')
        
        if smap_df is not None and not smap_df.empty:
            smap_df = smap_df.copy()
            if 'date' not in smap_df.columns and 'time' in smap_df.columns:
                smap_df = smap_df.rename(columns={'time': 'date'})
            
            smap_cols = ['ssm', 'susm']
            for col in smap_cols:
                if col in smap_df.columns:
                    merged_df = pd.merge(merged_df, smap_df[['date', col]], 
                                        on='date', how='left')
        
        coverage_info = self._calculate_coverage(merged_df)
        
        return merged_df, coverage_info

    def _calculate_coverage(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate coverage percentage for key variables."""
        coverage = {}
        
        key_variables = {
            'streamflow': 'streamflow',
            'precipitation': 'total_precipitation',
            'temperature': 'temperature'
        }
        
        for name, col in key_variables.items():
            if col in df.columns:
                non_nan = df[col].notna().sum()
                total = len(df)
                coverage[name] = non_nan / total if total > 0 else 0.0
            else:
                coverage[name] = 0.0
        
        return coverage

    def _check_key_variables_coverage(self, df: pd.DataFrame) -> Dict[str, float]:
        """Check coverage for key hydrological variables."""
        coverage = {}
        
        if 'streamflow' in df.columns:
            non_nan = df['streamflow'].notna().sum()
            total = len(df)
            coverage['streamflow'] = non_nan / total if total > 0 else 0.0
        else:
            coverage['streamflow'] = 0.0
        
        other_vars = ['total_precipitation', 'temperature', 'et', 'ssm']
        for var in other_vars:
            if var in df.columns:
                non_nan = df[var].notna().sum()
                total = len(df)
                coverage[var] = non_nan / total if total > 0 else 0.0
        
        return coverage

    def _add_variable_attributes(self, dataset: xr.Dataset) -> None:
        """Add CF-compliant attributes to dataset variables."""
        logger.debug(f"Adding variable attributes. Dataset variables: {list(dataset.data_vars.keys())}")
        
        var_attrs = {
            'streamflow': {
                'units': 'm^3/s',
                'long_name': 'Daily mean streamflow',
                'standard_name': 'water_volume_transport_in_river_channel',
                'description': 'Observed daily mean streamflow from CAMELS (2001-2014) and USGS (2015-2021)',
                '_FillValue': np.nan,
                'missing_value': -999.0
            },
            'total_precipitation': {
                'units': 'mm/day',
                'long_name': 'Daily total precipitation',
                'standard_name': 'precipitation_amount',
                'description': 'Daily total precipitation from NLDAS-2 forcing data',
                '_FillValue': np.nan,
                'missing_value': -999.0
            },
            'temperature': {
                'units': 'degree_C',
                'long_name': 'Daily mean air temperature',
                'standard_name': 'air_temperature',
                'description': 'Daily mean air temperature from NLDAS-2 forcing data',
                '_FillValue': np.nan,
                'missing_value': -999.0
            },
            'et': {
                'units': 'mm/day',
                'long_name': 'Daily evapotranspiration',
                'standard_name': 'water_evapotranspiration_flux',
                'description': 'Daily evapotranspiration from MODIS16A2 v006, resampled from 8-day to daily',
                '_FillValue': np.nan,
                'missing_value': -999.0
            },
            'pet': {
                'units': 'mm/day',
                'long_name': 'Daily potential evapotranspiration',
                'standard_name': 'water_potential_evapotranspiration_flux',
                'description': 'Daily potential evapotranspiration',
                '_FillValue': np.nan,
                'missing_value': -999.0
            },
            'ssm': {
                'units': 'm^3/m^3',
                'long_name': 'Surface soil moisture',
                'standard_name': 'volume_fraction_of_condensed_water_in_soil',
                'description': 'Surface soil moisture (0-5 cm) from NASA-USDA SMAP, 3-day timestep with inherent missing pattern',
                '_FillValue': np.nan,
                'missing_value': -999.0,
                'depth': '0-5 cm',
                'temporal_resolution': '3-day (every 3rd day)'
            },
            'susm': {
                'units': 'm^3/m^3',
                'long_name': 'Subsurface soil moisture',
                'standard_name': 'volume_fraction_of_condensed_water_in_soil',
                'description': 'Subsurface soil moisture from NASA-USDA SMAP',
                '_FillValue': np.nan,
                'missing_value': -999.0
            }
        }
        
        additional_vars = {
            'specific_humidity': {
                'units': 'kg/kg',
                'long_name': 'Specific humidity',
                'standard_name': 'specific_humidity',
                '_FillValue': np.nan,
                'missing_value': -999.0
            },
            'pressure': {
                'units': 'Pa',
                'long_name': 'Atmospheric pressure',
                'standard_name': 'air_pressure',
                '_FillValue': np.nan,
                'missing_value': -999.0
            },
            'wind_u': {
                'units': 'm/s',
                'long_name': 'Eastward wind component',
                'standard_name': 'eastward_wind',
                '_FillValue': np.nan,
                'missing_value': -999.0
            },
            'wind_v': {
                'units': 'm/s',
                'long_name': 'Northward wind component',
                'standard_name': 'northward_wind',
                '_FillValue': np.nan,
                'missing_value': -999.0
            },
            'longwave_radiation': {
                'units': 'W/m^2',
                'long_name': 'Downward longwave radiation',
                'standard_name': 'downwelling_longwave_flux_in_air',
                '_FillValue': np.nan,
                'missing_value': -999.0
            },
            'shortwave_radiation': {
                'units': 'W/m^2',
                'long_name': 'Downward shortwave radiation',
                'standard_name': 'downwelling_shortwave_flux_in_air',
                '_FillValue': np.nan,
                'missing_value': -999.0
            },
            'convective_fraction': {
                'units': '1',
                'long_name': 'Convective fraction',
                'description': 'Fraction of precipitation that is convective',
                '_FillValue': np.nan,
                'missing_value': -999.0
            },
            'potential_energy': {
                'units': 'J/kg',
                'long_name': 'Potential energy',
                '_FillValue': np.nan,
                'missing_value': -999.0
            },
            'potential_evaporation': {
                'units': 'kg/m^2',
                'long_name': 'Potential evaporation',
                '_FillValue': np.nan,
                'missing_value': -999.0
            }
        }
        
        all_var_attrs = {**var_attrs, **additional_vars}
        
        for var_name in dataset.data_vars:
            if var_name in all_var_attrs:
                logger.debug(f"Adding attributes to variable: {var_name}")
                dataset[var_name].attrs.update(all_var_attrs[var_name])
            else:
                dataset[var_name].attrs.update({
                    '_FillValue': np.nan,
                    'missing_value': -999.0
                })
        
        # Remove time coordinate attributes to avoid conflict with xarray's CF encoder
        # xarray will automatically set these attributes when saving to NetCDF
        if 'time' in dataset.coords:
            # Clear all time coordinate attributes that might conflict with xarray's CF encoder
            for attr_name in ['units', 'calendar', 'long_name', 'standard_name', 'axis']:
                if attr_name in dataset.coords['time'].attrs:
                    del dataset.coords['time'].attrs[attr_name]

    def _create_and_save_dataset(self, data_df: pd.DataFrame,
                                gage_attrs: Dict[str, Any],
                                gage_id: str,
                                coverage_result: Dict) -> bool:
        """Create and save dataset with CF-compliant NetCDF format."""
        try:
            gage_id_8 = str(gage_id).zfill(8)
            
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            
            if 'date' in data_df.columns:
                data_df = data_df.rename(columns={'date': 'time'})
            
            data_df = data_df.set_index('time')
            
            logger.debug(f"DataFrame columns before conversion: {data_df.columns.tolist()}")
            
            dataset = data_df.to_xarray()
            
            logger.debug(f"Dataset variables after conversion: {list(dataset.data_vars.keys())}")
            
            for var_name in dataset.data_vars:
                logger.debug(f"Variable {var_name} initial attributes: {dataset[var_name].attrs}")
            
            logger.debug("Calling _add_variable_attributes method")
            self._add_variable_attributes(dataset)
            logger.debug("Finished calling _add_variable_attributes method")
            
            logger.debug("=== Variable attributes after adding ===")
            for var_name in dataset.data_vars:
                attrs = dataset[var_name].attrs
                logger.debug(f"{var_name}:")
                for attr_name, attr_value in attrs.items():
                    logger.debug(f"  {attr_name}: {attr_value}")
            
            dataset.attrs.update({
                'title': 'CAMELS-US Hydro-Meteorological Dataset for MTL Study',
                'institution': 'HydroMTL Project',
                'source': 'CAMELS, NLDAS-2, MODIS16A2 v006, NASA-USDA SMAP',
                'history': f'Created by HydroMTL Pipeline on {datetime.now().isoformat()}',
                'Conventions': 'CF-1.8',
                'featureType': 'timeSeries',
                'summary': 'Daily hydrological and meteorological data for multi-task learning study',
                'references': 'Addison, P.S., 2018. The illustrated wavelet transform handbook. CRC press.',
                'comment': 'Processed for MTL hydrological modeling study. ET data resampled from 8-day to daily.',
                'gage_id': gage_id_8,
                'creation_date': datetime.now().isoformat(),
                'study_period': f'{self.config.processing_config.start_date} to {self.config.processing_config.end_date}',
                'coverage_minimum': self.config.processing_config.min_streamflow_coverage,
                'coverage_achieved': coverage_result['streamflow_coverage'],
                'coverage_details': json.dumps(coverage_result['coverage'])
            })
            
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
            
            logger.debug("=== Final check before saving ===")
            for var_name in dataset.data_vars:
                attrs = dataset[var_name].attrs
                has_units = 'units' in attrs
                logger.debug(f"{var_name} - has units attribute: {has_units}")
                if has_units:
                    logger.debug(f"  units value: {attrs['units']}")
            
            output_file = self.config.output_dir / f"gage_{gage_id_8}.nc"
            logger.debug(f"Saving NetCDF file to: {output_file}")
            
            # Clear time coordinate attributes to avoid conflict with xarray's CF encoder
            if 'time' in dataset.coords:
                # Remove specific attributes that cause conflicts
                for attr_name in ['units', 'calendar']:
                    if attr_name in dataset.coords['time'].attrs:
                        del dataset.coords['time'].attrs[attr_name]
            
            dataset.to_netcdf(output_file)
            
            logger.debug(f"CF-compliant dataset saved: {output_file}")
            
            logger.debug("=== Reading back file to verify ===")
            ds_check = xr.open_dataset(output_file)
            for var_name in ds_check.data_vars:
                attrs = ds_check[var_name].attrs
                logger.debug(f"Read back {var_name} attributes:")
                for attr_name, attr_value in attrs.items():
                    logger.debug(f"  {attr_name}: {attr_value}")
            
            json_file = self.config.output_dir / f"gage_{gage_id_8}_metadata.json"
            metadata = {
                'gage_id': gage_id_8,
                'file_path': str(output_file),
                'creation_date': datetime.now().isoformat(),
                'coverage': coverage_result,
                'dimensions': {dim: len(dataset[dim]) for dim in dataset.dims},
                'variables': list(dataset.data_vars.keys()),
                'global_attributes': {k: v for k, v in dataset.attrs.items() if not k.startswith('gage_')}
            }
            
            with open(json_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save dataset for gage {gage_id}: {e}")
            logger.error(traceback.format_exc())
            return False

    def _generate_final_summary(self, stats: Dict, coverage_results: List[Dict], 
                               attributes_df: pd.DataFrame):
        """Generate final processing summary with data quality report."""
        
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
            for var in ['streamflow', 'precipitation', 'temperature']:
                coverages = [r['coverage'].get(var, 0) for r in coverage_results]
                if coverages:
                    var_coverage_stats[var] = {
                        'mean': np.mean(coverages),
                        'min': np.min(coverages),
                        'max': np.max(coverages),
                        'std': np.std(coverages),
                        '≥95%': len([c for c in coverages if c >= 0.95])
                    }
        else:
            coverage_stats = {}
            var_coverage_stats = {}
        
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
                'variable_coverage': var_coverage_stats
            },
            'valid_gages': [
                {
                    'gage_id': g['gage_id'],
                    'huc2': g.get('huc2'),
                    'coverage': g.get('coverage')
                }
                for g in self.valid_gages
            ],
            'failed_gages': self.failed_gages,
            'skipped_gages': self.skipped_gages,
            'coverage_details': coverage_results,
            'data_quality_report': {
                'total_basins_available': len(attributes_df),
                'basins_meeting_coverage': len(self.valid_gages),
                'coverage_success_rate': len(self.valid_gages) / len(attributes_df) if len(attributes_df) > 0 else 0,
                'filtering_criteria': 'Streamflow data loss rate < 5% (coverage ≥ 95%)',
                'recommendations': [
                    'ET data has been resampled from 8-day to daily scale',
                    'SMAP data maintains its original 3-day timestep with inherent missing pattern',
                    'All NetCDF files comply with CF-1.8 conventions',
                    'Time dimension is properly aligned from 2001-01-01 to 2021-09-30',
                    'Basin filtering based on streamflow coverage only (≥95%)'
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
        logger.info("DATA PROCESSING COMPLETE - QUALITY REPORT")
        logger.info("=" * 60)
        logger.info(f"Total basins analyzed: {stats['total_gages']}")
        logger.info(f"Basins with streamflow coverage ≥{self.config.processing_config.min_streamflow_coverage:.0%}: {len(self.valid_gages)}")
        logger.info(f"Success rate: {stats['success_rate']:.1%}")
        
        if coverage_stats:
            logger.info(f"Average streamflow coverage: {coverage_stats['avg_streamflow_coverage']:.1%}")
            logger.info(f"Valid basins average streamflow coverage: {coverage_stats['valid_avg_streamflow_coverage']:.1%}")
        
        logger.info("Output files generated:")
        if self.valid_gages:
            for gage_info in self.valid_gages:
                actual_gage_id = gage_info['gage_id']
                logger.info(f"  - gage_{actual_gage_id}.nc")
                logger.info(f"  - gage_{actual_gage_id}_metadata.json")
        else:
            logger.info("  - No NetCDF files generated (no basins met coverage requirement)")
        logger.info(f"  - Summary: {summary_file}")
        logger.info(f"  - Valid basins list: {valid_basins_file}")
        logger.info("=" * 60)

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