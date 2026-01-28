"""
Main pipeline for Hydro Data Processor
Optimized for cleaner logging
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
import re

from hydro_data_processor.config.settings import ProjectConfig, DataSourceConfig
from hydro_data_processor.loaders.attribute_loader import AttributeLoader
from hydro_data_processor.loaders.streamflow_loader import StreamflowLoader
from hydro_data_processor.loaders.forcing_loader import ForcingLoader
from hydro_data_processor.loaders.et_loader import ETLoader
from hydro_data_processor.loaders.smap_loader import SMAPLoader
from hydro_data_processor.processors.data_merger import DataMerger
from hydro_data_processor.processors.quality_checker import QualityChecker
from hydro_data_processor.processors.gap_handler import GapHandler
from hydro_data_processor.processors.time_processor import TimeProcessor
from hydro_data_processor.utils.io import save_json
from hydro_data_processor.utils.time import calculate_coverage

logger = logging.getLogger(__name__)


class HydroDataPipeline:
    """Main pipeline for hydrological data processing."""
    
    def __init__(self, config: ProjectConfig):
        """
        Initialize pipeline.
        
        Args:
            config: Project configuration
        """
        self.config = config
        self.processing_config = config.processing_config
        self.selected_basins = getattr(config, 'selected_basins', None)
        
        # Initialize loaders
        self._initialize_loaders()
        
        # Initialize processors
        self._initialize_processors()
        
        # Tracking variables
        self.processed_basins: List[str] = []
        self.failed_basins: List[Dict[str, Any]] = []
        self.skipped_basins: List[Dict[str, Any]] = []
        self.processing_stats: Dict[str, Any] = {}
        
        logger.debug("Pipeline initialized successfully")
    
    def _initialize_loaders(self):
        """Initialize data loaders."""
        logger.debug("Initializing data loaders")
        
        # Attribute loader
        attribute_config = self.config.data_sources["attributes"]
        self.attribute_loader = AttributeLoader(attribute_config)
        logger.debug(f"Attribute loader initialized")
        
        # Streamflow loaders (CAMELS and USGS)
        camels_streamflow_config = self.config.data_sources["camels_streamflow"]
        self.camels_streamflow_loader = StreamflowLoader(camels_streamflow_config)
        logger.debug("CAMELS streamflow loader initialized")
        
        usgs_streamflow_config = self.config.data_sources["usgs_streamflow"]
        self.usgs_streamflow_loader = StreamflowLoader(usgs_streamflow_config)
        logger.debug("USGS streamflow loader initialized")
        
        # Forcing loader
        forcing_config = self.config.data_sources["nldas_forcing"]
        self.forcing_loader = ForcingLoader(forcing_config)
        logger.debug("Forcing loader initialized")
        
        # ET loader
        et_config = self.config.data_sources["et_data"]
        self.et_loader = ETLoader(et_config)
        logger.debug("ET loader initialized")
        
        # SMAP loader
        smap_config = self.config.data_sources["smap_data"]
        self.smap_loader = SMAPLoader(smap_config)
        logger.debug("SMAP loader initialized")
        
        logger.info(f"All data loaders initialized")
    
    def _initialize_processors(self):
        """Initialize data processors."""
        logger.debug("Initializing data processors")
        
        # Time processor
        self.time_processor = TimeProcessor(
            start_date=self.processing_config.start_date,
            end_date=self.processing_config.end_date
        )
        
        # Data merger
        self.data_merger = DataMerger(
            start_date=self.processing_config.start_date,
            end_date=self.processing_config.end_date
        )
        
        # Quality checker
        self.quality_checker = QualityChecker(
            min_streamflow_coverage=self.processing_config.min_streamflow_coverage
        )
        
        # Gap handler
        self.gap_handler = GapHandler(method='paper')
        
        logger.debug("All processors initialized")
    
    def _extract_basin_id_from_string(self, basin_str: str) -> Optional[str]:
        """
        Extract 8-digit basin ID from a string that may contain additional data.
        
        Examples:
            Input: "02464000;4.27230937713895;3.00240327173169;..."
            Output: "02464000"
            
            Input: "02464146;1.51207298872002;0.363503007312376;..."
            Output: "02464146"
        """
        if not basin_str or pd.isna(basin_str):
            return None
        
        basin_str = str(basin_str).strip()
        
        # Split by semicolon and take first part
        if ';' in basin_str:
            parts = basin_str.split(';')
            first_part = parts[0].strip()
        else:
            first_part = basin_str
        
        # Extract 8-digit number
        match = re.search(r'\b\d{8}\b', first_part)
        if match:
            return match.group(0)
        
        # Try to find any 8-digit sequence
        digits = re.findall(r'\d+', first_part)
        for d in digits:
            if len(d) == 8:
                return d
        
        # If it's a number but not 8 digits, pad it
        if first_part.isdigit():
            padded = first_part.zfill(8)
            if len(padded) == 8:
                return padded
        
        logger.warning(f"Cannot extract 8-digit basin ID from: {basin_str[:50]}...")
        return None
    
    def run(self):
        """Run the complete data processing pipeline."""
        logger.info(f"Time period: {self.processing_config.start_date} to {self.processing_config.end_date}")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info(f"Max basins: {self.config.max_basins}")
        
        # Step 1: Load basin attributes
        logger.info("Loading basin attributes")
        attributes_df = self.attribute_loader.load(
            max_basins=self.config.max_basins
        )
        
        if attributes_df.empty:
            logger.error("No basin attributes loaded. Exiting.")
            return
        
        logger.info(f"Loaded attributes for {len(attributes_df)} rows")
        
        # Step 1.5: Clean and extract basin IDs from attributes
        # The attribute file seems to have full rows with basin_id as first column
        logger.info("Processing basin attributes to extract basin IDs...")
        
        # Extract clean basin IDs from the 'basin_id' column
        if 'basin_id' in attributes_df.columns:
            # Create a new column with cleaned basin IDs
            attributes_df['basin_id_clean'] = attributes_df['basin_id'].apply(
                self._extract_basin_id_from_string
            )
            
            # Remove rows where we couldn't extract a basin ID
            original_count = len(attributes_df)
            attributes_df = attributes_df.dropna(subset=['basin_id_clean'])
            removed_count = original_count - len(attributes_df)
            
            if removed_count > 0:
                logger.warning(f"Removed {removed_count} rows with invalid basin IDs")
            
            # Use the cleaned IDs
            basin_ids = attributes_df['basin_id_clean'].tolist()
            logger.info(f"Extracted {len(basin_ids)} valid basin IDs from attributes")
            
            # Show sample of extracted IDs
            if len(basin_ids) > 0:
                logger.info(f"Sample basin IDs: {basin_ids[:5]}")
        else:
            logger.error("'basin_id' column not found in attributes DataFrame")
            logger.error(f"Available columns: {list(attributes_df.columns)}")
            return
        
        # Step 2: Process each basin
        total_basins = len(basin_ids)
        logger.info(f"Processing {total_basins} basins")
        
        for i, basin_id in enumerate(basin_ids):
            logger.info(f"[{i+1}/{total_basins}] Processing basin {basin_id}")
            
            try:
                # Check if output already exists
                if self._output_exists(basin_id) and not self.processing_config.overwrite_existing:
                    logger.info(f"Output exists, skipping basin {basin_id}")
                    self.skipped_basins.append({
                        'basin_id': basin_id,
                        'reason': 'Output already exists',
                        'timestamp': datetime.now().isoformat()
                    })
                    continue
                
                # Process basin
                success = self._process_basin(basin_id, attributes_df)
                
                if success:
                    self.processed_basins.append(basin_id)
                    logger.info(f"Basin {basin_id} processed successfully")
                else:
                    logger.warning(f"Failed to process basin {basin_id}")
                
            except Exception as e:
                logger.error(f"Error processing basin {basin_id}: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    traceback.print_exc()
                self._record_failure(basin_id, str(e))
        
        # Step 3: Create combined dataset if multiple basins processed
        if len(self.processed_basins) > 1:
            logger.info(f"Creating combined dataset from {len(self.processed_basins)} basins")
            self._create_combined_dataset()
        
        # Step 4: Save processing summary
        logger.info("Saving processing summary")
        self._save_processing_summary(attributes_df)
        
        # Final statistics
        logger.info(f"Processing completed:")
        logger.info(f"  Successfully processed: {len(self.processed_basins)} basins")
        logger.info(f"  Failed: {len(self.failed_basins)} basins")
        logger.info(f"  Skipped: {len(self.skipped_basins)} basins")
        
        if self.failed_basins:
            logger.info(f"First 5 failed basins:")
            for failure in self.failed_basins[:5]:
                logger.info(f"  {failure['basin_id']}: {failure['reason']}")
    
    def _process_basin(self, basin_id: str, attributes_df: pd.DataFrame) -> bool:
        """
        Process a single basin.
        
        Args:
            basin_id: Basin ID (8-digit string)
            attributes_df: DataFrame with basin attributes
            
        Returns:
            True if successful
        """
        try:
            # 1. Get basin attributes
            basin_attrs = self._extract_basin_attributes(attributes_df, basin_id)
            if not basin_attrs:
                self._record_failure(basin_id, "No basin attributes")
                return False
            
            # 2. Load streamflow data
            camels_streamflow = self.camels_streamflow_loader.load(basin_ids=[basin_id])
            usgs_streamflow = self.usgs_streamflow_loader.load(basin_ids=[basin_id])
            
            # Merge streamflow sources
            merged_streamflow = self._merge_streamflow_sources(camels_streamflow, usgs_streamflow, basin_id)
            if merged_streamflow is None or merged_streamflow.empty:
                self._record_failure(basin_id, "No streamflow data")
                return False
            
            # Check streamflow coverage
            streamflow_coverage = merged_streamflow['streamflow'].notna().mean()
            if streamflow_coverage < self.processing_config.min_streamflow_coverage:
                self._record_failure(basin_id, f"Streamflow coverage too low: {streamflow_coverage:.1%}")
                return False
            
            logger.info(f"Basin {basin_id}: Streamflow coverage {streamflow_coverage:.1%}")
            
            # 3. Load forcing data
            forcing_data = self.forcing_loader.load(basin_ids=[basin_id])
            if forcing_data is None or forcing_data.empty:
                self._record_failure(basin_id, "No forcing data")
                return False
            
            # 4. Load ET data (optional)
            et_data = self.et_loader.load(basin_ids=[basin_id])
            if et_data is None or et_data.empty:
                logger.debug(f"No ET data for basin {basin_id}")
            
            # 5. Load SMAP data (optional)
            smap_data = self.smap_loader.load(basin_ids=[basin_id])
            if smap_data is None or smap_data.empty:
                logger.debug(f"No SMAP data for basin {basin_id}")
            
            # 6. Merge all data
            merged_df = self.data_merger.merge_basin_data(
                streamflow_data=merged_streamflow,
                forcing_data=forcing_data,
                et_data=et_data,
                smap_data=smap_data
            )
            
            if merged_df is None or merged_df.empty:
                self._record_failure(basin_id, "Data merge failed")
                return False
            
            # 7. Handle gaps
            merged_df = self.gap_handler.handle_missing_data(merged_df)
            
            # 8. Check data quality
            quality_check = self.quality_checker.check_dataset_quality(merged_df)
            
            if not quality_check['overall_valid']:
                issues = quality_check.get('issues', [])
                if issues:
                    issue_msg = "; ".join(issues[:3])
                    self._record_failure(basin_id, f"Quality issues: {issue_msg}")
                else:
                    self._record_failure(basin_id, "Quality check failed")
                return False
            
            # 9. Create xarray dataset
            dataset = self.data_merger.create_xarray_dataset(merged_df, basin_id, basin_attrs)
            if dataset is None:
                self._record_failure(basin_id, "Dataset creation failed")
                return False
            
            # 10. Save dataset
            success = self._save_dataset(dataset, basin_id)
            if not success:
                self._record_failure(basin_id, "Dataset save failed")
                return False
            
            # 11. Save quality report
            self._save_quality_report(quality_check, basin_id)
            
            logger.debug(f"Basin {basin_id} processed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error processing basin {basin_id}: {e}")
            if logger.isEnabledFor(logging.DEBUG):
                traceback.print_exc()
            self._record_failure(basin_id, str(e))
            return False
    
    def _extract_basin_attributes(self, attributes_df: pd.DataFrame, 
                                 basin_id: str) -> Dict[str, Any]:
        """
        Extract attributes for a specific basin.
        
        Args:
            attributes_df: DataFrame with all basin attributes
            basin_id: 8-digit basin ID
            
        Returns:
            Dictionary of basin attributes
        """
        # Find the row for this basin_id
        # First try using the cleaned basin_id column
        if 'basin_id_clean' in attributes_df.columns:
            basin_row = attributes_df[attributes_df['basin_id_clean'] == basin_id]
        else:
            # Fallback: search in original basin_id column
            basin_row = attributes_df[attributes_df['basin_id'].astype(str).str.contains(basin_id)]
        
        if basin_row is None or basin_row.empty:
            logger.debug(f"No attributes found for basin {basin_id}")
            return {}
        
        # Take the first matching row
        attrs_series = basin_row.iloc[0]
        
        # Parse the original basin_id string to extract all attributes
        original_basin_str = str(attrs_series['basin_id'])
        clean_attrs = {}
        
        # Split by semicolon to get all fields
        if ';' in original_basin_str:
            parts = original_basin_str.split(';')
            
            # The first part is the basin ID (should match our basin_id)
            if parts[0].strip() != basin_id:
                logger.warning(f"Mismatch: extracted basin_id {basin_id} != {parts[0].strip()}")
            
            # Store all parts as attributes
            for i, part in enumerate(parts):
                key = f"attr_{i}" if i > 0 else "basin_id"
                value = part.strip()
                
                # Try to convert to numeric if possible
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except (ValueError, TypeError):
                    pass
                
                clean_attrs[key] = value
        else:
            # If no semicolons, just store the whole string
            clean_attrs['basin_id'] = basin_id
            clean_attrs['original'] = original_basin_str
        
        # Also include any other columns from the attributes DataFrame
        for col in attrs_series.index:
            if col != 'basin_id' and col != 'basin_id_clean':
                value = attrs_series[col]
                if pd.notna(value):
                    if isinstance(value, (int, np.integer)):
                        clean_attrs[col] = int(value)
                    elif isinstance(value, (float, np.floating)):
                        clean_attrs[col] = float(value)
                    elif isinstance(value, str):
                        clean_attrs[col] = value
                    else:
                        clean_attrs[col] = str(value)
        
        logger.debug(f"Extracted {len(clean_attrs)} attributes for basin {basin_id}")
        return clean_attrs
        
    def _merge_streamflow_sources(self, camels_df: pd.DataFrame, 
                                 usgs_df: pd.DataFrame, 
                                 basin_id: str) -> pd.DataFrame:
        """
        Merge CAMELS and USGS streamflow data.
        
        Args:
            camels_df: CAMELS streamflow data (1980-2014)
            usgs_df: USGS streamflow data (2015-2021)
            basin_id: Basin ID
            
        Returns:
            Merged streamflow data
        """
        # Create base dataframe for study period
        base_dates = self.time_processor.create_study_period_index()
        merged_df = pd.DataFrame({'date': base_dates})
        
        # Function to prepare streamflow data
        def prepare_streamflow(df, source_name):
            if df is None or df.empty:
                logger.debug(f"{source_name} streamflow data is empty")
                return pd.DataFrame()
            
            df = df.copy()
            
            # Ensure date column
            if 'date' not in df.columns:
                logger.error(f"{source_name} data missing 'date' column")
                return pd.DataFrame()
            
            # Ensure streamflow column
            streamflow_col = None
            for col in ['streamflow', 'q_obs', 'discharge', 'Q']:
                if col in df.columns:
                    streamflow_col = col
                    break
            
            if streamflow_col is None:
                logger.error(f"{source_name} data missing streamflow column")
                return pd.DataFrame()
            
            # Select and rename columns
            df = df[['date', streamflow_col]].copy()
            df = df.rename(columns={streamflow_col: 'streamflow'})
            
            # Convert to datetime and ensure within study period
            df['date'] = pd.to_datetime(df['date'])
            df = df[
                (df['date'] >= self.time_processor.study_start) & 
                (df['date'] <= self.time_processor.study_end)
            ]
            
            # Add basin_id
            df['basin_id'] = basin_id
            
            return df
        
        # Prepare both data sources
        camels_prepared = prepare_streamflow(camels_df, "CAMELS")
        usgs_prepared = prepare_streamflow(usgs_df, "USGS")
        
        # Merge with preference for USGS data (2015 onwards)
        if not camels_prepared.empty and not usgs_prepared.empty:
            temp_merge = pd.merge(
                camels_prepared, 
                usgs_prepared, 
                on=['date', 'basin_id'], 
                how='outer',
                suffixes=('_camels', '_usgs')
            )
            
            if 'streamflow_camels' in temp_merge.columns and 'streamflow_usgs' in temp_merge.columns:
                cutoff_date = pd.Timestamp('2015-01-01')
                
                temp_merge['streamflow'] = np.where(
                    temp_merge['date'] < cutoff_date,
                    temp_merge['streamflow_camels'],
                    temp_merge['streamflow_usgs']
                )
                
                temp_merge['streamflow'] = temp_merge['streamflow'].fillna(
                    temp_merge['streamflow_camels']
                )
                
                merged_df = pd.merge(merged_df, temp_merge[['date', 'streamflow', 'basin_id']], 
                                   on='date', how='left')
            else:
                merged_df = pd.merge(merged_df, camels_prepared[['date', 'streamflow', 'basin_id']], 
                                   on='date', how='left')
        
        elif not camels_prepared.empty:
            merged_df = pd.merge(merged_df, camels_prepared[['date', 'streamflow', 'basin_id']], 
                               on='date', how='left')
        elif not usgs_prepared.empty:
            merged_df = pd.merge(merged_df, usgs_prepared[['date', 'streamflow', 'basin_id']], 
                               on='date', how='left')
        else:
            logger.error(f"No streamflow data available for basin {basin_id}")
            return pd.DataFrame()
        
        # Ensure basin_id is present
        if 'basin_id' not in merged_df.columns:
            merged_df['basin_id'] = basin_id
        
        return merged_df
    
    def _extract_basin_attributes(self, attributes_df: pd.DataFrame, 
                                 basin_id: str) -> Dict[str, Any]:
        """
        Extract attributes for a specific basin.
        
        Args:
            attributes_df: DataFrame with all basin attributes
            basin_id: 8-digit basin ID
            
        Returns:
            Dictionary of basin attributes
        """
        attributes_df = attributes_df.copy()
        
        if 'basin_id' in attributes_df.columns:
            attributes_df['basin_id'] = attributes_df['basin_id'].astype(str)
            basin_row = attributes_df[attributes_df['basin_id'] == basin_id]
        else:
            basin_row = None
            for col in attributes_df.columns:
                if attributes_df[col].astype(str).str.contains(basin_id).any():
                    basin_row = attributes_df[attributes_df[col].astype(str) == basin_id]
                    break
        
        if basin_row is None or basin_row.empty:
            logger.debug(f"No attributes found for basin {basin_id}")
            return {}
        
        attrs = basin_row.iloc[0].to_dict()
        clean_attrs = {}
        
        for key, value in attrs.items():
            if pd.isna(value):
                continue
            
            if isinstance(value, (int, np.integer)):
                clean_attrs[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                clean_attrs[key] = float(value)
            elif isinstance(value, str):
                clean_attrs[str(key)] = value
            else:
                clean_attrs[str(key)] = str(value)
        
        return clean_attrs
    
    def _output_exists(self, basin_id: str) -> bool:
        """Check if output file already exists."""
        if self.processing_config.output_format == "netcdf":
            output_file = self.config.output_dir / f"basin_{basin_id}.nc"
        elif self.processing_config.output_format == "hdf5":
            output_file = self.config.output_dir / f"basin_{basin_id}.h5"
        else:
            output_file = self.config.output_dir / f"basin_{basin_id}.parquet"
        
        return output_file.exists()
    
    def _save_dataset(self, dataset, basin_id: str) -> bool:
        """Save dataset to file."""
        try:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            
            if self.processing_config.output_format == "netcdf":
                filename = f"basin_{basin_id}.nc"
                filepath = self.config.output_dir / filename
                dataset.to_netcdf(filepath)
                
            elif self.processing_config.output_format == "hdf5":
                filename = f"basin_{basin_id}.h5"
                filepath = self.config.output_dir / filename
                dataset.to_netcdf(filepath)
                
            else:
                df = dataset.to_dataframe()
                filename = f"basin_{basin_id}.parquet"
                filepath = self.config.output_dir / filename
                df.to_parquet(filepath)
            
            logger.debug(f"Dataset saved: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save dataset for basin {basin_id}: {e}")
            return False
    
    def _save_quality_report(self, quality_check: Dict[str, Any], basin_id: str):
        """Save quality check report for a basin."""
        try:
            report_dir = self.config.output_dir / "quality_reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = report_dir / f"quality_{basin_id}.json"
            
            quality_check['basin_id'] = basin_id
            quality_check['timestamp'] = datetime.now().isoformat()
            
            save_json(quality_check, report_file)
            logger.debug(f"Quality report saved: {report_file}")
            
        except Exception as e:
            logger.debug(f"Failed to save quality report for basin {basin_id}: {e}")
    
    def _create_combined_dataset(self):
        """Create combined dataset from all processed basins."""
        if not self.processed_basins:
            logger.warning("No processed basins to combine")
            return
        
        logger.info(f"Creating combined dataset from {len(self.processed_basins)} basins")
        
        try:
            combined_datasets = []
            
            for basin_id in self.processed_basins:
                if self.processing_config.output_format == "netcdf":
                    filepath = self.config.output_dir / f"basin_{basin_id}.nc"
                    if filepath.exists():
                        ds = xr.open_dataset(filepath)
                        combined_datasets.append(ds)
                
                elif self.processing_config.output_format == "hdf5":
                    filepath = self.config.output_dir / f"basin_{basin_id}.h5"
                    if filepath.exists():
                        ds = xr.open_dataset(filepath)
                        combined_datasets.append(ds)
            
            if combined_datasets:
                combined = xr.concat(combined_datasets, dim='basin')
                
                combined_file = self.config.output_dir / f"combined_{len(self.processed_basins)}_basins.nc"
                combined.to_netcdf(combined_file)
                
                logger.info(f"Combined dataset saved: {combined_file}")
            else:
                logger.warning("No datasets found to combine")
                
        except Exception as e:
            logger.error(f"Failed to create combined dataset: {e}")
    
    def _record_failure(self, basin_id: str, reason: str):
        """Record processing failure."""
        self.failed_basins.append({
            'basin_id': basin_id,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.debug(f"Recorded failure for basin {basin_id}: {reason}")
    
    def _save_processing_summary(self, attributes_df: pd.DataFrame):
        """Save processing summary to file."""
        summary = {
            'processing_date': datetime.now().isoformat(),
            'config': {
                'data_root': str(self.config.data_root),
                'output_dir': str(self.config.output_dir),
                'start_date': self.processing_config.start_date,
                'end_date': self.processing_config.end_date,
                'min_streamflow_coverage': self.processing_config.min_streamflow_coverage,
                'output_format': self.processing_config.output_format,
                'max_basins': self.config.max_basins,
                'selected_basins': self.selected_basins
            },
            'statistics': {
                'total_basins_available': len(attributes_df),
                'basins_processed': len(self.processed_basins),
                'basins_failed': len(self.failed_basins),
                'basins_skipped': len(self.skipped_basins),
                'success_rate': len(self.processed_basins) / len(attributes_df) if len(attributes_df) > 0 else 0
            },
            'processed_basins': self.processed_basins,
            'failed_basins': self.failed_basins,
            'skipped_basins': self.skipped_basins
        }
        
        summary_file = self.config.output_dir / "processing_summary.json"
        
        if save_json(summary, summary_file):
            logger.info(f"Processing summary saved to {summary_file}")
        else:
            logger.error("Failed to save processing summary")
    
    def explore_data_structure(self):
        """Explore data structure without processing."""
        logger.info("Exploring data structure...")
        
        attributes_df = self.attribute_loader.load(max_basins=5)
        
        if attributes_df.empty:
            logger.warning("No basin attributes found")
            return
        
        basin_ids = attributes_df['basin_id'].head(3).tolist()
        
        exploration_results = {
            "total_basins_in_attributes": len(attributes_df),
            "sample_basins": basin_ids,
            "data_availability": {}
        }
        
        for basin_id in basin_ids:
            logger.info(f"Exploring data for basin {basin_id}")
            
            availability = {}
            
            for source_name, loader in [
                ("camels_streamflow", self.camels_streamflow_loader),
                ("usgs_streamflow", self.usgs_streamflow_loader),
                ("nldas_forcing", self.forcing_loader),
                ("et_data", self.et_loader),
                ("smap_data", self.smap_loader)
            ]:
                try:
                    data = loader.load(basin_ids=[basin_id])
                    availability[source_name] = not (data is None or data.empty)
                except Exception as e:
                    availability[source_name] = False
                    logger.debug(f"Error loading {source_name} for {basin_id}: {e}")
            
            exploration_results["data_availability"][basin_id] = availability
            
            # Log availability
            available_sources = [k for k, v in availability.items() if v]
            unavailable_sources = [k for k, v in availability.items() if not v]
            
            logger.info(f"  Available: {', '.join(available_sources) if available_sources else 'None'}")
            logger.info(f"  Unavailable: {', '.join(unavailable_sources) if unavailable_sources else 'None'}")
        
        # Save exploration results
        exploration_file = self.config.output_dir / "data_exploration.json"
        
        if save_json(exploration_results, exploration_file):
            logger.info(f"Exploration results saved to {exploration_file}")
        else:
            logger.warning("Failed to save exploration results")