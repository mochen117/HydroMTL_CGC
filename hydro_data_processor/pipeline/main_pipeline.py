"""
Main pipeline for Hydro Data Processing.
"""

import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
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
from hydro_data_processor.utils.io import save_json

logger = logging.getLogger(__name__)

__all__ = ['HydroDataPipeline']


class HydroDataPipeline:
    """Main pipeline for processing hydrological data from multiple sources."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.data_root = config.data_root.resolve()

        self._initialize_loaders()

        self.processed_gauges: List[str] = []
        self.failed_gauges: List[Dict] = []
        self.skipped_gauges: List[Dict] = []

        self.huc2_mapping: Dict[str, str] = {}

        logger.debug("Hydro Data Pipeline initialized")
        logger.debug(f"Data root (resolved): {self.data_root}")

    def _initialize_loaders(self):
        """Initialize all data loaders."""
        logger.debug("Initializing data loaders")

        # Attribute loader
        attribute_config = self.config.data_sources.get("attributes")
        if attribute_config:
            if not attribute_config.data_source_path.is_absolute():
                attribute_config.data_source_path = self.data_root / attribute_config.data_source_path
            self.attribute_loader = AttributeLoader(attribute_config)
        else:
            logger.error("No attribute configuration found")
            self.attribute_loader = None

        # Streamflow loaders
        camels_streamflow_config = self.config.data_sources.get("camels_streamflow")
        if camels_streamflow_config:
            if not camels_streamflow_config.data_source_path.is_absolute():
                camels_streamflow_config.data_source_path = self.data_root / camels_streamflow_config.data_source_path
            self.camels_streamflow_loader = StreamflowLoader(camels_streamflow_config)
        else:
            logger.error("No CAMELS streamflow configuration found")
            self.camels_streamflow_loader = None

        # USGS streamflow loader (optional)
        usgs_streamflow_config = self.config.data_sources.get("usgs_streamflow")
        if usgs_streamflow_config:
            if not usgs_streamflow_config.data_source_path.is_absolute():
                usgs_streamflow_config.data_source_path = self.data_root / usgs_streamflow_config.data_source_path
            self.usgs_streamflow_loader = StreamflowLoader(usgs_streamflow_config)
        else:
            logger.debug("No USGS streamflow configuration found")
            self.usgs_streamflow_loader = None

        # Forcing loader
        forcing_config = self.config.data_sources.get("nldas_forcing")
        if forcing_config:
            if not forcing_config.data_source_path.is_absolute():
                forcing_config.data_source_path = self.data_root / forcing_config.data_source_path
            self.forcing_loader = ForcingLoader(forcing_config)
        else:
            logger.error("No forcing configuration found")
            self.forcing_loader = None

        # ET loader (optional)
        et_config = self.config.data_sources.get("et_data")
        if et_config:
            if not et_config.data_source_path.is_absolute():
                et_config.data_source_path = self.data_root / et_config.data_source_path
            self.et_loader = ETLoader(et_config)
        else:
            logger.debug("No ET configuration found")
            self.et_loader = None

        # SMAP loader (optional)
        smap_config = self.config.data_sources.get("smap_data")
        if smap_config:
            if not smap_config.data_source_path.is_absolute():
                smap_config.data_source_path = self.data_root / smap_config.data_source_path
            self.smap_loader = SMAPLoader(smap_config)
        else:
            logger.debug("No SMAP configuration found")
            self.smap_loader = None

        logger.debug("All loaders initialized")

    def run(self):
        """Run the complete pipeline."""
        logger.info("=" * 60)
        logger.info("Hydro Data Processing Pipeline")
        logger.info("=" * 60)
        logger.info(f"Data root: {self.data_root}")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info(f"Max gauges: {self.config.max_basins}")
        logger.info(f"Study period: {self.config.processing_config.start_date} to {self.config.processing_config.end_date}")

        # Step 1: Load gauge attributes and HUC2 mapping
        if not self.attribute_loader:
            logger.error("Attribute loader not available")
            return

        logger.info("Step 1: Loading gauge attributes and HUC2 mapping")
        attributes_df = self.attribute_loader.load(max_basins=self.config.max_basins)

        if attributes_df.empty:
            logger.error("No attributes loaded. Exiting.")
            return

        # Ensure gauge_id column exists
        if 'basin_id' in attributes_df.columns and 'gauge_id' not in attributes_df.columns:
            attributes_df = attributes_df.rename(columns={'basin_id': 'gauge_id'})
            logger.debug("Renamed 'basin_id' column to 'gauge_id'")

        if 'gauge_id' not in attributes_df.columns:
            logger.error("No gauge_id column found in attributes")
            return

        # Format gauge_id to 8-digit string
        def ensure_8_digits(gauge_id):
            if gauge_id is None:
                return None
            gauge_str = str(gauge_id)
            if len(gauge_str) == 7:
                return '0' + gauge_str
            elif len(gauge_str) < 8:
                return gauge_str.zfill(8)
            return gauge_str

        attributes_df['gauge_id'] = attributes_df['gauge_id'].astype(str).apply(ensure_8_digits)

        # Extract HUC2 mapping
        if 'huc_02' in attributes_df.columns:
            attributes_df['huc_02'] = attributes_df['huc_02'].astype(str).str.zfill(2)
            self.huc2_mapping = dict(zip(attributes_df['gauge_id'], attributes_df['huc_02']))
            logger.info(f"Loaded HUC2 mapping for {len(self.huc2_mapping)} gauges")
            # Log first few mappings for debugging
            for gauge_id, huc2 in list(self.huc2_mapping.items())[:3]:
                logger.debug(f"HUC2 mapping: {gauge_id} -> {huc2}")
        else:
            logger.debug("No huc_02 column found in attributes")

        logger.debug(f"Attributes loaded: {len(attributes_df)} gauges")
        logger.debug(f"First gauge IDs: {attributes_df['gauge_id'].head().tolist()}")

        # Get gauge IDs to process
        gauge_ids = attributes_df['gauge_id'].tolist()
        logger.info(f"Step 2: Processing {len(gauge_ids)} gauges")

        # Process each gauge
        for i, gauge_id in enumerate(gauge_ids, 1):
            logger.info(f"[{i}/{len(gauge_ids)}] Processing gauge {gauge_id}")

            try:
                success = self._process_gauge_with_huc2(gauge_id, attributes_df)

                if success:
                    self.processed_gauges.append(gauge_id)
                    logger.info(f"✓ Gauge {gauge_id} processed successfully")
                else:
                    self.failed_gauges.append({'gauge_id': gauge_id, 'reason': 'Processing failed'})
                    logger.warning(f"✗ Failed to process gauge {gauge_id}")

            except Exception as e:
                logger.error(f"Error processing gauge {gauge_id}: {e}")
                traceback.print_exc()
                self.failed_gauges.append({'gauge_id': gauge_id, 'reason': str(e)[:100]})

        # Generate summary
        logger.info("Step 3: Generating processing summary")
        self._generate_summary(attributes_df)

        logger.info("=" * 60)
        logger.info(f"Pipeline completed:")
        logger.info(f"  Successfully processed: {len(self.processed_gauges)} gauges")
        logger.info(f"  Failed: {len(self.failed_gauges)} gauges")
        logger.info(f"  Skipped: {len(self.skipped_gauges)} gauges")

        if self.failed_gauges:
            logger.info("First 5 failed gauges:")
            for failure in self.failed_gauges[:5]:
                logger.info(f"  {failure['gauge_id']}: {failure['reason']}")

    def _process_gauge_with_huc2(self, gauge_id: str, attributes_df: pd.DataFrame) -> bool:
        """Process a single gauge with HUC2 directory support."""
        # Get gauge attributes
        gauge_attrs = self._get_gauge_attributes(gauge_id, attributes_df)

        # Get HUC2 code
        huc2 = self._get_huc2_for_gauge(gauge_id, gauge_attrs)
        if not huc2:
            logger.debug(f"No HUC2 found for gauge {gauge_id}")
            return False

        # Load streamflow data
        streamflow_data = self._load_streamflow_with_huc2(gauge_id, huc2)
        if streamflow_data is None or streamflow_data.empty:
            logger.warning(f"No streamflow data for gauge {gauge_id} in HUC2 {huc2}")
            return False

        # Load forcing data
        forcing_data = self._load_forcing_with_huc2(gauge_id, huc2)
        if forcing_data is None or forcing_data.empty:
            logger.warning(f"No forcing data for gauge {gauge_id} in HUC2 {huc2}")
            return False

        # Load optional ET data
        et_data = None
        if self.et_loader:
            et_data = self.et_loader.load([gauge_id], huc2=huc2)
            if et_data is None or et_data.empty:
                logger.debug(f"No ET data for gauge {gauge_id}")

        # Load optional SMAP data
        smap_data = None
        if self.smap_loader:
            smap_data = self.smap_loader.load([gauge_id], huc2=huc2)
            if smap_data is None or smap_data.empty:
                logger.debug(f"No SMAP data for gauge {gauge_id}")

        # Merge all data
        merged_data = self._merge_all_data(streamflow_data, forcing_data, et_data, smap_data, gauge_id)

        if merged_data is None or merged_data.empty:
            logger.warning(f"Failed to merge data for gauge {gauge_id}")
            return False

        # Create and save dataset
        success = self._create_and_save_dataset(merged_data, gauge_attrs, gauge_id)
        return success

    def _get_gauge_attributes(self, gauge_id: str, attributes_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract attributes for a specific gauge."""
        # Ensure gauge_id is 8-digit for comparison
        gauge_id_8 = gauge_id.zfill(8) if len(gauge_id) < 8 else gauge_id

        # Look for exact match
        gauge_row = attributes_df[attributes_df['gauge_id'].astype(str) == gauge_id_8]

        if gauge_row.empty:
            gauge_row = attributes_df[attributes_df['gauge_id'].astype(str) == gauge_id]

        if gauge_row.empty:
            logger.warning(f"No attributes found for gauge {gauge_id}")
            return {}

        attrs = gauge_row.iloc[0].to_dict()
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

    def _get_huc2_for_gauge(self, gauge_id: str, gauge_attrs: Dict[str, Any]) -> Optional[str]:
        """Get HUC2 code for a gauge from mapping or attributes."""
        # Ensure gauge_id is 8-digit
        gauge_id_8 = gauge_id.zfill(8)

        # First check cache
        if gauge_id_8 in self.huc2_mapping:
            huc2 = self.huc2_mapping[gauge_id_8]
            if huc2 and pd.notna(huc2):
                huc2_str = str(huc2).zfill(2)
                logger.debug(f"Found HUC2 {huc2_str} for gauge {gauge_id} from cache")
                return huc2_str

        # Check gauge attributes
        if 'huc_02' in gauge_attrs and gauge_attrs['huc_02']:
            huc2 = gauge_attrs['huc_02']
            huc2_str = str(huc2).zfill(2)
            logger.debug(f"Found HUC2 {huc2_str} for gauge {gauge_id} from attributes")
            return huc2_str

        logger.debug(f"No HUC2 mapping found for gauge {gauge_id}")
        return None

    def _load_streamflow_with_huc2(self, gauge_id: str, huc2: str) -> Optional[pd.DataFrame]:
        """Load streamflow data for a gauge using HUC2 directory."""
        if not self.camels_streamflow_loader:
            return None

        try:
            data = self.camels_streamflow_loader.load([gauge_id])
            if data is not None and not data.empty:
                return data
        except Exception as e:
            logger.debug(f"Loader failed for gauge {gauge_id}: {e}")

        return self._load_streamflow_direct(gauge_id, huc2)

    def _load_streamflow_direct(self, gauge_id: str, huc2: str) -> Optional[pd.DataFrame]:
        """Direct file access for streamflow data."""
        camels_config = self.config.data_sources.get("camels_streamflow")
        if not camels_config:
            return None

        file_path = camels_config.get_file_path(gauge_id, huc2)
        if not file_path.exists():
            return None

        try:
            df = pd.read_csv(file_path, sep=r'\s+', header=None, dtype=str)

            if df.shape[1] == 6:
                df.columns = ['file_gauge_id', 'year', 'month', 'day', 'streamflow', 'qc_flag']
            elif df.shape[1] == 5:
                df.columns = ['file_gauge_id', 'year', 'month', 'day', 'streamflow']
                df['qc_flag'] = None

            df['year'] = df['year'].astype(int)
            df['month'] = df['month'].astype(int)
            df['day'] = df['day'].astype(int)
            df['streamflow'] = pd.to_numeric(df['streamflow'], errors='coerce')

            df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
            df['gauge_id'] = gauge_id

            logger.debug(f"Loaded streamflow from {file_path}")
            return df[['date', 'streamflow', 'qc_flag', 'gauge_id']]

        except Exception as e:
            logger.warning(f"Failed to read streamflow file {file_path}: {e}")
            return None

    def _load_forcing_with_huc2(self, gauge_id: str, huc2: str) -> Optional[pd.DataFrame]:
        """Load forcing data for a gauge using HUC2 directory."""
        if not self.forcing_loader:
            return None

        try:
            data = self.forcing_loader.load([gauge_id], huc2=huc2)
            if data is not None and not data.empty:
                logger.debug(f"Loaded forcing data for gauge {gauge_id} using loader")
                return data
        except Exception as e:
            logger.debug(f"Forcing loader failed for gauge {gauge_id}: {e}")

        return self._load_forcing_direct(gauge_id, huc2)

    def _load_forcing_direct(self, gauge_id: str, huc2: str) -> Optional[pd.DataFrame]:
        """Direct file access for forcing data."""
        forcing_config = self.config.data_sources.get("nldas_forcing")
        if not forcing_config:
            return None

        huc2_2digit = str(huc2).zfill(2)

        possible_paths = [
            forcing_config.data_source_path / "basin_mean_forcing" / huc2_2digit / f"{gauge_id}_lump_nldas_forcing_leap.txt",
            forcing_config.data_source_path / huc2_2digit / f"{gauge_id}_lump_nldas_forcing_leap.txt",
        ]

        file_path = None
        for path in possible_paths:
            if path.exists():
                file_path = path
                logger.debug(f"Found forcing file at: {path}")
                break

        if not file_path or not file_path.exists():
            logger.warning(f"Forcing file not found for gauge {gauge_id} in HUC2 {huc2}")
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

            df['gauge_id'] = gauge_id
            df = df.drop(columns=['year', 'month', 'day', 'hour'])

            cols = ['date', 'gauge_id'] + [c for c in df.columns if c not in ['date', 'gauge_id']]
            df = df[cols]

            logger.debug(f"Loaded forcing data from {file_path} with {len(df)} records")
            return df

        except Exception as e:
            logger.warning(f"Failed to read forcing file {file_path}: {e}")
            return None

    def _merge_all_data(self, streamflow_df: pd.DataFrame,
                       forcing_df: pd.DataFrame,
                       et_df: Optional[pd.DataFrame],
                       smap_df: Optional[pd.DataFrame],
                       gauge_id: str) -> Optional[pd.DataFrame]:
        """Merge all data sources for a gauge."""
        if streamflow_df is None or streamflow_df.empty:
            return None

        merged_df = streamflow_df.copy()

        if forcing_df is not None and not forcing_df.empty:
            forcing_cols_to_drop = [col for col in ['gauge_id'] if col in forcing_df.columns]
            merged_df = pd.merge(
                merged_df,
                forcing_df.drop(columns=forcing_cols_to_drop),
                on='date',
                how='inner'
            )

        if et_df is not None and not et_df.empty:
            et_cols_to_drop = [col for col in ['gauge_id'] if col in et_df.columns]
            merged_df = pd.merge(
                merged_df,
                et_df.drop(columns=et_cols_to_drop),
                on='date',
                how='left'
            )

        if smap_df is not None and not smap_df.empty:
            smap_cols_to_drop = [col for col in ['gauge_id'] if col in smap_df.columns]
            merged_df = pd.merge(
                merged_df,
                smap_df.drop(columns=smap_cols_to_drop),
                on='date',
                how='left'
            )

        merged_df['gauge_id'] = str(gauge_id).zfill(8)

        return merged_df

    def _create_and_save_dataset(self, data_df: pd.DataFrame,
                                gauge_attrs: Dict[str, Any],
                                gauge_id: str) -> bool:
        """Create and save dataset for a gauge."""
        try:
            gauge_id_8 = str(gauge_id).zfill(8)

            self.config.output_dir.mkdir(parents=True, exist_ok=True)

            dataset = xr.Dataset.from_dataframe(data_df.set_index('date'))

            gauge_attrs['gauge_id'] = gauge_id_8
            dataset.attrs.update(gauge_attrs)
            dataset.attrs['creation_date'] = datetime.now().isoformat()

            output_format = self.config.processing_config.output_format

            if output_format == "netcdf":
                output_file = self.config.output_dir / f"gauge_{gauge_id_8}.nc"
                dataset.to_netcdf(output_file)
            elif output_format == "hdf5":
                output_file = self.config.output_dir / f"gauge_{gauge_id_8}.h5"
                dataset.to_netcdf(output_file)
            else:
                output_file = self.config.output_dir / f"gauge_{gauge_id_8}.parquet"
                data_df.to_parquet(output_file)

            logger.debug(f"Dataset saved: {output_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save dataset for gauge {gauge_id}: {e}")
            return False

    def _generate_summary(self, attributes_df: pd.DataFrame):
        """Generate processing summary."""
        summary = {
            'processing_date': datetime.now().isoformat(),
            'config': {
                'data_root': str(self.config.data_root),
                'output_dir': str(self.config.output_dir),
                'start_date': self.config.processing_config.start_date,
                'end_date': self.config.processing_config.end_date,
                'max_basins': self.config.max_basins,
                'output_format': self.config.processing_config.output_format,
                'overwrite_existing': self.config.processing_config.overwrite_existing
            },
            'statistics': {
                'total_gauges_available': len(attributes_df),
                'gauges_processed': len(self.processed_gauges),
                'gauges_failed': len(self.failed_gauges),
                'gauges_skipped': len(self.skipped_gauges),
                'success_rate': len(self.processed_gauges) / len(attributes_df)
                            if len(attributes_df) > 0 else 0
            },
            'processed_gauges': self.processed_gauges,
            'failed_gauges': self.failed_gauges,
            'skipped_gauges': self.skipped_gauges
        }

        summary_file = self.config.output_dir / "processing_summary.json"

        try:
            save_json(summary, summary_file)
            logger.info(f"Processing summary saved to {summary_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save processing summary: {e}")
            return False

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
                "basin_dataset_public_v1p2" / "usgs_streamflow"
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