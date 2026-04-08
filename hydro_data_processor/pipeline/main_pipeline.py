"""
Main pipeline for Hydro Data Processing.
Includes static attribute validation, coverage checks, and summary.
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

from tqdm import tqdm
from hydro_data_processor.config.settings import ProjectConfig
from hydro_data_processor.loaders.attribute_loader import AttributeLoader
from hydro_data_processor.loaders.streamflow_loader import StreamflowLoader
from hydro_data_processor.loaders.forcing_loader import ForcingLoader
from hydro_data_processor.loaders.et_loader import ETLoader
from hydro_data_processor.loaders.smap_loader import SMAPLoader
from hydro_data_processor.processors.multi_source_processor import MultiSourceProcessor
from hydro_data_processor.processors.batch_processor import BatchProcessor

logger = logging.getLogger(__name__)

__all__ = ['HydroDataPipeline']


class HydroDataPipeline:
    """Main pipeline for processing hydrological data with quality checks."""
    
    REQUIRED_STATIC = [
        'elev_mean', 'slope_mean', 'area_gages2',
        'frac_forest', 'lai_max', 'lai_diff', 'dom_land_cover_frac',
        'root_depth_50', 'soil_depth_statgso', 'soil_porosity',
        'soil_conductivity', 'max_water_content',
        'geol_porosity', 'geol_permeability'
        # 'dom_land_cover', 'geol_class_1st', 'geol_class_2nd'  # 已删除
    ]

    REQUIRED_DYNAMIC = [
        'total_precipitation', 'temperature', 'specific_humidity',
        'shortwave_radiation', 'potential_energy', 'streamflow',
        'evapotranspiration', 'ssm'
    ]

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.data_root = config.data_root.resolve()
        self._initialize_loaders()
        self.processed_gages: List[str] = []
        self.failed_gages: List[Dict] = []
        self.skipped_gages: List[Dict] = []
        self.valid_gages: List[Dict] = []
        self.huc2_mapping: Dict[str, str] = {}
        self.static_compliance_results: List[Dict] = []
        self.dynamic_coverage_results: List[Dict] = []

    def _initialize_loaders(self):
        logger.debug("Initializing data loaders")
        logger.info(f"Data root resolved to: {self.data_root}")

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root directory not found: {self.data_root}")

        attribute_config = self.config.data_sources.get("attributes")
        if attribute_config:
            if not attribute_config.data_source_path.is_absolute():
                attribute_config.data_source_path = self.data_root / attribute_config.data_source_path
            self.attribute_loader = AttributeLoader(attribute_config)
        else:
            self.attribute_loader = None

        camels_streamflow_config = self.config.data_sources.get("camels_streamflow")
        if camels_streamflow_config:
            if not camels_streamflow_config.data_source_path.is_absolute():
                camels_streamflow_config.data_source_path = self.data_root / camels_streamflow_config.data_source_path
            self.camels_streamflow_loader = StreamflowLoader(camels_streamflow_config)
            self.camels_streamflow_loader.data_source_type = "camels"
        else:
            self.camels_streamflow_loader = None

        usgs_streamflow_config = self.config.data_sources.get("usgs_streamflow")
        if usgs_streamflow_config:
            if not usgs_streamflow_config.data_source_path.is_absolute():
                usgs_streamflow_config.data_source_path = self.data_root / usgs_streamflow_config.data_source_path
            self.usgs_streamflow_loader = StreamflowLoader(usgs_streamflow_config)
            self.usgs_streamflow_loader.data_source_type = "usgs"
        else:
            self.usgs_streamflow_loader = None

        forcing_config = self.config.data_sources.get("nldas_forcing")
        if forcing_config:
            if not forcing_config.data_source_path.is_absolute():
                forcing_config.data_source_path = self.data_root / forcing_config.data_source_path
            self.forcing_loader = ForcingLoader(forcing_config)
        else:
            self.forcing_loader = None

        et_config = self.config.data_sources.get("et_data")
        if et_config:
            if not et_config.data_source_path.is_absolute():
                et_config.data_source_path = self.data_root / et_config.data_source_path
            self.et_loader = ETLoader(et_config)
        else:
            self.et_loader = None

        smap_config = self.config.data_sources.get("smap_data")
        if smap_config:
            if not smap_config.data_source_path.is_absolute():
                smap_config.data_source_path = self.data_root / smap_config.data_source_path
            self.smap_loader = SMAPLoader(smap_config)
        else:
            self.smap_loader = None

        logger.info("All loaders initialized")

    def _check_static_completeness(self, gage_attrs: Dict[str, Any], gage_id: str) -> Dict[str, Any]:
        present = []
        missing = []
        for var in self.REQUIRED_STATIC:
            if var in gage_attrs and not pd.isna(gage_attrs[var]):
                present.append(var)
            else:
                missing.append(var)
        return {
            'gage_id': gage_id,
            'present_count': len(present),
            'missing_count': len(missing),
            'missing_vars': missing,
            'complete': len(missing) == 0
        }

    def _filter_basins_by_streamflow_coverage(self, attributes_df: pd.DataFrame) -> pd.DataFrame:
        min_coverage = self.config.processing_config.min_streamflow_coverage
        logger.info(f"Filtering basins by streamflow coverage >= {min_coverage:.0%}...")

        gage_ids = attributes_df['gage_id'].tolist()
        valid_gage_ids = []

        for i, gage_id in enumerate(gage_ids, 1):
            if i % 50 == 0:
                logger.debug(f"Processed {i}/{len(gage_ids)} basins for coverage check")

            try:
                gage_attrs = self._get_gage_attributes(gage_id, attributes_df)
                huc2 = self._get_huc2_for_gage(gage_id, gage_attrs)
                if not huc2:
                    continue

                streamflow_df = self._load_streamflow_with_huc2(gage_id, huc2)
                if streamflow_df is None or streamflow_df.empty:
                    continue

                coverage = streamflow_df['streamflow'].notna().mean()
                if coverage >= min_coverage:
                    valid_gage_ids.append(gage_id)
            except Exception as e:
                logger.debug(f"Error checking coverage for {gage_id}: {e}")
                continue

        logger.info(f"Coverage check completed: {len(valid_gage_ids)}/{len(gage_ids)} basins passed")
        filtered_df = attributes_df[attributes_df['gage_id'].isin(valid_gage_ids)].copy()
        logger.info(f"Filtered attributes to {len(filtered_df)} basins")
        return filtered_df

    def run(self):
        logger.info("=" * 60)
        logger.info("Hydro Data Processing Pipeline")
        logger.info("=" * 60)
        logger.info(f"Data root: {self.data_root}")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info(f"Study period: {self.config.processing_config.start_date} to {self.config.processing_config.end_date}")
        logger.info(f"Min coverage: {self.config.processing_config.min_streamflow_coverage:.0%}")

        if not self.attribute_loader:
            logger.error("Attribute loader not available")
            return

        logger.info("Step 1: Loading all basin attributes")
        attributes_df = self.attribute_loader.load(max_basins=None, skip_validation=True)
        if attributes_df.empty:
            logger.error("No attributes loaded")
            return

        attributes_df = self._filter_basins_by_streamflow_coverage(attributes_df)

        if self.config.max_basins is not None and self.config.max_basins < len(attributes_df):
            logger.info(f"Limiting to {self.config.max_basins} basins")
            attributes_df = attributes_df.head(self.config.max_basins)

        attributes_df['gage_id'] = attributes_df['gage_id'].astype(str).str.zfill(8)
        gage_ids = attributes_df['gage_id'].tolist()
        logger.info(f"Step 2: Processing {len(gage_ids)} gages")

        multi_source_processor = MultiSourceProcessor(self.config)
        batch_processor = BatchProcessor(self, multi_source_processor)
        self._process_gages(gage_ids, attributes_df, batch_processor)

        logger.info("Step 3: Generating final summary")
        self._generate_summary()

    def _process_gages(self, gage_ids: List[str], attributes_df: pd.DataFrame, batch_processor):
        min_coverage = self.config.processing_config.min_streamflow_coverage
        total = len(gage_ids)
        logger.info(f"Processing {total} gages")

        for idx, gage_id in enumerate(tqdm(gage_ids, desc="Gages", unit="basin")):
            try:
                gage_attrs = self._get_gage_attributes(gage_id, attributes_df)
                static_check = self._check_static_completeness(gage_attrs, gage_id)
                self.static_compliance_results.append(static_check)

                if not static_check['complete']:
                    logger.debug(f"Gage {gage_id} missing static attributes: {static_check['missing_vars']}")

                huc2 = self._get_huc2_for_gage(gage_id, gage_attrs)
                if not huc2:
                    self.failed_gages.append({'gage_id': gage_id, 'reason': 'No HUC2'})
                    continue

                streamflow_data = self._load_streamflow_with_huc2(gage_id, huc2)
                forcing_data = self._load_forcing_with_huc2(gage_id, huc2)

                if streamflow_data is None or forcing_data is None:
                    self.failed_gages.append({'gage_id': gage_id, 'reason': 'Missing data'})
                    continue

                et_data = None
                if self.et_loader:
                    et_data = self.et_loader.load(
                        [gage_id], huc2=huc2,
                        align_to_study_period=True,
                        start_date=self.config.processing_config.start_date,
                        end_date=self.config.processing_config.end_date
                    )
  
                smap_data = None
                if self.smap_loader:
                    smap_data = self.smap_loader.load([gage_id], huc2=huc2)

                merged_data = self._merge_data(streamflow_data, forcing_data, et_data, smap_data)
                if merged_data is None or merged_data.empty:
                    self.failed_gages.append({'gage_id': gage_id, 'reason': 'Merge failed'})
                    continue

                coverage = self._calculate_dynamic_coverage(merged_data)
                self.dynamic_coverage_results.append({'gage_id': gage_id, **coverage})
                streamflow_cov = coverage.get('streamflow', 0.0)

                if streamflow_cov < min_coverage:
                    self.skipped_gages.append({'gage_id': gage_id, 'reason': f'Coverage {streamflow_cov:.2%}'})
                    continue

                success = self._save_dataset(merged_data, gage_attrs, gage_id, static_check, coverage)
                if success:
                    self.valid_gages.append({'gage_id': gage_id, 'coverage': streamflow_cov, 'huc2': huc2})
                    self.processed_gages.append(gage_id)
                    logger.debug(f"Gage {gage_id} processed successfully")
                else:
                    self.failed_gages.append({'gage_id': gage_id, 'reason': 'Save failed'})

            except Exception as e:
                logger.error(f"Error processing gage {gage_id}: {e}")
                self.failed_gages.append({'gage_id': gage_id, 'reason': str(e)})

        logger.info(f"Processing completed. Valid: {len(self.valid_gages)}, Failed: {len(self.failed_gages)}, Skipped: {len(self.skipped_gages)}")

    def _merge_data(self, streamflow_df, forcing_df, et_df, smap_df) -> Optional[pd.DataFrame]:
        start = pd.Timestamp(self.config.processing_config.start_date)
        end = pd.Timestamp(self.config.processing_config.end_date)
        base = pd.DataFrame({'date': pd.date_range(start, end, freq='D')})

        sf = streamflow_df[['date', 'streamflow']].copy()
        sf['streamflow'] = sf['streamflow'].replace(-999.0, np.nan)
        merged = pd.merge(base, sf, on='date', how='left')

        if forcing_df is not None:
            if 'time' in forcing_df.columns:
                forcing_df = forcing_df.rename(columns={'time': 'date'})
            for col in self.REQUIRED_DYNAMIC[:5]:
                if col in forcing_df.columns:
                    merged = pd.merge(merged, forcing_df[['date', col]], on='date', how='left')

        if et_df is not None:
            if 'time' in et_df.columns:
                et_df = et_df.rename(columns={'time': 'date'})
            if 'evapotranspiration' in et_df.columns:
                merged = pd.merge(merged, et_df[['date', 'evapotranspiration']], on='date', how='left')

        if smap_df is not None:
            if 'time' in smap_df.columns:
                smap_df = smap_df.rename(columns={'time': 'date'})
            if 'ssm' in smap_df.columns:
                merged = pd.merge(merged, smap_df[['date', 'ssm']], on='date', how='left')

        return merged

    def _calculate_dynamic_coverage(self, df: pd.DataFrame) -> Dict[str, float]:
        coverage = {}
        for var in self.REQUIRED_DYNAMIC:
            if var in df.columns:
                coverage[var] = df[var].notna().mean()
            else:
                coverage[var] = 0.0
        return coverage

    def _save_dataset(self, data_df: pd.DataFrame, gage_attrs: Dict, gage_id: str,
                    static_check: Dict, coverage: Dict) -> bool:
        try:
            gage_id_8 = str(gage_id).zfill(8)
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            if 'date' in data_df.columns:
                data_df = data_df.rename(columns={'date': 'time'})
            data_df = data_df.set_index('time')
            ds = data_df.to_xarray()

            # Add static attributes as data variables (skip gage_id and huc_02, they go to global attrs)
            for var_name, value in gage_attrs.items():
                # Skip gage_id and huc_02 (they will be added as global attributes)
                if var_name in ['gage_id', 'huc_02']:
                    continue
                # Skip if already present in dataset (e.g., time coordinate)
                if var_name in ds.coords or var_name in ds.data_vars:
                    continue
                # Skip NaN values
                if pd.isna(value):
                    continue
                # Convert to float if possible
                try:
                    val = float(value)
                    ds[var_name] = val
                except (TypeError, ValueError):
                    logger.debug(f"Cannot convert {var_name}={value} to float, skipping")

            # Add global attributes (including gage_id and huc_02)
            ds.attrs.update({
                'title': 'Hydro-Meteorological Dataset',
                'gage_id': gage_id_8,
                'huc_02': str(gage_attrs.get('huc_02', '')),
                'creation_date': datetime.now().isoformat(),
                'study_period': f"{self.config.processing_config.start_date} to {self.config.processing_config.end_date}",
                'static_completeness': int(static_check['complete']),
                'static_missing_count': int(static_check['missing_count']),
                'streamflow_coverage': float(coverage.get('streamflow', 0.0))
            })

            # Add variable attributes for dynamic variables
            for var in self.REQUIRED_DYNAMIC:
                if var in ds:
                    ds[var].attrs['missing_value'] = -999.0

            # Encoding
            encoding = {}
            if 'time' in ds.coords:
                encoding['time'] = {
                    'dtype': 'int64',
                    'units': 'days since 2001-01-01',
                    'calendar': 'proleptic_gregorian'
                }
            for var in ds.data_vars:
                if var not in encoding and ds[var].dtype.kind in 'iufc':
                    encoding[var] = {'zlib': True, '_FillValue': np.nan}

            output_file = self.config.output_dir / f"gage_{gage_id_8}.nc"
            ds.to_netcdf(output_file, encoding=encoding)
            logger.debug(f"Saved: {output_file}")
            return True
        except Exception as e:
            logger.error(f"Save failed for {gage_id}: {e}")
            return False

    def _generate_summary(self):
        total = len(self.static_compliance_results)
        if total > 0:
            complete_static = sum(1 for r in self.static_compliance_results if r['complete'])
            partial_static = total - complete_static
        else:
            complete_static = 0
            partial_static = 0

        dynamic_summary = {}
        if self.dynamic_coverage_results:
            for var in self.REQUIRED_DYNAMIC:
                covs = [r.get(var, 0.0) for r in self.dynamic_coverage_results if var in r]
                if covs:
                    dynamic_summary[var] = {
                        'mean_coverage': np.mean(covs),
                        'min_coverage': np.min(covs),
                        'max_coverage': np.max(covs),
                        'basins_above_95': sum(1 for c in covs if c >= 0.95)
                    }

        summary = {
            'processing_date': datetime.now().isoformat(),
            'config': {
                'data_root': str(self.config.data_root),
                'output_dir': str(self.config.output_dir),
                'start_date': self.config.processing_config.start_date,
                'end_date': self.config.processing_config.end_date,
                'max_basins': self.config.max_basins,
                'min_coverage': self.config.processing_config.min_streamflow_coverage,
            },
            'static_attribute_completeness': {
                'total_basins_analyzed': total,
                'fully_complete': complete_static,
                'partially_complete': partial_static,
                'completeness_rate': complete_static / total if total > 0 else 0,
            },
            'dynamic_coverage_summary': dynamic_summary,
            'processing_stats': {
                'total_gages_analyzed': len(self.processed_gages) + len(self.failed_gages) + len(self.skipped_gages),
                'processed_gages': len(self.processed_gages),
                'failed_gages': len(self.failed_gages),
                'skipped_gages': len(self.skipped_gages),
                'success_rate': len(self.processed_gages) / (len(self.processed_gages) + len(self.failed_gages) + len(self.skipped_gages)) if (len(self.processed_gages) + len(self.failed_gages) + len(self.skipped_gages)) > 0 else 0,
            },
            'valid_gages': self.valid_gages,
            'failed_gages': self.failed_gages,
            'skipped_gages': self.skipped_gages
        }

        summary_file = self.config.output_dir / "final_processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary saved to {summary_file}")

        valid_file = self.config.output_dir / "valid_basins.txt"
        with open(valid_file, 'w') as f:
            for g in self.valid_gages:
                f.write(f"{g['gage_id']}\n")
        logger.info(f"Valid basins list saved to {valid_file}")

        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total basins analyzed: {summary['static_attribute_completeness']['total_basins_analyzed']}")
        logger.info(f"Basins with complete static attributes: {complete_static} ({complete_static/total*100:.1f}%)")
        logger.info(f"Basins meeting coverage >= {self.config.processing_config.min_streamflow_coverage:.0%}: {len(self.valid_gages)}")
        logger.info(f"Successfully processed: {len(self.processed_gages)}")
        logger.info(f"Success rate: {summary['processing_stats']['success_rate']:.1%}")
        logger.info("=" * 60)

    # ---------- Helper methods ----------
    def _get_gage_attributes(self, gage_id: str, attributes_df: pd.DataFrame) -> Dict:
        gage_id_8 = gage_id.zfill(8)
        row = attributes_df[attributes_df['gage_id'].astype(str) == gage_id_8]
        if row.empty:
            row = attributes_df[attributes_df['gage_id'].astype(str) == gage_id]
        if row.empty:
            return {}
        attrs = row.iloc[0].to_dict()
        clean = {}
        for k, v in attrs.items():
            if isinstance(v, (np.integer, np.int64)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                clean[k] = float(v)
            elif isinstance(v, np.ndarray):
                clean[k] = v.tolist()
            elif pd.isna(v):
                continue
            else:
                clean[k] = v
        return clean

    def _get_huc2_for_gage(self, gage_id: str, gage_attrs: Dict) -> Optional[str]:
        gage_id_8 = gage_id.zfill(8)
        if gage_id_8 in self.huc2_mapping:
            return str(self.huc2_mapping[gage_id_8]).zfill(2)
        if 'huc_02' in gage_attrs and gage_attrs['huc_02']:
            return str(gage_attrs['huc_02']).zfill(2)
        return None

    def _load_streamflow_with_huc2(self, gage_id: str, huc2: str) -> Optional[pd.DataFrame]:
        try:
            camels = None
            usgs = None
            if self.camels_streamflow_loader:
                camels = self.camels_streamflow_loader.load([gage_id], huc2=huc2)
                if camels is not None and not camels.empty:
                    camels = camels[(camels['date'] >= '2001-01-01') & (camels['date'] <= '2014-12-31')]
            if self.usgs_streamflow_loader:
                usgs = self.usgs_streamflow_loader.load([gage_id], huc2=huc2)
                if usgs is not None and not usgs.empty:
                    usgs = usgs[(usgs['date'] >= '2015-01-01') & (usgs['date'] <= '2021-09-30')]
            combined = []
            if camels is not None and not camels.empty:
                combined.append(camels)
            if usgs is not None and not usgs.empty:
                combined.append(usgs)
            if not combined:
                return None
            df = pd.concat(combined, ignore_index=True).sort_values('date').drop_duplicates('date')
            full = pd.DataFrame({'date': pd.date_range('2001-01-01', '2021-09-30', freq='D')})
            merged = pd.merge(full, df[['date', 'streamflow']], on='date', how='left')
            logger.debug(f"Gage {gage_id}: loaded {len(df)} records, coverage {merged['streamflow'].notna().mean():.2%}")
            return merged
        except Exception as e:
            logger.warning(f"Error loading streamflow for {gage_id}: {e}")
            return None

    def _load_forcing_with_huc2(self, gage_id: str, huc2: str) -> Optional[pd.DataFrame]:
        if not self.forcing_loader:
            return None
        try:
            data = self.forcing_loader.load([gage_id], huc2=huc2)
            if data is not None and not data.empty:
                return data
        except:
            pass
        forcing_config = self.config.data_sources.get("nldas_forcing")
        if not forcing_config:
            return None
        h2 = str(huc2).zfill(2)
        file_path = forcing_config.data_source_path / "basin_mean_forcing" / h2 / f"{gage_id}_lump_nldas_forcing_leap.txt"
        if not file_path.exists():
            return None
        try:
            df = pd.read_csv(file_path, sep=r'\s+', header=0)
            col_map = {
                'Year': 'year', 'Mnth': 'month', 'Day': 'day', 'Hr': 'hour',
                'temperature(C)': 'temperature', 'specific_humidity(kg/kg)': 'specific_humidity',
                'shortwave_radiation(W/m^2)': 'shortwave_radiation',
                'potential_energy(J/kg)': 'potential_energy',
                'total_precipitation(kg/m^2)': 'total_precipitation'
            }
            df = df.rename(columns=col_map)
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
            df = df[['date', 'temperature', 'specific_humidity', 'shortwave_radiation', 'potential_energy', 'total_precipitation']]
            return df
        except Exception as e:
            logger.warning(f"Failed to read forcing file {file_path}: {e}")
            return None

    def explore_data_structure(self):
        logger.info("Exploring data structure...")
        camels_dir = self.data_root / "camels" / "camels_us"
        if camels_dir.exists():
            logger.info(f"Found CAMELS directory: {camels_dir}")
            txt_files = list(camels_dir.glob("camels_*.txt"))
            logger.info(f"Attribute files: {len(txt_files)}")
        else:
            logger.error(f"CAMELS directory not found: {camels_dir}")