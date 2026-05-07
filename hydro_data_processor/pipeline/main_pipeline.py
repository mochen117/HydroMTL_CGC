"""
Main pipeline orchestration for integrating multi-source hydrological data.
Ensures strict physical state preservation (no unauthorized scaling) and robust NaN handling.
"""

import sys
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import json
from datetime import datetime

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
    REQUIRED_STATIC = [
        'elev_mean', 'slope_mean', 'area_gages2',
        'frac_forest', 'lai_max', 'lai_diff', 'dom_land_cover_frac',
        'root_depth_50', 'soil_depth_statgso', 'soil_porosity',
        'soil_conductivity', 'max_water_content',
        'geol_porosity', 'geol_permeability',
        'sand_frac', 'clay_frac', 'organic_frac', 'carbonate_rocks_frac',
        'p_mean', 'pet_mean', 'aridity', 'p_seasonality', 'frac_snow'
    ]
    
    REQUIRED_CATEGORICAL_STATIC = ['dom_land_cover', 'geol_class_1st', 'geol_class_2nd']

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
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root directory not found: {self.data_root}")

        def resolve_path(cfg):
            if cfg and not cfg.data_source_path.is_absolute():
                cfg.data_source_path = self.data_root / cfg.data_source_path
            return cfg

        attr_cfg = resolve_path(self.config.data_sources.get("attributes"))
        self.attribute_loader = AttributeLoader(attr_cfg) if attr_cfg else None

        camels_sf_cfg = resolve_path(self.config.data_sources.get("camels_streamflow"))
        if camels_sf_cfg:
            self.camels_streamflow_loader = StreamflowLoader(camels_sf_cfg)
            self.camels_streamflow_loader.data_source_type = "camels"
        else:
            self.camels_streamflow_loader = None

        usgs_sf_cfg = resolve_path(self.config.data_sources.get("usgs_streamflow"))
        if usgs_sf_cfg:
            self.usgs_streamflow_loader = StreamflowLoader(usgs_sf_cfg)
            self.usgs_streamflow_loader.data_source_type = "usgs"
        else:
            self.usgs_streamflow_loader = None

        forcing_cfg = resolve_path(self.config.data_sources.get("nldas_forcing"))
        self.forcing_loader = ForcingLoader(forcing_cfg) if forcing_cfg else None

        et_cfg = resolve_path(self.config.data_sources.get("et_data"))
        self.et_loader = ETLoader(et_cfg) if et_cfg else None

        smap_cfg = resolve_path(self.config.data_sources.get("smap_data"))
        self.smap_loader = SMAPLoader(smap_cfg) if smap_cfg else None

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
        gage_ids = attributes_df['gage_id'].tolist()
        valid_gage_ids = []

        for i, gage_id in enumerate(gage_ids, 1):
            try:
                gage_attrs = self._get_gage_attributes(gage_id, attributes_df)
                huc2 = self._get_huc2_for_gage(gage_id, gage_attrs)
                if not huc2: continue

                streamflow_df = self._load_streamflow_with_huc2(gage_id, huc2)
                if streamflow_df is None or streamflow_df.empty: continue

                coverage = streamflow_df['streamflow'].notna().mean()
                if coverage >= min_coverage:
                    valid_gage_ids.append(gage_id)
            except Exception:
                continue

        return attributes_df[attributes_df['gage_id'].isin(valid_gage_ids)].copy()

    def run(self):
        if not self.attribute_loader: return

        attributes_df = self.attribute_loader.load(max_basins=None, skip_validation=True)
        if attributes_df.empty: return

        attributes_df = self._filter_basins_by_streamflow_coverage(attributes_df)

        if self.config.max_basins is not None and self.config.max_basins < len(attributes_df):
            attributes_df = attributes_df.head(self.config.max_basins)

        attributes_df['gage_id'] = attributes_df['gage_id'].astype(str).str.zfill(8)
        
        # REMOVED: Redundant categorical coding that broke the downstream Embedding schema.

        gage_ids = attributes_df['gage_id'].tolist()
        multi_source_processor = MultiSourceProcessor(self.config)
        batch_processor = BatchProcessor(self, multi_source_processor)
        self._process_gages(gage_ids, attributes_df, batch_processor)
        self._generate_summary()

    def _process_gages(self, gage_ids: List[str], attributes_df: pd.DataFrame, batch_processor):
        min_coverage = self.config.processing_config.min_streamflow_coverage

        # Configured tqdm for single-line robust updating
        pbar = tqdm(
            gage_ids, 
            desc="Processing Gages", 
            unit="basin", 
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
            mininterval=0.5
        )

        for idx, gage_id in enumerate(pbar):
            try:
                pbar.set_postfix({'Current Gage': gage_id})
                
                gage_attrs = self._get_gage_attributes(gage_id, attributes_df)
                static_check = self._check_static_completeness(gage_attrs, gage_id)
                self.static_compliance_results.append(static_check)

                huc2 = self._get_huc2_for_gage(gage_id, gage_attrs)
                if not huc2:
                    self.failed_gages.append({'gage_id': gage_id, 'reason': 'No HUC2'})
                    continue

                streamflow_data = self._load_streamflow_with_huc2(gage_id, huc2)
                forcing_data = self._load_forcing_with_huc2(gage_id, huc2)

                if streamflow_data is None or forcing_data is None:
                    self.failed_gages.append({'gage_id': gage_id, 'reason': 'Missing data'})
                    continue

                # REMOVED: Premature area division logic.
                # Physical unit constraints must remain intact for downstream Scaler.

                et_data = None
                if self.et_loader:
                    et_data = self.et_loader.load(
                        [gage_id], huc2=huc2, align_to_study_period=True,
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
                else:
                    self.failed_gages.append({'gage_id': gage_id, 'reason': 'Save failed'})

            except Exception as e:
                self.failed_gages.append({'gage_id': gage_id, 'reason': str(e)})

    def _merge_data(self, streamflow_df, forcing_df, et_df, smap_df) -> Optional[pd.DataFrame]:
        start = pd.Timestamp(self.config.processing_config.start_date)
        end = pd.Timestamp(self.config.processing_config.end_date)
        base = pd.DataFrame({'date': pd.date_range(start, end, freq='D')})

        sf = streamflow_df[['date', 'streamflow']].copy()
        sf.loc[sf['streamflow'] < -900.0, 'streamflow'] = np.nan
        merged = pd.merge(base, sf, on='date', how='left')

        if forcing_df is not None:
            if 'time' in forcing_df.columns: forcing_df = forcing_df.rename(columns={'time': 'date'})
            for col in self.REQUIRED_DYNAMIC[:5]:
                if col in forcing_df.columns:
                    merged = pd.merge(merged, forcing_df[['date', col]], on='date', how='left')

        if et_df is not None:
            if 'time' in et_df.columns: et_df = et_df.rename(columns={'time': 'date'})
            if 'evapotranspiration' in et_df.columns:
                merged = pd.merge(merged, et_df[['date', 'evapotranspiration']], on='date', how='left')

        if smap_df is not None:
            if 'time' in smap_df.columns: smap_df = smap_df.rename(columns={'time': 'date'})
            if 'ssm' in smap_df.columns:
                merged = pd.merge(merged, smap_df[['date', 'ssm']], on='date', how='left')

        return merged

    def _calculate_dynamic_coverage(self, df: pd.DataFrame) -> Dict[str, float]:
        coverage = {}
        for var in self.REQUIRED_DYNAMIC:
            coverage[var] = df[var].notna().mean() if var in df.columns else 0.0
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

            for var_name, value in gage_attrs.items():
                if var_name in ['gage_id', 'huc_02'] or var_name in ds.coords or var_name in ds.data_vars:
                    continue
                
                if var_name in self.REQUIRED_CATEGORICAL_STATIC:
                    try: val = int(value) if not pd.isna(value) else 0
                    except (TypeError, ValueError): val = 0
                    ds[var_name] = val
                else:
                    if pd.isna(value): ds[var_name] = np.nan
                    else:
                        try: ds[var_name] = float(value)
                        except (TypeError, ValueError): pass

            # PROTECTED: Categorical variables are shielded from float32 coercion
            for var_name in list(ds.variables):
                if var_name == 'time' or var_name in self.REQUIRED_CATEGORICAL_STATIC: 
                    continue
                if ds[var_name].dtype.kind in 'iu':
                    ds[var_name] = ds[var_name].astype('float32')

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

            for var in self.REQUIRED_DYNAMIC:
                if var in ds: ds[var].attrs['missing_value'] = np.nan

            encoding = {}
            if 'time' in ds.coords:
                encoding['time'] = {'dtype': 'int64', 'units': 'days since 2001-01-01', 'calendar': 'proleptic_gregorian'}
            
            # FIXED: Differentiate _FillValue assignment based on data type to prevent NaN to Int cast error
            for var in ds.data_vars:
                if var not in encoding:
                    # 'f' is float, 'c' is complex
                    if ds[var].dtype.kind in 'fc':
                        encoding[var] = {'zlib': True, '_FillValue': np.nan}
                    # 'i' is signed integer, 'u' is unsigned integer
                    elif ds[var].dtype.kind in 'iu':
                        encoding[var] = {'zlib': True, '_FillValue': 0}

            output_file = self.config.output_dir / f"gage_{gage_id_8}.nc"
            ds.to_netcdf(output_file, encoding=encoding)
            return True
        except Exception as e:
            # We use tqdm.write to prevent the error message from breaking the progress bar layout
            tqdm.write(f"ERROR: Save failed for {gage_id}: {e}")
            logger.error(f"Save failed for {gage_id}: {e}")
            return False

    def _generate_summary(self):
        total = len(self.static_compliance_results)
        complete_static = sum(1 for r in self.static_compliance_results if r['complete']) if total > 0 else 0
        partial_static = total - complete_static if total > 0 else 0

        dynamic_summary = {}
        if self.dynamic_coverage_results:
            for var in self.REQUIRED_DYNAMIC:
                covs = [r.get(var, 0.0) for r in self.dynamic_coverage_results if var in r]
                if covs:
                    dynamic_summary[var] = {
                        'mean_coverage': np.mean(covs), 'min_coverage': np.min(covs),
                        'max_coverage': np.max(covs), 'basins_above_95': sum(1 for c in covs if c >= 0.95)
                    }

        summary = {
            'processing_date': datetime.now().isoformat(),
            'config': {
                'data_root': str(self.config.data_root), 'output_dir': str(self.config.output_dir),
                'start_date': self.config.processing_config.start_date, 'end_date': self.config.processing_config.end_date,
                'max_basins': self.config.max_basins, 'min_coverage': self.config.processing_config.min_streamflow_coverage,
            },
            'static_attribute_completeness': {
                'total_basins_analyzed': total, 'fully_complete': complete_static,
                'partially_complete': partial_static, 'completeness_rate': complete_static / total if total > 0 else 0,
            },
            'dynamic_coverage_summary': dynamic_summary,
            'processing_stats': {
                'total_gages_analyzed': len(self.processed_gages) + len(self.failed_gages) + len(self.skipped_gages),
                'processed_gages': len(self.processed_gages), 'failed_gages': len(self.failed_gages),
                'skipped_gages': len(self.skipped_gages),
                'success_rate': len(self.processed_gages) / max(1, len(self.processed_gages) + len(self.failed_gages) + len(self.skipped_gages)),
            },
            'valid_gages': self.valid_gages, 'failed_gages': self.failed_gages, 'skipped_gages': self.skipped_gages
        }

        summary_file = self.config.output_dir / "final_processing_summary.json"
        with open(summary_file, 'w') as f: json.dump(summary, f, indent=2, default=str)
        
        valid_file = self.config.output_dir / "valid_basins.txt"
        with open(valid_file, 'w') as f:
            for g in self.valid_gages: f.write(f"{g['gage_id']}\n")

    def _get_gage_attributes(self, gage_id: str, attributes_df: pd.DataFrame) -> Dict:
        gage_id_8 = gage_id.zfill(8)
        row = attributes_df[attributes_df['gage_id'].astype(str) == gage_id_8]
        if row.empty: row = attributes_df[attributes_df['gage_id'].astype(str) == gage_id]
        if row.empty: return {}
        attrs = row.iloc[0].to_dict()
        clean = {}
        for k, v in attrs.items():
            if isinstance(v, (np.integer, np.int64)): clean[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)): clean[k] = float(v) if not pd.isna(v) else np.nan
            elif isinstance(v, np.ndarray): clean[k] = v.tolist()
            else: clean[k] = v
        return clean

    def _get_huc2_for_gage(self, gage_id: str, gage_attrs: Dict) -> Optional[str]:
        gage_id_8 = gage_id.zfill(8)
        if gage_id_8 in self.huc2_mapping: return str(self.huc2_mapping[gage_id_8]).zfill(2)
        if 'huc_02' in gage_attrs and gage_attrs['huc_02']:
            try: return str(int(float(gage_attrs['huc_02']))).zfill(2)
            except (ValueError, TypeError): return str(gage_attrs['huc_02']).zfill(2)
        return None

    def _load_streamflow_with_huc2(self, gage_id: str, huc2: str) -> Optional[pd.DataFrame]:
        try:
            camels, usgs = None, None
            start_date = self.config.processing_config.start_date
            end_date = self.config.processing_config.end_date
            
            if self.camels_streamflow_loader:
                camels = self.camels_streamflow_loader.load([gage_id], huc2=huc2)
                if camels is not None and not camels.empty:
                    camels = camels[(camels['date'] >= '2001-01-01') & (camels['date'] <= '2014-12-31')]
            if self.usgs_streamflow_loader:
                usgs = self.usgs_streamflow_loader.load([gage_id], huc2=huc2)
                if usgs is not None and not usgs.empty:
                    usgs = usgs[(usgs['date'] >= '2015-01-01') & (usgs['date'] <= '2021-09-30')]
            
            combined = []
            if camels is not None and not camels.empty: combined.append(camels)
            if usgs is not None and not usgs.empty: combined.append(usgs)
            if not combined: return None
                
            df = pd.concat(combined, ignore_index=True).sort_values('date').drop_duplicates('date')
            full = pd.DataFrame({'date': pd.date_range(start_date, end_date, freq='D')})
            return pd.merge(full, df[['date', 'streamflow']], on='date', how='left')
        except Exception:
            return None

    def _load_forcing_with_huc2(self, gage_id: str, huc2: str) -> Optional[pd.DataFrame]:
        if not self.forcing_loader: return None
        try:
            data = self.forcing_loader.load([gage_id], huc2=huc2)
            if data is not None and not data.empty: return data
        except: pass
        
        forcing_config = self.config.data_sources.get("nldas_forcing")
        if not forcing_config: return None
        file_path = forcing_config.data_source_path / "basin_mean_forcing" / str(huc2).zfill(2) / f"{gage_id}_lump_nldas_forcing_leap.txt"
        if not file_path.exists(): return None
        
        try:
            df = pd.read_csv(file_path, sep=r'\s+', header=0)
            col_map = {
                'Year': 'year', 'Mnth': 'month', 'Day': 'day',
                'temperature(C)': 'temperature', 'specific_humidity(kg/kg)': 'specific_humidity',
                'shortwave_radiation(W/m^2)': 'shortwave_radiation', 'potential_energy(J/kg)': 'potential_energy',
                'total_precipitation(kg/m^2)': 'total_precipitation'
            }
            df = df.rename(columns=col_map)
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
            return df[['date', 'temperature', 'specific_humidity', 'shortwave_radiation', 'potential_energy', 'total_precipitation']]
        except Exception:
            return None