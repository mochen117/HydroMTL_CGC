"""
Evapotranspiration data loader for MODIS16A2 v006.
Strictly follows paper method:
- Raw data unit is 0.1 mm per period (kg/m^2/8day * 0.1 = mm/period).
- Daily average = (raw_value * 0.1) / actual_period_length.
- Period length: from date difference (usually 8), last period of year: 5 (non-leap) or 6 (leap) days.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import numpy as np
import calendar

from .base_loader import BaseDataLoader
from hydro_data_processor.config.settings import DataSourceConfig

logger = logging.getLogger(__name__)


class ETLoader(BaseDataLoader):
    """Loader for evapotranspiration data with strict MOD16A2 period handling and unit conversion."""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config, "et")
        logger.info(f"ETLoader initialized for {config.data_source_path}")

    def load(self, gage_ids: List[str], **kwargs) -> pd.DataFrame:
        if not gage_ids:
            logger.warning("No gage IDs provided")
            return pd.DataFrame()

        all_data = []
        align_to_study_period = kwargs.get('align_to_study_period', True)
        study_start = kwargs.get('start_date', '2001-01-01')
        study_end = kwargs.get('end_date', '2021-09-30')

        for gage_id in gage_ids:
            try:
                raw_data = self._load_single_gage(gage_id, **kwargs)
                if raw_data is not None and not raw_data.empty:
                    daily_data = self._resample_8day_to_daily(raw_data, gage_id)
                    if align_to_study_period:
                        daily_data = self._align_to_study_period(daily_data, study_start, study_end)
                    all_data.append(daily_data)
                    logger.debug(f"Successfully processed ET for gage {gage_id}")
                else:
                    logger.debug(f"No ET data found for gage {gage_id}")
            except Exception as e:
                logger.error(f"Error loading ET data for gage {gage_id}: {e}")

        if not all_data:
            return pd.DataFrame()

        combined_df = pd.concat(all_data, ignore_index=True)
        self.data = combined_df

        return combined_df

    def _resample_8day_to_daily(self, df: pd.DataFrame, gage_id: str) -> pd.DataFrame:
        """
        Convert 8-day cumulative ET to daily averages.
        Steps:
        1. Convert raw value from 0.1 mm to mm: value_mm = raw_value * 0.1
        2. Determine period length (days):
           - For all but last record: days between consecutive dates (fallback to 8 if out of 5-9)
           - For last record: days from start date to end of year (5 for non-leap, 6 for leap)
        3. Daily average = value_mm / period_length
        """
        if df.empty:
            return df

        df = df.sort_values('date').reset_index(drop=True)
        daily_records = []
        n = len(df)

        for i in range(n):
            period_start = df.iloc[i]['date']
            raw_value = df.iloc[i].get('evapotranspiration')
            if pd.isna(raw_value):
                continue

            # Step 1: convert from 0.1mm to mm
            period_value_mm = raw_value * 0.1

            # Step 2: determine period length
            if i < n - 1:
                next_date = df.iloc[i + 1]['date']
                period_length = (next_date - period_start).days
                if period_length < 5 or period_length > 9:
                    period_length = 8
            else:
                year = period_start.year
                last_day = pd.Timestamp(year, 12, 31)
                period_length = (last_day - period_start).days + 1
                if period_length <= 0:
                    period_length = 8

            # Step 3: daily average
            daily_value = period_value_mm / period_length

            for day_offset in range(period_length):
                daily_date = period_start + pd.Timedelta(days=day_offset)
                daily_records.append({
                    'date': daily_date,
                    'gage_id': gage_id,
                    'evapotranspiration': daily_value
                })

        if not daily_records:
            logger.warning(f"Gage {gage_id}: No daily records created")
            return pd.DataFrame(columns=['date', 'gage_id', 'evapotranspiration'])

        daily_df = pd.DataFrame(daily_records)
        daily_df = daily_df.sort_values('date').reset_index(drop=True)

        # Fill missing dates to ensure continuous daily series
        min_date = daily_df['date'].min()
        max_date = daily_df['date'].max()
        complete_dates = pd.date_range(start=min_date, end=max_date, freq='D')
        complete_df = pd.DataFrame({'date': complete_dates})
        daily_df = pd.merge(complete_df, daily_df, on='date', how='left')
        daily_df['gage_id'] = gage_id

        # Linear interpolation for short gaps (up to 3 days)
        if 'evapotranspiration' in daily_df.columns:
            daily_df['evapotranspiration'] = daily_df['evapotranspiration'].interpolate(
                method='linear', limit=3, limit_direction='both'
            ).copy()

        return daily_df

    def _align_to_study_period(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Align to full study period with forward/backward fill for short gaps."""
        study_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        study_df = pd.DataFrame({'date': study_dates})

        if df.empty:
            result = study_df.copy()
            result['evapotranspiration'] = np.nan
            return result

        merged = pd.merge(study_df, df, on='date', how='left')

        if 'evapotranspiration' in merged.columns:
            merged['evapotranspiration'] = merged['evapotranspiration'].ffill(limit=8).bfill(limit=8)

        if 'gage_id' not in merged.columns and 'gage_id' in df.columns:
            merged['gage_id'] = df['gage_id'].iloc[0]

        return merged

    # ========== Helper methods (unchanged, kept for completeness) ==========

    def _load_single_gage(self, gage_id: str, **kwargs) -> Optional[pd.DataFrame]:
        huc2 = kwargs.get('huc2')
        if not huc2:
            huc2 = self._find_huc2_for_gage(gage_id)
            if not huc2:
                logger.debug(f"Cannot determine HUC2 for gage {gage_id}")
                return None
        file_path = self._build_file_path(gage_id, huc2)
        if not file_path or not file_path.exists():
            file_path = self._find_alternative_path(gage_id, huc2)
            if not file_path or not file_path.exists():
                logger.debug(f"ET file not found for gage {gage_id} in HUC2 {huc2}")
                return None
        try:
            return self._read_and_process_file(file_path, gage_id)
        except Exception as e:
            logger.error(f"Failed to process ET file {file_path}: {e}")
            return None

    def _build_file_path(self, gage_id: str, huc2: str) -> Optional[Path]:
        if not self.config.data_source_path.exists():
            logger.warning(f"Data source path does not exist: {self.config.data_source_path}")
            return None
        if hasattr(self.config, 'get_file_path'):
            return self.config.get_file_path(gage_id, huc2)
        path = self.config.data_source_path
        if self.config.subdirectory:
            subdir = self.config.subdirectory.format(huc2=huc2)
            path = path / subdir
        if self.config.file_pattern:
            filename = self.config.file_pattern.replace('{basin_id}', gage_id).replace('{gage_id}', gage_id)
            path = path / filename
        return path

    def _find_huc2_for_gage(self, gage_id: str) -> Optional[str]:
        if not self.config.data_source_path.exists():
            return None
        huc2_dirs = [d for d in self.config.data_source_path.iterdir()
                     if d.is_dir() and d.name.isdigit() and len(d.name) == 2]
        for huc2_dir in huc2_dirs:
            test_path = self._build_file_path(gage_id, huc2_dir.name)
            if test_path and test_path.exists():
                return huc2_dir.name
        return None

    def _find_alternative_path(self, gage_id: str, huc2: str) -> Optional[Path]:
        alt_path = self.config.data_source_path / "basin_mean_forcing" / huc2 / f"{gage_id}_lump_modis16a2v006_et.txt"
        if alt_path.exists():
            return alt_path
        return None

    def _read_and_process_file(self, file_path: Path, gage_id: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path, sep=',', header=0)
            df = df.rename(columns={
                'Year': 'year', 'Mnth': 'month', 'Day': 'day', 'Hr': 'hour',
                'ET(kg/m^2/8day)': 'evapotranspiration',
                'LE(J/m^2/day)': 'le', 'PET(kg/m^2/8day)': 'pet',
                'PLE(J/m^2/day)': 'ple', 'ET_QC': 'et_qc'
            })
            if 'evapotranspiration' not in df.columns:
                if 'ET(kg/m^2/8day)' in df.columns:
                    df = df.rename(columns={'ET(kg/m^2/8day)': 'evapotranspiration'})
                elif 'ET' in df.columns:
                    df = df.rename(columns={'ET': 'evapotranspiration'})
                else:
                    raise ValueError(f"No evapotranspiration column in {file_path}")
            df['year'] = df['year'].astype(int)
            df['month'] = df['month'].astype(int)
            df['day'] = df['day'].astype(int)
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
            numeric_cols = ['evapotranspiration', 'le', 'pet', 'ple', 'et_qc']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df['gage_id'] = gage_id
            cols_to_drop = ['year', 'month', 'day', 'hour']
            cols_to_drop = [col for col in cols_to_drop if col in df.columns]
            df = df.drop(columns=cols_to_drop)
            df = df[['date', 'gage_id', 'evapotranspiration']]
            logger.debug(f"Loaded ET data from {file_path}: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error reading ET file {file_path}: {e}")
            try:
                df = pd.read_csv(file_path, sep=r'\s+', header=0)
                return self._process_dataframe(df, gage_id)
            except Exception as e2:
                logger.error(f"Failed to read ET file with whitespace separator: {e2}")
                raise

    def _process_dataframe(self, df: pd.DataFrame, gage_id: str) -> pd.DataFrame:
        actual_columns = list(df.columns)
        column_mapping = {}
        for col in actual_columns:
            col_lower = col.lower()
            if 'year' in col_lower:
                column_mapping[col] = 'year'
            elif 'mnth' in col_lower or 'month' in col_lower:
                column_mapping[col] = 'month'
            elif 'day' in col_lower:
                column_mapping[col] = 'day'
            elif 'hr' in col_lower or 'hour' in col_lower:
                column_mapping[col] = 'hour'
            elif 'et' in col_lower and 'qc' not in col_lower:
                column_mapping[col] = 'evapotranspiration'
            elif 'le' in col_lower:
                column_mapping[col] = 'le'
            elif 'pet' in col_lower:
                column_mapping[col] = 'pet'
            elif 'ple' in col_lower:
                column_mapping[col] = 'ple'
            elif 'et_qc' in col_lower or 'qc' in col_lower:
                column_mapping[col] = 'et_qc'
        df = df.rename(columns=column_mapping)
        df['year'] = df['year'].astype(int)
        df['month'] = df['month'].astype(int)
        df['day'] = df['day'].astype(int)
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        if 'evapotranspiration' not in df.columns:
            if 'et' in df.columns:
                df['evapotranspiration'] = df['et']
            else:
                df['evapotranspiration'] = np.nan
        numeric_cols = ['evapotranspiration', 'le', 'pet', 'ple', 'et_qc']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['gage_id'] = gage_id
        cols_to_drop = ['year', 'month', 'day', 'hour']
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        df = df[['date', 'gage_id', 'evapotranspiration']]
        return df

    def _filter_by_dates(self, df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        if df.empty or 'date' not in df.columns:
            return df
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        if start_date:
            df = df[df['date'] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df['date'] <= pd.Timestamp(end_date)]
        return df.reset_index(drop=True)

    def get_et_processing_summary(self) -> Dict[str, Any]:
        if self.data is None or self.data.empty:
            return {}
        summary = {
            'total_gages': self.data['gage_id'].nunique() if 'gage_id' in self.data.columns else 0,
            'total_records': len(self.data),
            'evapotranspiration_coverage': 0.0,
            'date_range': None
        }
        if 'evapotranspiration' in self.data.columns:
            summary['evapotranspiration_coverage'] = self.data['evapotranspiration'].notna().sum() / len(self.data)
        if 'date' in self.data.columns and len(self.data) > 0:
            summary['date_range'] = f"{self.data['date'].min()} to {self.data['date'].max()}"
        return summary