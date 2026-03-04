"""
Evapotranspiration data loader for MODIS16A2 v006.
Modified to properly resample 8-day ET data to daily scale as per paper requirements.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import numpy as np
from datetime import datetime
import calendar

from .base_loader import BaseDataLoader
from hydro_data_processor.config.settings import DataSourceConfig

logger = logging.getLogger(__name__)


class ETLoader(BaseDataLoader):
    """Loader for evapotranspiration data with 8-day to daily resampling."""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config, "et")
        logger.info(f"ETLoader initialized for {config.data_source_path}")

    def load(self, gage_ids: List[str], **kwargs) -> pd.DataFrame:
        """
        Load ET data for specified gages and resample to daily scale.

        Args:
            gage_ids: List of gage IDs to load (8-digit format, e.g., "01013500")
            **kwargs: Additional parameters including:
                - huc2: HUC2 code (optional, can be inferred)
                - start_date: Start date for filtering (optional)
                - end_date: End date for filtering (optional)
                - align_to_study_period: Align to paper study period (default: True)

        Returns:
            DataFrame with ET data resampled to daily scale
        """
        if not gage_ids:
            logger.warning("No gage IDs provided")
            return pd.DataFrame()

        all_data = []
        align_to_study_period = kwargs.get('align_to_study_period', True)
        study_start = kwargs.get('start_date', '2001-01-01')
        study_end = kwargs.get('end_date', '2021-09-30')

        for gage_id in gage_ids:
            try:
                # Load raw 8-day ET data
                raw_data = self._load_single_gage(gage_id, **kwargs)
                
                if raw_data is not None and not raw_data.empty:
                    # Validate raw data
                    validation = self._validate_raw_et_data(raw_data, gage_id)
                    
                    # Resample 8-day ET data to daily
                    resampled_data = self._resample_8day_to_daily(raw_data, gage_id)
                    
                    # Validate resampled data
                    resampled_validation = self._validate_resampled_et_data(resampled_data, gage_id)
                    
                    # Align to study period if requested
                    if align_to_study_period:
                        resampled_data = self._align_to_study_period(resampled_data, study_start, study_end)
                    
                    all_data.append(resampled_data)
                    logger.info(f"Successfully processed ET for gage {gage_id}: "
                               f"raw={len(raw_data)} records, "
                               f"daily={len(resampled_data)} records, "
                               f"coverage={resampled_validation['evapotranspiration_coverage']:.1%}")
                else:
                    logger.warning(f"No ET data found for gage {gage_id}")
            except Exception as e:
                logger.error(f"Error loading ET data for gage {gage_id}: {e}")

        if not all_data:
            return pd.DataFrame()

        # Combine all gage data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Store and return
        self.data = combined_df
        logger.info(f"Loaded and resampled ET data for {len(all_data)} gages")
        return combined_df

    def _resample_8day_to_daily(self, df: pd.DataFrame, gage_id: str) -> pd.DataFrame:
        """
        Resample 8-day ET data to daily scale according to MODIS16A2 v006 specifications.
        
        MODIS16A2 v006: 8-day composites with adjustments for leap years.
        Last period of year: 5 days for non-leap years, 6 days for leap years.
        
        Args:
            df: DataFrame with 8-day ET data
            gage_id: Gage ID for logging
            
        Returns:
            DataFrame with daily ET values
        """
        if df.empty:
            return df
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Create list to store daily values
        daily_data = []
        
        # Process each 8-day period
        for i in range(len(df)):
            period_start = df.iloc[i]['date']
            period_value = df.iloc[i].get('evapotranspiration')
            
            # Skip if value is NaN
            if pd.isna(period_value):
                continue
            
            # Determine period length
            if i < len(df) - 1:
                # Not the last period
                next_date = df.iloc[i + 1]['date']
                period_length = (next_date - period_start).days
                
                # For MODIS 8-day data, period length should be 8, but check for consistency
                if period_length not in [5, 6, 7, 8, 9]:
                    logger.debug(f"Gage {gage_id}: Unexpected period length {period_length} days "
                                f"between {period_start} and {next_date}, assuming 8 days")
                    period_length = 8
            else:
                # Last period of the year
                year = period_start.year
                is_leap = calendar.isleap(year)
                period_length = 6 if is_leap else 5
                logger.debug(f"Gage {gage_id}: Last period of {year} has {period_length} days "
                            f"(leap year: {is_leap})")
            
            # Calculate daily average for this period
            daily_value = period_value / period_length if period_length > 0 else np.nan
            
            # Create daily records for this period
            for day_offset in range(period_length):
                daily_date = period_start + pd.Timedelta(days=day_offset)
                daily_data.append({
                    'date': daily_date,
                    'gage_id': gage_id,
                    'evapotranspiration': daily_value
                })
        
        # Convert to DataFrame
        daily_df = pd.DataFrame(daily_data)
        
        # Sort by date
        daily_df = daily_df.sort_values('date').reset_index(drop=True)
        
        # Create complete date range from min to max date
        if not daily_df.empty:
            min_date = daily_df['date'].min()
            max_date = daily_df['date'].max()
            complete_dates = pd.date_range(start=min_date, end=max_date, freq='D')
            
            # Create DataFrame with all dates
            complete_df = pd.DataFrame({'date': complete_dates})
            
            # Merge with daily data
            daily_df = pd.merge(complete_df, daily_df, on='date', how='left')
            daily_df['gage_id'] = gage_id
            
            # Interpolate small gaps (up to 3 days) for continuity
            if 'evapotranspiration' in daily_df.columns:
                daily_df['evapotranspiration'] = daily_df['evapotranspiration'].interpolate(
                    method='linear',
                    limit=3,
                    limit_direction='both'
                )
        
        logger.debug(f"Resampled ET from {len(df)} 8-day records to {len(daily_df)} daily records for gage {gage_id}")
        return daily_df

    def _validate_raw_et_data(self, df: pd.DataFrame, gage_id: str) -> Dict[str, Any]:
        """Validate raw 8-day ET data quality."""
        validation = {
            'gage_id': gage_id,
            'record_count': len(df),
            'date_range': None,
            'evapotranspiration_coverage': 0.0,
            'period_consistency': False,
            'issues': []
        }
        
        if df.empty:
            validation['issues'].append('Empty DataFrame')
            return validation
        
        # Check date range
        if 'date' in df.columns:
            min_date = df['date'].min()
            max_date = df['date'].max()
            validation['date_range'] = f"{min_date} to {max_date}"
        
        # Check coverage
        if 'evapotranspiration' in df.columns:
            non_nan = df['evapotranspiration'].notna().sum()
            validation['evapotranspiration_coverage'] = non_nan / len(df)
        
        # Check period consistency (should be roughly 8-day intervals)
        if 'date' in df.columns and len(df) > 1:
            df_sorted = df.sort_values('date')
            date_diffs = df_sorted['date'].diff().dt.days.dropna()
            
            # Count intervals that are close to 8 days (7-9 days)
            valid_intervals = date_diffs[(date_diffs >= 7) & (date_diffs <= 9)].shape[0]
            total_intervals = len(date_diffs)
            
            if total_intervals > 0:
                interval_ratio = valid_intervals / total_intervals
                validation['period_consistency'] = interval_ratio >= 0.8
                
                if interval_ratio < 0.8:
                    validation['issues'].append(f'Irregular 8-day intervals: only {interval_ratio:.1%} are 7-9 days')
        
        return validation

    def _validate_resampled_et_data(self, df: pd.DataFrame, gage_id: str) -> Dict[str, Any]:
        """Validate resampled daily ET data quality."""
        validation = {
            'gage_id': gage_id,
            'record_count': len(df),
            'evapotranspiration_coverage': 0.0,
            'has_gaps': False,
            'gap_count': 0
        }
        
        if df.empty:
            return validation
        
        # Check coverage
        if 'evapotranspiration' in df.columns:
            non_nan = df['evapotranspiration'].notna().sum()
            validation['evapotranspiration_coverage'] = non_nan / len(df)
        
        # Check for gaps in the date sequence
        if 'date' in df.columns and len(df) > 1:
            df_sorted = df.sort_values('date')
            date_diffs = df_sorted['date'].diff().dt.days.dropna()
            validation['gap_count'] = (date_diffs > 1).sum()
            validation['has_gaps'] = validation['gap_count'] > 0
        
        return validation

    def _align_to_study_period(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Align ET data to paper study period."""
        study_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        study_df = pd.DataFrame({'date': study_dates})
        
        if df.empty:
            return study_df
        
        # Merge with study dates
        merged = pd.merge(study_df, df, on='date', how='left')
        
        # Fill small gaps at the beginning/end with nearest valid value
        if 'evapotranspiration' in merged.columns:
            # Forward fill for up to 8 days at start
            merged['evapotranspiration'] = merged['evapotranspiration'].ffill(limit=8)
            # Backward fill for up to 8 days at end
            merged['evapotranspiration'] = merged['evapotranspiration'].bfill(limit=8)
        
        # Add gage_id if missing
        if 'gage_id' not in merged.columns and 'gage_id' in df.columns:
            merged['gage_id'] = df['gage_id'].iloc[0]
        
        logger.debug(f"Aligned ET data to study period {start_date} to {end_date}: "
                    f"{len(df)} -> {len(merged)} records")
        
        return merged

    def _load_single_gage(self, gage_id: str, **kwargs) -> Optional[pd.DataFrame]:
        """Load ET data for a single gage."""
        huc2 = kwargs.get('huc2')

        # Try to determine HUC2 if not provided
        if not huc2:
            huc2 = self._find_huc2_for_gage(gage_id)
            if not huc2:
                logger.warning(f"Cannot determine HUC2 for gage {gage_id}")
                return None

        # Build file path
        file_path = self._build_file_path(gage_id, huc2)
        if not file_path or not file_path.exists():
            # Try alternative path patterns
            file_path = self._find_alternative_path(gage_id, huc2)
            if not file_path or not file_path.exists():
                logger.debug(f"ET file not found for gage {gage_id} in HUC2 {huc2}")
                return None

        # Read and process the file
        try:
            return self._read_and_process_file(file_path, gage_id)
        except Exception as e:
            logger.error(f"Failed to process ET file {file_path}: {e}")
            return None

    def _build_file_path(self, gage_id: str, huc2: str) -> Optional[Path]:
        """Build file path from configuration."""
        if not self.config.data_source_path.exists():
            logger.warning(f"Data source path does not exist: {self.config.data_source_path}")
            return None

        # Use the configuration's get_file_path method if available
        if hasattr(self.config, 'get_file_path'):
            return self.config.get_file_path(gage_id, huc2)

        # Otherwise build manually
        path = self.config.data_source_path

        if self.config.subdirectory:
            subdir = self.config.subdirectory.format(huc2=huc2)
            path = path / subdir

        if self.config.file_pattern:
            # Use the original 8-digit gage_id in file pattern
            filename = self.config.file_pattern.replace('{basin_id}', gage_id)
            filename = filename.replace('{gage_id}', gage_id)
            path = path / filename

        return path

    def _find_huc2_for_gage(self, gage_id: str) -> Optional[str]:
        """Try to find HUC2 code for a gage by scanning directories."""
        if not self.config.data_source_path.exists():
            return None

        # Look for HUC2 directories
        huc2_dirs = [d for d in self.config.data_source_path.iterdir()
                     if d.is_dir() and d.name.isdigit() and len(d.name) == 2]

        for huc2_dir in huc2_dirs:
            # Check if file exists in this HUC2 directory
            test_path = self._build_file_path(gage_id, huc2_dir.name)
            if test_path and test_path.exists():
                return huc2_dir.name

        return None

    def _find_alternative_path(self, gage_id: str, huc2: str) -> Optional[Path]:
        """Try alternative path patterns if default doesn't exist."""
        # Try standard CAMELS pattern
        alt_path = self.config.data_source_path / "basin_mean_forcing" / huc2 / f"{gage_id}_lump_modis16a2v006_et.txt"
        if alt_path.exists():
            return alt_path

        return None

    def _read_and_process_file(self, file_path: Path, gage_id: str) -> pd.DataFrame:
        """Read and process ET data file."""
        try:
            # Read ET data (comma-separated with header)
            df = pd.read_csv(file_path, sep=',', header=0)

            # Standardize column names - ensure evapotranspiration column exists
            df = df.rename(columns={
                'Year': 'year',
                'Mnth': 'month',
                'Day': 'day',
                'Hr': 'hour',
                'ET(kg/m^2/8day)': 'evapotranspiration',
                'LE(J/m^2/day)': 'le',
                'PET(kg/m^2/8day)': 'pet',
                'PLE(J/m^2/day)': 'ple',
                'ET_QC': 'et_qc'
            })

            # Check if evapotranspiration column exists
            if 'evapotranspiration' not in df.columns:
                # Try alternative names
                if 'ET(kg/m^2/8day)' in df.columns:
                    df = df.rename(columns={'ET(kg/m^2/8day)': 'evapotranspiration'})
                elif 'ET' in df.columns:
                    df = df.rename(columns={'ET': 'evapotranspiration'})
                else:
                    logger.error(f"No evapotranspiration column found in ET file {file_path}")
                    raise ValueError(f"No evapotranspiration column in {file_path}")

            # Convert numeric columns
            df['year'] = df['year'].astype(int)
            df['month'] = df['month'].astype(int)
            df['day'] = df['day'].astype(int)

            # Create date column
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

            # Convert data columns to numeric
            numeric_cols = ['evapotranspiration', 'le', 'pet', 'ple', 'et_qc']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Add gage_id (original 8-digit format)
            df['gage_id'] = gage_id

            # Drop year/month/day/hour columns
            cols_to_drop = ['year', 'month', 'day', 'hour']
            cols_to_drop = [col for col in cols_to_drop if col in df.columns]
            df = df.drop(columns=cols_to_drop)

            # Keep only evapotranspiration column (paper requirement)
            keep_cols = ['date', 'gage_id', 'evapotranspiration']
            df = df[keep_cols]

            logger.debug(f"Loaded ET data from {file_path}: {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Error reading ET file {file_path}: {e}")
            # Try with different separators if comma fails
            try:
                df = pd.read_csv(file_path, sep=r'\s+', header=0)
                # Apply same column renaming and processing
                return self._process_dataframe(df, gage_id)
            except Exception as e2:
                logger.error(f"Failed to read ET file with whitespace separator: {e2}")
                raise

    def _process_dataframe(self, df: pd.DataFrame, gage_id: str) -> pd.DataFrame:
        """Process ET dataframe after reading with alternative separator."""
        # Standardize column names (may have different format)
        # Find the actual column names
        actual_columns = list(df.columns)
        
        # Map expected column names based on position or partial matching
        column_mapping = {}
        
        for i, col in enumerate(actual_columns):
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
        
        # Continue with processing
        df['year'] = df['year'].astype(int)
        df['month'] = df['month'].astype(int)
        df['day'] = df['day'].astype(int)
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        
        # Ensure evapotranspiration column exists
        if 'evapotranspiration' not in df.columns:
            if 'et' in df.columns:
                df['evapotranspiration'] = df['et']
            else:
                logger.error(f"No evapotranspiration column found in ET data")
                df['evapotranspiration'] = np.nan
        
        numeric_cols = ['evapotranspiration', 'le', 'pet', 'ple', 'et_qc']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['gage_id'] = gage_id
        
        cols_to_drop = ['year', 'month', 'day', 'hour']
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        # Keep only evapotranspiration column (paper requirement)
        col_order = ['date', 'gage_id', 'evapotranspiration']
        df = df[col_order]
        
        return df

    def _filter_by_dates(self, df: pd.DataFrame, start_date: Optional[str],
                         end_date: Optional[str]) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        if df.empty or 'date' not in df.columns:
            return df

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        if start_date:
            start_dt = pd.Timestamp(start_date)
            df = df[df['date'] >= start_dt]

        if end_date:
            end_dt = pd.Timestamp(end_date)
            df = df[df['date'] <= end_dt]

        return df.reset_index(drop=True)

    def get_et_processing_summary(self) -> Dict[str, Any]:
        """Get summary of ET processing."""
        if self.data is None or self.data.empty:
            return {}
        
        summary = {
            'total_gages': self.data['gage_id'].nunique() if 'gage_id' in self.data.columns else 0,
            'total_records': len(self.data),
            'evapotranspiration_coverage': 0.0,
            'date_range': None
        }
        
        if 'evapotranspiration' in self.data.columns:
            non_nan = self.data['evapotranspiration'].notna().sum()
            summary['evapotranspiration_coverage'] = non_nan / len(self.data)
        
        if 'date' in self.data.columns and len(self.data) > 0:
            min_date = self.data['date'].min()
            max_date = self.data['date'].max()
            summary['date_range'] = f"{min_date} to {max_date}"
        
        return summary