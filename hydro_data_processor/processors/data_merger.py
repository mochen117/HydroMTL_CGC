"""
Data Merger for Hydro Data Processor
Merges multiple data sources for hydrological analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

from hydro_data_processor.processors.time_processor import TimeProcessor

logger = logging.getLogger(__name__)


class DataMerger:
    """Merges hydrological data from multiple sources."""

    def __init__(
            self,
            start_date: str = "2001-01-01",
            end_date: str = "2021-09-30"):
        """
        Initialize data merger.

        Args:
            start_date: Study start date
            end_date: Study end date
        """
        self.time_processor = TimeProcessor(start_date, end_date)
        self.study_start = pd.Timestamp(start_date)
        self.study_end = pd.Timestamp(end_date)

        # Define required variables
        self.required_variables = {
            'forcing': ['prcp', 'tmax', 'tmin', 'srad', 'vp', 'dayl'],
            'streamflow': ['streamflow'],
            'et': ['et_daily'],
            'smap': ['ssm']
        }

    def merge_basin_data(self,
                         streamflow_data: pd.DataFrame,
                         forcing_data: pd.DataFrame,
                         et_data: Optional[pd.DataFrame] = None,
                         smap_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Merge all data sources for a single basin.

        Args:
            streamflow_data: Streamflow data
            forcing_data: Meteorological forcing data
            et_data: Evapotranspiration data (optional)
            smap_data: Soil moisture data (optional)

        Returns:
            Merged DataFrame with all variables
        """
        logger.info("Merging data for basin")

        # Create base dataframe for study period
        base_df = pd.DataFrame({
            'date': self.time_processor.create_study_period_index()
        })

        # Merge streamflow data
        merged_df = self._merge_data_source(
            base_df, streamflow_data, 'streamflow')

        # Merge forcing data
        merged_df = self._merge_data_source(merged_df, forcing_data, 'forcing')

        # Merge ET data (if available)
        if et_data is not None and not et_data.empty:
            merged_df = self._merge_et_data(merged_df, et_data)

        # Merge SMAP data (if available)
        if smap_data is not None and not smap_data.empty:
            merged_df = self._merge_smap_data(merged_df, smap_data)

        # Ensure all dates are within study period
        merged_df = merged_df[
            (merged_df['date'] >= self.study_start) &
            (merged_df['date'] <= self.study_end)
        ].copy()

        # Sort by date
        merged_df = merged_df.sort_values('date').reset_index(drop=True)

        # Add data availability flags
        merged_df = self._add_data_availability_flags(merged_df)

        logger.info(
            f"Merged data with {len(merged_df)} days and {len(merged_df.columns)} columns")
        return merged_df

    def _merge_data_source(self, base_df: pd.DataFrame,
                           source_df: pd.DataFrame,
                           source_type: str) -> pd.DataFrame:
        """
        Merge a single data source with base dataframe.

        Args:
            base_df: Base dataframe
            source_df: Source dataframe to merge
            source_type: Type of data source

        Returns:
            Merged dataframe
        """
        if source_df.empty:
            logger.warning(f"No {source_type} data to merge")
            return base_df

        # Ensure date column is datetime
        source_df = source_df.copy()
        if 'date' in source_df.columns:
            source_df['date'] = pd.to_datetime(source_df['date'])

        # Merge with base dataframe
        if source_type == 'streamflow':
            # Streamflow data must have 'streamflow' column
            if 'streamflow' not in source_df.columns:
                logger.error("Streamflow data missing 'streamflow' column")
                return base_df

            merged_df = pd.merge(base_df,
                                 source_df[['date', 'streamflow']],
                                 on='date',
                                 how='left')
        elif source_type == 'forcing':
            # Select relevant forcing variables
            forcing_vars = [col for col in self.required_variables['forcing']
                            if col in source_df.columns]

            if not forcing_vars:
                logger.error("Forcing data missing required variables")
                return base_df

            # Merge forcing variables
            merged_df = pd.merge(base_df,
                                 source_df[['date'] + forcing_vars],
                                 on='date',
                                 how='left')
        else:
            # Generic merge for other data types
            merged_df = pd.merge(base_df, source_df, on='date', how='left')

        return merged_df

    def _merge_et_data(
            self,
            base_df: pd.DataFrame,
            et_data: pd.DataFrame) -> pd.DataFrame:
        """Merge evapotranspiration data."""
        # Process MODIS 8-day composite to daily
        et_daily = self.time_processor.process_modis_8day_composite(et_data)

        if et_daily.empty or 'et_daily' not in et_daily.columns:
            logger.warning("No daily ET data available after processing")
            return base_df

        # Merge ET data
        merged_df = pd.merge(base_df,
                             et_daily[['date', 'et_daily']],
                             on='date',
                             how='left')

        return merged_df

    def _merge_smap_data(
            self,
            base_df: pd.DataFrame,
            smap_data: pd.DataFrame) -> pd.DataFrame:
        """Merge SMAP soil moisture data."""
        # Process SMAP 3-day pattern
        smap_processed = self.time_processor.process_smap_3day_pattern(
            smap_data)

        if smap_processed.empty or 'ssm' not in smap_processed.columns:
            logger.warning("No SMAP data available after processing")
            return base_df

        # Merge SMAP data
        merge_cols = ['date', 'ssm']
        if 'has_smap_data' in smap_processed.columns:
            merge_cols.append('has_smap_data')

        merged_df = pd.merge(base_df,
                             smap_processed[merge_cols],
                             on='date',
                             how='left')

        return merged_df

    def _add_data_availability_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add flags indicating data availability for each variable."""
        df = df.copy()

        # Check streamflow availability
        if 'streamflow' in df.columns:
            df['has_streamflow'] = df['streamflow'].notna()

        # Check forcing variables availability
        forcing_vars = [var for var in self.required_variables['forcing']
                        if var in df.columns]
        if forcing_vars:
            df['has_forcing'] = df[forcing_vars].notna().all(axis=1)

        # Check ET availability
        if 'et_daily' in df.columns:
            df['has_et'] = df['et_daily'].notna()

        # Check SMAP availability
        if 'ssm' in df.columns:
            if 'has_smap_data' not in df.columns:
                df['has_smap_data'] = df['ssm'].notna()

        # Overall data availability
        available_cols = [col for col in df.columns if col.startswith('has_')]
        if available_cols:
            df['has_data'] = df[available_cols].any(axis=1)

        return df

    def check_data_completeness(
            self, merged_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check completeness of merged data.

        Args:
            merged_df: Merged dataframe

        Returns:
            Dictionary with completeness statistics
        """
        stats = {
            'total_days': len(merged_df),
            'variables': {},
            'overall_coverage': 0.0
        }

        # Check each variable group
        for group, variables in self.required_variables.items():
            group_stats = {}

            for var in variables:
                if var in merged_df.columns:
                    available = merged_df[var].notna().sum()
                    total = len(merged_df)
                    coverage = available / total if total > 0 else 0.0

                    group_stats[var] = {
                        'available': int(available),
                        'total': total,
                        'coverage': coverage
                    }

            if group_stats:
                stats['variables'][group] = group_stats

        # Calculate overall coverage (proportion of days with any data)
        if 'has_data' in merged_df.columns:
            overall_available = merged_df['has_data'].sum()
            stats['overall_coverage'] = overall_available / len(merged_df)

        return stats

    def create_xarray_dataset(self, merged_df: pd.DataFrame,
                              basin_id: str,
                              attributes: Dict[str, Any]) -> "xarray.Dataset":
        """
        Create xarray Dataset from merged dataframe.

        Args:
            merged_df: Merged dataframe
            basin_id: Basin identifier
            attributes: Basin attributes

        Returns:
            xarray Dataset
        """
        try:
            import xarray as xr

            # Ensure date is index
            df_copy = merged_df.copy()
            if 'date' in df_copy.columns:
                df_copy.set_index('date', inplace=True)

            # Convert to xarray Dataset
            ds = xr.Dataset.from_dataframe(df_copy)

            # Add basin attributes
            ds.attrs['basin_id'] = basin_id
            ds.attrs['processing_date'] = datetime.now().isoformat()
            ds.attrs['study_period'] = f"{self.study_start.date()} to {self.study_end.date()}"

            # Add variable attributes
            if 'streamflow' in ds:
                ds['streamflow'].attrs = {
                    'units': 'mm/day',
                    'long_name': 'Streamflow',
                    'source': 'USGS NWIS'
                }

            # Add basin-specific attributes
            for key, value in attributes.items():
                if isinstance(value, (int, float, str, bool)):
                    ds.attrs[key] = value

            return ds

        except ImportError:
            logger.error("xarray not installed, cannot create xarray dataset")
            return None
        except Exception as e:
            logger.error(f"Error creating xarray dataset: {e}")
            return None
