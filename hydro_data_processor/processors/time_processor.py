"""
Time Processor for Hydro Data Processor
Handles temporal operations and study period alignment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TimeProcessor:
    """Process temporal aspects of hydrological data."""
    
    def __init__(self, start_date: str = "2001-01-01", end_date: str = "2021-09-30"):
        """
        Initialize time processor with study period.
        
        Args:
            start_date: Study start date (YYYY-MM-DD)
            end_date: Study end date (YYYY-MM-DD)
        """
        self.study_start = pd.Timestamp(start_date)
        self.study_end = pd.Timestamp(end_date)
        self.study_days = (self.study_end - self.study_start).days + 1
        
        # Critical dates for MODIS year-end adjustments
        self.missing_dates = self._get_missing_dates()
    
    def create_study_period_index(self) -> pd.DatetimeIndex:
        """Create daily datetime index for the entire study period."""
        return pd.date_range(start=self.study_start, end=self.study_end, freq='D')
    
    def align_to_study_period(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        Align data to the study period (2001-01-01 to 2021-09-30).
        
        Args:
            df: Input DataFrame with date column
            date_col: Name of date column
            
        Returns:
            DataFrame aligned to study period
        """
        if df.empty:
            return pd.DataFrame({'date': self.create_study_period_index()})
        
        # Ensure date column is datetime
        if date_col in df.columns:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Create base dataframe for study period
        study_dates = self.create_study_period_index()
        base_df = pd.DataFrame({'date': study_dates})
        
        # Merge with input data
        aligned_df = pd.merge(base_df, df, on=date_col, how='left')
        
        # Add indicator for dates with data
        aligned_df['in_study_period'] = True
        
        return aligned_df
    
    def process_modis_8day_composite(self, modis_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process MOD16A2 8-day composite ET data.
        
        Args:
            modis_df: DataFrame with MODIS ET data (8-day composite)
            
        Returns:
            DataFrame with daily ET values
        """
        if modis_df.empty:
            return pd.DataFrame()
        
        # Ensure date column is datetime
        modis_df = modis_df.copy()
        modis_df['date'] = pd.to_datetime(modis_df['date'])
        
        # MOD16A2 values are sum over 8 days (except year-end adjustments)
        daily_data = []
        
        for _, row in modis_df.iterrows():
            period_start = row['date']
            period_length = 8  # Default 8-day period
            
            # Year-end adjustments (from paper)
            if period_start.month == 12 and period_start.day > 23:
                if period_start.year % 4 == 0:  # Leap year
                    period_length = 6
                else:
                    period_length = 5
            
            # Calculate daily average ET (sum over period divided by period length)
            et_total = row.get('et', row.get('ET', 0))
            daily_et = et_total / period_length if period_length > 0 else 0
            
            # Create daily entries for this period
            for day in range(period_length):
                current_date = period_start + timedelta(days=day)
                if current_date <= self.study_end:  # Only within study period
                    daily_data.append({
                        'date': current_date,
                        'et_daily': daily_et,
                        'period_length': period_length,
                        'et_total': et_total
                    })
        
        daily_df = pd.DataFrame(daily_data)
        
        # Align to study period
        aligned_df = self.align_to_study_period(daily_df)
        
        return aligned_df
    
    def process_smap_3day_pattern(self, smap_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process SMAP 3-day pattern with inherent gaps.
        
        Args:
            smap_df: DataFrame with SMAP soil moisture data
            
        Returns:
            DataFrame with SMAP data and missing pattern indicator
        """
        if smap_df.empty:
            return pd.DataFrame()
        
        # Ensure date column is datetime
        smap_df = smap_df.copy()
        smap_df['date'] = pd.to_datetime(smap_df['date'])
        
        # Align to study period
        aligned_df = self.align_to_study_period(smap_df)
        
        # Add indicator for days with SMAP data (following 3-day pattern)
        if 'ssm' in aligned_df.columns:
            # SMAP has data every 2 days in 3-day cycle
            # We'll mark which days have data based on actual values
            aligned_df['has_smap_data'] = ~aligned_df['ssm'].isna()
        else:
            aligned_df['has_smap_data'] = False
        
        return aligned_df
    
    def aggregate_to_daily(self, hourly_df: pd.DataFrame, 
                          variable_agg_methods: Dict[str, str] = None) -> pd.DataFrame:
        """
        Aggregate hourly NLDAS-2 data to daily scale.
        
        Args:
            hourly_df: DataFrame with hourly data
            variable_agg_methods: Dictionary mapping variables to aggregation methods
            
        Returns:
            DataFrame with daily aggregated data
        """
        if hourly_df.empty:
            return pd.DataFrame()
        
        hourly_df = hourly_df.copy()
        
        # Ensure date column is datetime
        if 'date' in hourly_df.columns:
            hourly_df['date'] = pd.to_datetime(hourly_df['date'])
        elif 'datetime' in hourly_df.columns:
            hourly_df['date'] = pd.to_datetime(hourly_df['datetime'])
            hourly_df = hourly_df.drop('datetime', axis=1)
        
        # Create date-only column for grouping
        hourly_df['date_day'] = hourly_df['date'].dt.date
        
        # Default aggregation methods
        if variable_agg_methods is None:
            variable_agg_methods = {
                'prcp': 'sum',      # Precipitation: sum
                'tmax': 'max',      # Temperature: max
                'tmin': 'min',      # Temperature: min
                'tmean': 'mean',    # Temperature: mean
                'srad': 'mean',     # Radiation: mean
                'vp': 'mean',       # Vapor pressure: mean
                'dayl': 'mean',     # Day length: mean
                'swe': 'mean',      # Snow water equivalent: mean
            }
        
        # Group by day and aggregate
        daily_data = []
        
        for date_day, group in hourly_df.groupby('date_day'):
            day_entry = {'date': pd.Timestamp(date_day)}
            
            for var, agg_method in variable_agg_methods.items():
                if var in group.columns:
                    if agg_method == 'sum':
                        day_entry[var] = group[var].sum()
                    elif agg_method == 'mean':
                        day_entry[var] = group[var].mean()
                    elif agg_method == 'max':
                        day_entry[var] = group[var].max()
                    elif agg_method == 'min':
                        day_entry[var] = group[var].min()
                    elif agg_method == 'median':
                        day_entry[var] = group[var].median()
            
            daily_data.append(day_entry)
        
        daily_df = pd.DataFrame(daily_data)
        
        # Align to study period
        aligned_df = self.align_to_study_period(daily_df)
        
        return aligned_df
    
    def calculate_temporal_coverage(self, df: pd.DataFrame, 
                                   value_column: str) -> Dict[str, float]:
        """
        Calculate temporal coverage statistics.
        
        Args:
            df: DataFrame with data
            value_column: Column to check for missing values
            
        Returns:
            Dictionary with coverage statistics
        """
        if df.empty or value_column not in df.columns:
            return {
                'total_days': self.study_days,
                'available_days': 0,
                'coverage_rate': 0.0,
                'missing_days': self.study_days
            }
        
        # Filter to study period
        study_df = df.copy()
        if 'date' in study_df.columns:
            study_df['date'] = pd.to_datetime(study_df['date'])
            study_df = study_df[
                (study_df['date'] >= self.study_start) & 
                (study_df['date'] <= self.study_end)
            ]
        
        total_days = self.study_days
        available_days = study_df[value_column].notna().sum()
        coverage_rate = available_days / total_days if total_days > 0 else 0.0
        
        return {
            'total_days': total_days,
            'available_days': int(available_days),
            'coverage_rate': coverage_rate,
            'missing_days': total_days - available_days
        }
    
    def _get_missing_dates(self) -> List[pd.Timestamp]:
        """Get dates within study period that should be missing for SMAP."""
        missing_dates = []
        current = self.study_start
        
        # SMAP missing pattern: missing every 2 days in 3-day cycle
        day_counter = 0
        while current <= self.study_end:
            if day_counter % 3 == 1:  # Every 2nd day in 3-day cycle is missing
                missing_dates.append(current)
            current += timedelta(days=1)
            day_counter += 1
        
        return missing_dates