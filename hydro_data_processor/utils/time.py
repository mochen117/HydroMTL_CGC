"""
Time utilities for Hydro Data Processor
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Any
import logging

logger = logging.getLogger(__name__)


def parse_camels_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse dates in CAMELS data files.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with parsed dates
    """
    df = df.copy()

    # Try to find date column
    date_column = None

    # Check common date column names
    date_candidates = ['date', 'Date', 'DATE', 'time', 'Time', 'TIME']

    for col in date_candidates:
        if col in df.columns:
            date_column = col
            break

    # If not found, check for Year/Month/Day columns
    if date_column is None:
        if all(col in df.columns for col in ['Year', 'Mnth', 'Day']):
            try:
                # Create date from Year, Month, Day columns
                df['date'] = pd.to_datetime(
                    df[['Year', 'Mnth', 'Day']].rename(
                        columns={'Year': 'year', 'Mnth': 'month', 'Day': 'day'}
                    )
                )
                date_column = 'date'
            except Exception as e:
                logger.warning(
                    "Could not parse Year/Mnth/Day columns: %s", str(e))

    # Parse date column if found
    if date_column and date_column in df.columns:
        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except Exception as e:
            logger.warning(
                "Could not parse %s column: %s",
                date_column,
                str(e))

        # Rename to standard 'date' if different
        if date_column != 'date':
            df = df.rename(columns={date_column: 'date'})

    return df


def calculate_coverage(series: pd.Series,
                       start_date: str,
                       end_date: str) -> float:
    """
    Calculate data coverage ratio for given date range.

    Args:
        series: Time series data
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Coverage ratio (0.0 to 1.0)
    """
    if series.empty:
        return 0.0

    # Create full date range
    full_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Align series with full range
    aligned = series.reindex(full_range)

    # Calculate coverage
    coverage = 1.0 - aligned.isna().mean()

    return float(coverage)


def resample_8day_to_daily(series: pd.Series,
                           method: str = 'forward_fill') -> pd.Series:
    """
    Resample 8-day data to daily resolution.

    Args:
        series: 8-day time series (data every 8 days)
        method: Resampling method ('forward_fill', 'interpolate', 'divide')

    Returns:
        Daily time series
    """
    if series.empty:
        return series

    # Create daily index for the period
    start_date = series.index.min()
    end_date = series.index.max()
    daily_index = pd.date_range(start=start_date, end=end_date, freq='D')

    # Reindex to daily frequency
    daily_series = series.reindex(daily_index)

    if method == 'forward_fill':
        # Forward fill for 8-day values
        daily_series = daily_series.ffill(limit=7)

    elif method == 'interpolate':
        # Linear interpolation
        daily_series = daily_series.interpolate(method='linear')

    elif method == 'divide':
        # Divide cumulative values by 8 (for MODIS ET)
        # First forward fill, then divide
        daily_series = daily_series.ffill(limit=7)
        daily_series = daily_series / 8.0

    return daily_series


def handle_3day_pattern(series: pd.Series) -> pd.Series:
    """
    Handle SMAP 3-day pattern (data every 3 days).

    Args:
        series: SMAP time series with 3-day pattern

    Returns:
        Series with NaN for missing days (preserving 3-day pattern)
    """
    # SMAP has data on day 1, NaN on day 2-3 of each 3-day cycle
    # We keep this pattern as per paper methodology
    return series


def create_date_range(start_date: str,
                      end_date: str,
                      freq: str = 'D') -> pd.DatetimeIndex:
    """
    Create date range.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        freq: Frequency string

    Returns:
        DatetimeIndex
    """
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def filter_by_date(df: pd.DataFrame,
                   start_date: str,
                   end_date: str) -> pd.DataFrame:
    """
    Filter DataFrame by date range.

    Args:
        df: Input DataFrame with datetime index
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("DataFrame does not have datetime index")
        return df

    return df.loc[start_date:end_date]
