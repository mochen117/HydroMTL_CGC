"""
Gap Handler for Hydro Data Processor
Handles missing data according to paper requirements
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from scipy import interpolate

logger = logging.getLogger(__name__)


class GapHandler:
    """Handles gaps and missing data in hydrological datasets."""

    def __init__(self, method: str = 'paper'):
        """
        Initialize gap handler.

        Args:
            method: Gap handling method ('paper', 'interpolate', 'none')
        """
        self.method = method

        # Gap handling methods based on data type
        self.handling_strategies = {
            'streamflow': {
                'max_gap_days': 7,
                'method': 'linear',
                'allow_fill': True
            },
            'forcing': {
                'max_gap_days': 3,
                'method': 'linear',
                'allow_fill': True
            },
            'et': {
                'max_gap_days': 14,  # MODIS has 8-day resolution
                'method': 'linear',
                'allow_fill': True
            },
            'smap': {
                'max_gap_days': 0,  # Paper: do not fill SMAP gaps
                'method': 'none',
                'allow_fill': False
            }
        }

    def handle_missing_data(self,
                            df: pd.DataFrame,
                            variable_types: Dict[str,
                                                 List[str]] = None) -> pd.DataFrame:
        """
        Handle missing data according to paper requirements.

        Args:
            df: DataFrame with potential missing values
            variable_types: Dictionary mapping variable types to column names

        Returns:
            DataFrame with handled missing values
        """
        if df.empty:
            return df

        df_processed = df.copy()

        # Default variable type mapping
        if variable_types is None:
            variable_types = {
                'streamflow': ['streamflow'],
                'forcing': ['prcp', 'tmax', 'tmin', 'srad', 'vp', 'dayl'],
                'et': ['et_daily'],
                'smap': ['ssm']
            }

        # Apply gap handling for each variable type
        for var_type, columns in variable_types.items():
            strategy = self.handling_strategies.get(var_type, {})

            for column in columns:
                if column in df_processed.columns:
                    df_processed = self._handle_column_gaps(
                        df_processed, column, strategy
                    )

        # Log gap handling summary
        self._log_gap_handling_summary(df, df_processed)

        return df_processed

    def _handle_column_gaps(self, df: pd.DataFrame,
                            column: str,
                            strategy: Dict[str, Any]) -> pd.DataFrame:
        """
        Handle gaps for a specific column.

        Args:
            df: DataFrame
            column: Column name to process
            strategy: Gap handling strategy

        Returns:
            DataFrame with handled gaps
        """
        if self.method == 'none' or not strategy.get('allow_fill', True):
            return df

        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        series = df_processed[column]

        # Identify gaps
        is_missing = series.isna()

        if not is_missing.any():
            return df_processed  # No missing values

        # Find gap segments
        gap_segments = self._find_gap_segments(is_missing)

        # Handle each gap according to strategy
        for gap_start, gap_end in gap_segments:
            gap_length = gap_end - gap_start + 1
            max_gap = strategy.get('max_gap_days', 0)

            if gap_length <= max_gap or max_gap == 0:
                # Fill the gap
                filled_values = self._fill_gap(
                    series, gap_start, gap_end, strategy.get(
                        'method', 'linear'))

                if filled_values is not None:
                    df_processed.loc[gap_start:gap_end, column] = filled_values

        return df_processed

    def _find_gap_segments(
            self, is_missing: pd.Series) -> List[Tuple[int, int]]:
        """
        Find segments of consecutive missing values.

        Args:
            is_missing: Boolean series indicating missing values

        Returns:
            List of (start_index, end_index) tuples for each gap
        """
        gap_segments = []
        current_gap_start = None

        for i, missing in enumerate(is_missing):
            if missing and current_gap_start is None:
                current_gap_start = i
            elif not missing and current_gap_start is not None:
                gap_segments.append((current_gap_start, i - 1))
                current_gap_start = None

        # Check if there's a gap at the end
        if current_gap_start is not None:
            gap_segments.append((current_gap_start, len(is_missing) - 1))

        return gap_segments

    def _fill_gap(self, series: pd.Series,
                  gap_start: int,
                  gap_end: int,
                  method: str = 'linear') -> Optional[pd.Series]:
        """
        Fill a gap using specified method.

        Args:
            series: Original series
            gap_start: Start index of gap
            gap_end: End index of gap
            method: Interpolation method ('linear', 'cubic', 'nearest')

        Returns:
            Filled values for the gap, or None if cannot fill
        """
        # Get valid values before and after gap
        pre_gap_idx = gap_start - 1
        post_gap_idx = gap_end + 1

        # Check if we have valid values on both sides
        valid_before = pre_gap_idx >= 0 and not pd.isna(
            series.iloc[pre_gap_idx])
        valid_after = post_gap_idx < len(
            series) and not pd.isna(series.iloc[post_gap_idx])

        if not (valid_before and valid_after):
            return None  # Cannot fill without both sides

        # Get values for interpolation
        x_before = pre_gap_idx
        x_after = post_gap_idx
        y_before = series.iloc[pre_gap_idx]
        y_after = series.iloc[post_gap_idx]

        # Create interpolation function
        if method == 'linear':
            # Simple linear interpolation
            slope = (y_after - y_before) / (x_after - x_before)

            filled_values = []
            for i in range(gap_start, gap_end + 1):
                interpolated = y_before + slope * (i - x_before)
                filled_values.append(interpolated)

            return pd.Series(
                filled_values, index=range(
                    gap_start, gap_end + 1))

        elif method == 'cubic':
            # Cubic spline interpolation
            try:
                # Need at least 4 points for cubic spline
                # Get more points around the gap
                extra_points = 2
                start_idx = max(0, gap_start - extra_points)
                end_idx = min(len(series) - 1, gap_end + extra_points)

                # Extract valid values in the window
                window_indices = []
                window_values = []

                for idx in range(start_idx, end_idx + 1):
                    if not pd.isna(series.iloc[idx]):
                        window_indices.append(idx)
                        window_values.append(series.iloc[idx])

                if len(window_values) >= 4:  # Need at least 4 points for cubic
                    spline = interpolate.CubicSpline(
                        window_indices, window_values)

                    filled_values = []
                    for i in range(gap_start, gap_end + 1):
                        interpolated = spline(i)
                        filled_values.append(interpolated)

                    return pd.Series(
                        filled_values, index=range(
                            gap_start, gap_end + 1))

            except Exception as e:
                logger.warning(f"Cubic spline interpolation failed: {e}")
                # Fall back to linear
                return self._fill_gap(series, gap_start, gap_end, 'linear')

        elif method == 'nearest':
            # Nearest neighbor (forward fill)
            filled_values = [y_before] * (gap_end - gap_start + 1)
            return pd.Series(
                filled_values, index=range(
                    gap_start, gap_end + 1))

        return None

    def calculate_missing_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate statistics about missing data.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with missing data statistics
        """
        stats = {
            'total_values': len(df),
            'missing_by_column': {},
            'missing_patterns': {},
            'recommendations': []
        }

        # Analyze each column
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                missing_count = df[column].isna().sum()
                missing_pct = missing_count / len(df) if len(df) > 0 else 0

                stats['missing_by_column'][column] = {
                    'missing_count': int(missing_count),
                    'missing_pct': float(missing_pct),
                    'has_data': missing_count < len(df)
                }

                # Check for specific patterns
                if missing_pct > 0:
                    # Find consecutive missing patterns
                    is_missing = df[column].isna()
                    gap_segments = self._find_gap_segments(is_missing)

                    if gap_segments:
                        max_gap = max(
                            end - start + 1 for start,
                            end in gap_segments)
                        stats['missing_patterns'][column] = {
                            'total_gaps': len(gap_segments),
                            'max_gap_length': max_gap,
                            'avg_gap_length': np.mean([end - start + 1 for start, end in gap_segments])
                        }

        # Generate recommendations
        for column, col_stats in stats['missing_by_column'].items():
            missing_pct = col_stats['missing_pct']

            if missing_pct > 0.05:  # More than 5% missing
                stats['recommendations'].append(
                    f"Column '{column}' has {missing_pct:.1%} missing data. "
                    "Consider data source validation."
                )

            if column in stats['missing_patterns']:
                max_gap = stats['missing_patterns'][column]['max_gap_length']
                if max_gap > 30:
                    stats['recommendations'].append(
                        f"Column '{column}' has a gap of {max_gap} days. "
                        "May affect temporal analysis."
                    )

        return stats

    def _log_gap_handling_summary(self, original_df: pd.DataFrame,
                                  processed_df: pd.DataFrame):
        """Log summary of gap handling."""
        if original_df.empty or processed_df.empty:
            return

        numeric_cols = original_df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in original_df.columns and col in processed_df.columns:
                original_missing = original_df[col].isna().sum()
                processed_missing = processed_df[col].isna().sum()

                if original_missing > processed_missing:
                    filled = original_missing - processed_missing
                    logger.info(
                        f"Gap handling for '{col}': "
                        f"filled {filled} values ({filled/original_missing:.1%} of missing)")

    def export_gap_report(self, missing_stats: Dict[str, Any],
                          output_path: str) -> bool:
        """
        Export gap analysis report to file.

        Args:
            missing_stats: Missing statistics dictionary
            output_path: Path to save report

        Returns:
            True if successful
        """
        try:
            import json

            with open(output_path, 'w') as f:
                json.dump(missing_stats, f, indent=2, default=str)

            logger.info(f"Gap report saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export gap report: {e}")
            return False

    def validate_gap_handling(self, original_df: pd.DataFrame,
                              processed_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate gap handling results.

        Args:
            original_df: Original dataframe
            processed_df: Processed dataframe

        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'issues': [],
            'improvements': {}
        }

        # Check basic consistency
        if len(original_df) != len(processed_df):
            results['valid'] = False
            results['issues'].append(f"Length mismatch: "
                                     f"original={len(original_df)}, "
                                     f"processed={len(processed_df)}")

        # Compare missing values
        numeric_cols = original_df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in original_df.columns and col in processed_df.columns:
                orig_missing = original_df[col].isna().sum()
                proc_missing = processed_df[col].isna().sum()

                if proc_missing < orig_missing:
                    filled = orig_missing - proc_missing
                    results['improvements'][col] = {
                        'original_missing': int(orig_missing),
                        'processed_missing': int(proc_missing),
                        'filled': int(filled),
                        'improvement_pct': filled / orig_missing if orig_missing > 0 else 0}
                elif proc_missing > orig_missing:
                    results['issues'].append(
                        f"Column '{col}': increased missing values "
                        f"(original: {orig_missing}, processed: {proc_missing})")

        return results
