"""
Quality Checker for Hydro Data Processor
Validates data quality based on paper requirements
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class QualityChecker:
    """Checks quality of hydrological data."""
    
    def __init__(self, min_streamflow_coverage: float = 0.95):
        """
        Initialize quality checker.
        
        Args:
            min_streamflow_coverage: Minimum acceptable streamflow coverage (0-1)
        """
        self.min_streamflow_coverage = min_streamflow_coverage
        
        # Quality thresholds
        self.thresholds = {
            'streamflow': {
                'min_coverage': min_streamflow_coverage,
                'max_consecutive_missing': 30,  # days
                'valid_range': (0, 10000)  # mm/day
            },
            'forcing': {
                'min_coverage': 0.90,
                'temperature_range': (-50, 60), # °C
                'precipitation_range': (0, 1000)  # mm/day
            },
            'et': {
                'min_coverage': 0.80,
                'valid_range': (0, 20)  # mm/day
            }
        }
    
    def check_dataset_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive quality check on dataset.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with quality check results
        """
        results = {
            'overall_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {},
            'passed_checks': 0,
            'failed_checks': 0
        }
        
        if df.empty:
            results['overall_valid'] = False
            results['issues'].append('Dataset is empty')
            results['failed_checks'] += 1
            return results
        
        # Check 1: Basic structure
        if not self._check_basic_structure(df, results):
            results['overall_valid'] = False
        
        # Check 2: Streamflow quality (most critical)
        if 'streamflow' in df.columns:
            self._check_streamflow_quality(df, results)
        
        # Check 3: Forcing data quality
        self._check_forcing_quality(df, results)
        
        # Check 4: ET data quality
        if 'et_daily' in df.columns:
            self._check_et_quality(df, results)
        
        # Check 5: Temporal consistency
        self._check_temporal_consistency(df, results)
        
        # Check 6: Value ranges
        self._check_value_ranges(df, results)
        
        # Update overall validity based on critical issues
        if results['failed_checks'] > 0:
            # Check if any critical issues exist
            critical_issues = ['Streamflow coverage below threshold', 
                             'Missing date column',
                             'No streamflow data']
            if any(issue in str(results['issues']) for issue in critical_issues):
                results['overall_valid'] = False
        
        # Calculate statistics
        results['statistics'] = self._calculate_statistics(df)
        
        logger.info(f"Quality check: {results['passed_checks']} passed, "
                   f"{results['failed_checks']} failed")
        
        return results
    
    def _check_basic_structure(self, df: pd.DataFrame, results: Dict[str, Any]) -> bool:
        """Check basic dataset structure."""
        checks_passed = 0
        total_checks = 3
        
        # Check 1: Date column exists
        if 'date' not in df.columns:
            results['issues'].append('Missing date column')
            results['failed_checks'] += 1
            return False
        else:
            checks_passed += 1
        
        # Check 2: Date column is datetime
        try:
            pd.to_datetime(df['date'])
            checks_passed += 1
        except:
            results['issues'].append('Date column cannot be parsed as datetime')
            results['failed_checks'] += 1
        
        # Check 3: No duplicate dates
        if df['date'].duplicated().any():
            results['issues'].append('Duplicate dates found')
            results['failed_checks'] += 1
        else:
            checks_passed += 1
        
        if checks_passed == total_checks:
            results['passed_checks'] += 1
            return True
        return False
    
    def _check_streamflow_quality(self, df: pd.DataFrame, results: Dict[str, Any]):
        """Check streamflow data quality."""
        streamflow = df['streamflow']
        
        # Coverage check
        coverage = streamflow.notna().mean()
        min_coverage = self.thresholds['streamflow']['min_coverage']
        
        if coverage < min_coverage:
            issue = f'Streamflow coverage below threshold: {coverage:.1%} < {min_coverage:.1%}'
            results['issues'].append(issue)
            results['failed_checks'] += 1
        else:
            results['passed_checks'] += 1
        
        # Check for consecutive missing values
        max_consecutive = self._find_max_consecutive_missing(streamflow)
        threshold = self.thresholds['streamflow']['max_consecutive_missing']
        
        if max_consecutive > threshold:
            warning = f'Streamflow has {max_consecutive} consecutive missing days (max: {threshold})'
            results['warnings'].append(warning)
        
        # Check value range
        valid_min, valid_max = self.thresholds['streamflow']['valid_range']
        valid_values = streamflow.dropna()
        
        if not valid_values.empty:
            if (valid_values < valid_min).any() or (valid_values > valid_max).any():
                warning = f'Streamflow values outside expected range: [{valid_min}, {valid_max}]'
                results['warnings'].append(warning)
    
    def _check_forcing_quality(self, df: pd.DataFrame, results: Dict[str, Any]):
        """Check forcing data quality."""
        forcing_vars = ['prcp', 'tmax', 'tmin', 'srad', 'vp']
        available_vars = [var for var in forcing_vars if var in df.columns]
        
        if not available_vars:
            results['warnings'].append('No forcing variables found')
            return
        
        # Check coverage for each forcing variable
        for var in available_vars:
            coverage = df[var].notna().mean()
            min_coverage = self.thresholds['forcing']['min_coverage']
            
            if coverage < min_coverage:
                warning = f'{var} coverage low: {coverage:.1%} < {min_coverage:.1%}'
                results['warnings'].append(warning)
            else:
                results['passed_checks'] += 1
        
        # Check temperature range
        if 'tmax' in df.columns and 'tmin' in df.columns:
            temp_min, temp_max = self.thresholds['forcing']['temperature_range']
            
            tmax_valid = df['tmax'].dropna()
            tmin_valid = df['tmin'].dropna()
            
            if not tmax_valid.empty and ((tmax_valid < temp_min) | (tmax_valid > temp_max)).any():
                warning = f'Temperature values outside expected range: [{temp_min}, {temp_max}]°C'
                results['warnings'].append(warning)
            
            if not tmin_valid.empty and ((tmin_valid < temp_min) | (tmin_valid > temp_max)).any():
                warning = f'Temperature values outside expected range: [{temp_min}, {temp_max}]°C'
                results['warnings'].append(warning)
        
        # Check precipitation range
        if 'prcp' in df.columns:
            prcp_min, prcp_max = self.thresholds['forcing']['precipitation_range']
            prcp_valid = df['prcp'].dropna()
            
            if not prcp_valid.empty and ((prcp_valid < prcp_min) | (prcp_valid > prcp_max)).any():
                warning = f'Precipitation values outside expected range: [{prcp_min}, {prcp_max}] mm/day'
                results['warnings'].append(warning)
    
    def _check_et_quality(self, df: pd.DataFrame, results: Dict[str, Any]):
        """Check ET data quality."""
        et_coverage = df['et_daily'].notna().mean()
        min_coverage = self.thresholds['et']['min_coverage']
        
        if et_coverage < min_coverage:
            warning = f'ET coverage low: {et_coverage:.1%} < {min_coverage:.1%}'
            results['warnings'].append(warning)
        else:
            results['passed_checks'] += 1
        
        # Check ET value range
        et_min, et_max = self.thresholds['et']['valid_range']
        et_valid = df['et_daily'].dropna()
        
        if not et_valid.empty and ((et_valid < et_min) | (et_valid > et_max)).any():
            warning = f'ET values outside expected range: [{et_min}, {et_max}] mm/day'
            results['warnings'].append(warning)
    
    def _check_temporal_consistency(self, df: pd.DataFrame, results: Dict[str, Any]):
        """Check temporal consistency of data."""
        if 'date' not in df.columns:
            return
        
        # Check for gaps in date sequence
        df_sorted = df.sort_values('date')
        date_diff = df_sorted['date'].diff().dt.days
        
        if (date_diff > 1).any():
            max_gap = date_diff.max()
            warning = f'Date sequence has gaps, maximum gap: {max_gap} days'
            results['warnings'].append(warning)
        else:
            results['passed_checks'] += 1
        
        # Check for seasonality in streamflow (basic check)
        if 'streamflow' in df.columns:
            monthly_mean = df.groupby(df['date'].dt.month)['streamflow'].mean()
            if monthly_mean.std() < 0.1 * monthly_mean.mean():
                warning = 'Streamflow shows weak seasonality'
                results['warnings'].append(warning)
    
    def _check_value_ranges(self, df: pd.DataFrame, results: Dict[str, Any]):
        """Check for physically implausible values."""
        # Check for negative precipitation (except trace amounts)
        if 'prcp' in df.columns:
            negative_prcp = df[df['prcp'] < 0]
            if not negative_prcp.empty:
                warning = f'Found {len(negative_prcp)} days with negative precipitation'
                results['warnings'].append(warning)
        
        # Check tmax < tmin
        if 'tmax' in df.columns and 'tmin' in df.columns:
            tmax_tmin_invalid = df[df['tmax'] < df['tmin']]
            if not tmax_tmin_invalid.empty:
                warning = f'Found {len(tmax_tmin_invalid)} days where tmax < tmin'
                results['issues'].append(warning)
                results['failed_checks'] += 1
            else:
                results['passed_checks'] += 1
        
        # Check for extremely high values
        extreme_checks = [
            ('prcp', 500, 'Extremely high precipitation (>500 mm/day)'),
            ('streamflow', 5000, 'Extremely high streamflow (>5000 mm/day)'),
            ('et_daily', 15, 'Extremely high ET (>15 mm/day)')
        ]
        
        for var, threshold, message in extreme_checks:
            if var in df.columns:
                extreme_values = df[df[var] > threshold]
                if not extreme_values.empty:
                    warning = f'{message}: {len(extreme_values)} occurrences'
                    results['warnings'].append(warning)
    
    def _find_max_consecutive_missing(self, series: pd.Series) -> int:
        """Find maximum consecutive missing values in a series."""
        if series.empty:
            return 0
        
        # Create a boolean series: True for missing, False for not missing
        is_missing = series.isna()
        
        # Find consecutive runs of True values
        max_consecutive = 0
        current_consecutive = 0
        
        for value in is_missing:
            if value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for the dataset."""
        stats = {
            'total_days': len(df),
            'date_range': {},
            'variable_stats': {}
        }
        
        # Date range
        if 'date' in df.columns and not df.empty:
            stats['date_range'] = {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d'),
                'days': len(df)
            }
        
        # Statistics for each variable
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            valid_values = df[col].dropna()
            
            if not valid_values.empty:
                stats['variable_stats'][col] = {
                    'count': len(valid_values),
                    'mean': float(valid_values.mean()),
                    'std': float(valid_values.std()),
                    'min': float(valid_values.min()),
                    'max': float(valid_values.max()),
                    'median': float(valid_values.median()),
                    'missing': int(df[col].isna().sum()),
                    'missing_pct': float(df[col].isna().mean())
                }
        
        return stats
    
    def generate_quality_report(self, quality_results: Dict[str, Any]) -> str:
        """Generate a readable quality report."""
        report_lines = []
        
        report_lines.append("=" * 70)
        report_lines.append("DATA QUALITY REPORT")
        report_lines.append("=" * 70)
        
        # Overall status
        status = "PASS" if quality_results['overall_valid'] else "FAIL"
        report_lines.append(f"\nOverall Status: {status}")
        
        # Statistics
        stats = quality_results.get('statistics', {})
        if 'date_range' in stats:
            date_range = stats['date_range']
            report_lines.append(f"\nDate Range: {date_range.get('start', 'N/A')} to "
                              f"{date_range.get('end', 'N/A')} ({date_range.get('days', 0)} days)")
        
        # Issues
        if quality_results['issues']:
            report_lines.append(f"\nISSUES ({len(quality_results['issues'])}):")
            for issue in quality_results['issues']:
                report_lines.append(f"  - {issue}")
        
        # Warnings
        if quality_results['warnings']:
            report_lines.append(f"\nWARNINGS ({len(quality_results['warnings'])}):")
            for warning in quality_results['warnings'][:10]:  # Limit to first 10
                report_lines.append(f"  - {warning}")
            if len(quality_results['warnings']) > 10:
                report_lines.append(f"  ... and {len(quality_results['warnings']) - 10} more warnings")
        
        # Check counts
        report_lines.append(f"\nChecks: {quality_results['passed_checks']} passed, "
                          f"{quality_results['failed_checks']} failed")
        
        report_lines.append("\n" + "=" * 70)
        
        return "\n".join(report_lines)