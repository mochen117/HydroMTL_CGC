"""
Multi-source data processor for CAMELS dataset.
Integrated with streamflow, forcing, ET, SMAP data.
Modified to handle only paper-required variables from Table 1.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MultiSourceProcessor:
    """Processor for integrating and validating multiple data sources."""
    
    # Paper-required variables from Table 1
    PAPER_REQUIRED_VARIABLES = [
        # Meteorological forcing variables (5)
        'total_precipitation',
        'temperature',
        'specific_humidity',
        'shortwave_radiation',
        'potential_energy',
        
        # Model output variables (2)
        'streamflow',
        'evapotranspiration',
        
        # Model evaluation variable (1)
        'ssm'
    ]
    
    def __init__(self, config: Any):
        """
        Initialize the multi-source processor.
        
        Args:
            config: Project configuration
        """
        self.config = config
        self.study_start = pd.Timestamp(config.processing_config.start_date)
        self.study_end = pd.Timestamp(config.processing_config.end_date)
        self.min_coverage = config.processing_config.min_streamflow_coverage
        
        logger.debug(f"MultiSourceProcessor initialized for period: "
                    f"{self.study_start} to {self.study_end}")
        logger.debug(f"Paper-required variables: {self.PAPER_REQUIRED_VARIABLES}")
    
    def process_single_gauge(self, 
                           streamflow_df: pd.DataFrame,
                           forcing_df: pd.DataFrame,
                           et_df: Optional[pd.DataFrame],
                           smap_df: Optional[pd.DataFrame],
                           gauge_id: str,
                           gauge_attrs: Dict[str, Any]) -> Tuple[bool, Optional[pd.DataFrame], float]:
        """
        Process a single gauge with all data sources.
        
        Returns:
            Tuple of (success, processed_data, coverage_rate)
        """
        try:
            # 1. Merge all data sources with paper variable filtering
            merged_data = self._merge_all_data_with_paper_variables(
                streamflow_df, forcing_df, et_df, smap_df, gauge_id
            )
            
            if merged_data is None or merged_data.empty:
                logger.warning(f"No data to merge for gauge {gauge_id}")
                return False, None, 0.0
            
            # 2. Filter to study period (2001-01-01 to 2021-09-30)
            study_data = self._filter_to_study_period(merged_data)
            
            if study_data.empty:
                logger.warning(f"No data in study period for gauge {gauge_id}")
                return False, None, 0.0
            
            # 3. Calculate missing rate for streamflow
            coverage = self._calculate_coverage(study_data, 'streamflow')
            
            # 4. Check if meets minimum coverage requirement
            if coverage < self.min_coverage:
                logger.info(f"Gauge {gauge_id} skipped: coverage {coverage:.2%} < {self.min_coverage:.2%}")
                return False, study_data, coverage
            
            # 5. Add gauge attributes
            processed_data = self._add_attributes(study_data, gauge_attrs, gauge_id)
            
            # 6. Validate paper requirements
            validation = self._validate_paper_requirements(processed_data, gauge_id)
            if not validation['all_required_present']:
                logger.warning(f"Gauge {gauge_id}: Missing paper-required variables: {validation['missing_variables']}")
            
            logger.info(f"Gauge {gauge_id} processed: {len(processed_data)} days, coverage {coverage:.2%}")
            return True, processed_data, coverage
            
        except Exception as e:
            logger.error(f"Error processing gauge {gauge_id}: {e}")
            return False, None, 0.0
    
    def _merge_all_data_with_paper_variables(self,
                                           streamflow_df: pd.DataFrame,
                                           forcing_df: pd.DataFrame,
                                           et_df: Optional[pd.DataFrame],
                                           smap_df: Optional[pd.DataFrame],
                                           gauge_id: str) -> Optional[pd.DataFrame]:
        """Merge all data sources with paper variable filtering."""
        if streamflow_df is None or streamflow_df.empty:
            return None
        
        # Start with streamflow data
        merged_df = streamflow_df.copy()
        
        # Merge forcing data (only paper-required variables)
        if forcing_df is not None and not forcing_df.empty:
            # Filter forcing data to only paper-required variables
            forcing_paper_vars = [var for var in self.PAPER_REQUIRED_VARIABLES 
                                 if var in forcing_df.columns]
            forcing_cols = ['date'] + forcing_paper_vars
            
            if 'gage_id' in forcing_df.columns:
                forcing_df = forcing_df.drop(columns=['gage_id'])
            
            merged_df = pd.merge(
                merged_df,
                forcing_df[forcing_cols],
                on='date',
                how='inner'
            )
        
        # Merge ET data (only evapotranspiration)
        if et_df is not None and not et_df.empty:
            # Filter to only evapotranspiration
            et_cols = ['date']
            if 'evapotranspiration' in et_df.columns:
                et_cols.append('evapotranspiration')
            
            if 'gage_id' in et_df.columns:
                et_df = et_df.drop(columns=['gage_id'])
            
            merged_df = pd.merge(
                merged_df,
                et_df[et_cols],
                on='date',
                how='left'
            )
        
        # Merge SMAP data (only ssm)
        if smap_df is not None and not smap_df.empty:
            # Filter to only ssm
            smap_cols = ['date']
            if 'ssm' in smap_df.columns:
                smap_cols.append('ssm')
            
            if 'gage_id' in smap_df.columns:
                smap_df = smap_df.drop(columns=['gage_id'])
            
            merged_df = pd.merge(
                merged_df,
                smap_df[smap_cols],
                on='date',
                how='left'
            )
        
        # Ensure gauge_id is included
        merged_df['gauge_id'] = gauge_id
        
        # Filter to only paper-required variables plus date and gauge_id
        paper_vars_in_data = [var for var in self.PAPER_REQUIRED_VARIABLES 
                             if var in merged_df.columns]
        required_cols = ['date', 'gauge_id'] + paper_vars_in_data
        merged_df = merged_df[required_cols]
        
        logger.debug(f"Merged data for gauge {gauge_id}: {len(merged_df)} records, "
                    f"{len(paper_vars_in_data)} paper variables")
        
        return merged_df
    
    def _filter_to_study_period(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """Filter data to the study period (2001-01-01 to 2021-09-30)."""
        mask = (data_df['date'] >= self.study_start) & (data_df['date'] <= self.study_end)
        return data_df[mask].copy()
    
    def _calculate_coverage(self, data_df: pd.DataFrame, variable: str) -> float:
        """
        Calculate data coverage for a specific variable.
        
        Args:
            data_df: DataFrame with data
            variable: Variable name to check coverage
        
        Returns:
            Coverage rate (0.0 to 1.0)
        """
        if variable not in data_df.columns:
            return 0.0
        
        total_days = len(data_df)
        if total_days == 0:
            return 0.0
        
        valid_days = data_df[variable].notna().sum()
        return valid_days / total_days
    
    def _add_attributes(self, data_df: pd.DataFrame, 
                       gauge_attrs: Dict[str, Any], 
                       gauge_id: str) -> pd.DataFrame:
        """Add gauge attributes to the dataset."""
        # Convert gauge_attrs to a format suitable for DataFrame
        for key, value in gauge_attrs.items():
            if isinstance(value, (int, float, str, bool)):
                data_df[f'attr_{key}'] = value
        
        return data_df
    
    def _validate_paper_requirements(self, data_df: pd.DataFrame, gauge_id: str) -> Dict[str, Any]:
        """Validate that the dataset contains all paper-required variables."""
        validation_result = {
            'gauge_id': gauge_id,
            'all_required_present': True,
            'missing_variables': [],
            'present_variables': []
        }
        
        for var in self.PAPER_REQUIRED_VARIABLES:
            if var in data_df.columns:
                validation_result['present_variables'].append(var)
            else:
                validation_result['missing_variables'].append(var)
                validation_result['all_required_present'] = False
        
        if not validation_result['all_required_present']:
            logger.debug(f"Gauge {gauge_id}: Missing paper variables: {validation_result['missing_variables']}")
        
        return validation_result
    
    def generate_basin_summary(self, all_results: List[Dict]) -> pd.DataFrame:
        """
        Generate summary DataFrame for all processed basins.
        
        Args:
            all_results: List of processing results for each basin
        
        Returns:
            Summary DataFrame
        """
        summary_data = []
        
        for result in all_results:
            row = {
                'gauge_id': result.get('gauge_id'),
                'processed': result.get('processed', False),
                'coverage': result.get('coverage', 0.0),
                'days_in_study': result.get('days_in_study', 0),
                'has_et': result.get('has_et', False),
                'has_smap': result.get('has_smap', False),
                'reason': result.get('reason', '')
            }
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by coverage (descending)
        summary_df = summary_df.sort_values('coverage', ascending=False)
        
        # Filter to meet minimum coverage
        valid_basins = summary_df[
            (summary_df['processed'] == True) & 
            (summary_df['coverage'] >= self.min_coverage)
        ]
        
        logger.info(f"Total basins: {len(summary_df)}")
        logger.info(f"Valid basins (coverage >= {self.min_coverage:.0%}): {len(valid_basins)}")
        
        return summary_df