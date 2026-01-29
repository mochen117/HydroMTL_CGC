"""
Batch processor for handling multiple basins.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import logging
import json
from tqdm import tqdm
import traceback

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Batch processor for multiple basins."""
    
    def __init__(self, pipeline, multi_source_processor):
        """
        Initialize batch processor.
        
        Args:
            pipeline: Main pipeline instance
            multi_source_processor: MultiSourceProcessor instance
        """
        self.pipeline = pipeline
        self.processor = multi_source_processor
        
        # Results tracking
        self.all_results = []
        self.valid_basins = []
    
    def process_batch(self, gauge_ids: List[str], attributes_df: pd.DataFrame) -> Dict:
        """
        Process a batch of basins.
        
        Args:
            gauge_ids: List of gauge IDs to process
            attributes_df: DataFrame with basin attributes
        
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Starting batch processing for {len(gauge_ids)} basins")
        
        # Clear previous results
        self.all_results = []
        self.valid_basins = []
        
        # Process each basin
        for i, gauge_id in enumerate(tqdm(gauge_ids, desc="Processing basins")):
            try:
                result = self._process_single_basin(gauge_id, attributes_df)
                self.all_results.append(result)
                
                # Track valid basins
                if result.get('processed', False):
                    coverage = result.get('coverage', 0.0)
                    if coverage >= self.processor.min_coverage:
                        self.valid_basins.append(gauge_id)
                
                # Log progress periodically
                if (i + 1) % 10 == 0 or (i + 1) == len(gauge_ids):
                    valid_count = len(self.valid_basins)
                    logger.info(f"Progress: {i+1}/{len(gauge_ids)}, "
                               f"Valid: {valid_count}, "
                               f"Success rate: {valid_count/(i+1):.1%}")
                    
            except Exception as e:
                logger.error(f"Error in batch processing for gauge {gauge_id}: {e}")
                logger.debug(traceback.format_exc())
                
                # Record failure
                self.all_results.append({
                    'gauge_id': gauge_id,
                    'processed': False,
                    'coverage': 0.0,
                    'reason': str(e)[:200]
                })
        
        # Generate statistics
        stats = self._generate_statistics()
        
        # Save valid basins list
        self._save_valid_basins_list()
        
        return stats
    
    def _process_single_basin(self, gauge_id: str, attributes_df: pd.DataFrame) -> Dict:
        """
        Process a single basin using the existing pipeline methods.
        """
        # Get gauge attributes
        gauge_attrs = self.pipeline._get_gauge_attributes(gauge_id, attributes_df)
        
        # Get HUC2 code
        huc2 = self.pipeline._get_huc2_for_gauge(gauge_id, gauge_attrs)
        if not huc2:
            return {
                'gauge_id': gauge_id,
                'processed': False,
                'coverage': 0.0,
                'reason': 'No HUC2 mapping found'
            }
        
        # Load streamflow data
        streamflow_data = self.pipeline._load_streamflow_with_huc2(gauge_id, huc2)
        if streamflow_data is None or streamflow_data.empty:
            return {
                'gauge_id': gauge_id,
                'processed': False,
                'coverage': 0.0,
                'reason': 'No streamflow data'
            }
        
        # Load forcing data
        forcing_data = self.pipeline._load_forcing_with_huc2(gauge_id, huc2)
        if forcing_data is None or forcing_data.empty:
            return {
                'gauge_id': gauge_id,
                'processed': False,
                'coverage': 0.0,
                'reason': 'No forcing data'
            }
        
        # Load ET data (optional)
        et_data = None
        if self.pipeline.et_loader:
            et_data = self.pipeline.et_loader.load([gauge_id], huc2=huc2)
        
        # Load SMAP data (optional)
        smap_data = None
        if self.pipeline.smap_loader:
            smap_data = self.pipeline.smap_loader.load([gauge_id], huc2=huc2)
        
        # Process with multi-source processor
        success, processed_data, coverage = self.processor.process_single_gauge(
            streamflow_data, forcing_data, et_data, smap_data, gauge_id, gauge_attrs
        )
        
        result = {
            'gauge_id': gauge_id,
            'processed': success,
            'coverage': coverage,
            'days_in_study': len(processed_data) if processed_data is not None else 0,
            'has_et': et_data is not None and not et_data.empty,
            'has_smap': smap_data is not None and not smap_data.empty,
            'reason': 'Success' if success else f'Coverage {coverage:.2%} < {self.processor.min_coverage:.2%}'
        }
        
        # Save dataset if successful
        if success and processed_data is not None:
            save_success = self.pipeline._create_and_save_dataset(
                processed_data, gauge_attrs, gauge_id
            )
            if not save_success:
                result['processed'] = False
                result['reason'] = 'Failed to save dataset'
        
        return result
    
    def _generate_statistics(self) -> Dict:
        """Generate processing statistics."""
        total = len(self.all_results)
        processed = sum(1 for r in self.all_results if r.get('processed', False))
        valid = len(self.valid_basins)
        
        # Calculate average coverage for valid basins
        valid_coverage = []
        for result in self.all_results:
            if result.get('processed', False):
                coverage = result.get('coverage', 0.0)
                if coverage >= self.processor.min_coverage:
                    valid_coverage.append(coverage)
        
        avg_coverage = np.mean(valid_coverage) if valid_coverage else 0.0
        
        stats = {
            'total_basins': total,
            'successfully_processed': processed,
            'valid_basins': valid,
            'success_rate': processed / total if total > 0 else 0.0,
            'valid_rate': valid / total if total > 0 else 0.0,
            'average_coverage': avg_coverage,
            'min_coverage_required': self.processor.min_coverage
        }
        
        logger.info(f"Batch processing completed:")
        logger.info(f"  Total basins: {total}")
        logger.info(f"  Successfully processed: {processed}")
        logger.info(f"  Valid basins (coverage >= {self.processor.min_coverage:.0%}): {valid}")
        logger.info(f"  Average coverage of valid basins: {avg_coverage:.2%}")
        
        return stats
    
    def _save_valid_basins_list(self):
        """Save list of valid basins to a file."""
        if not self.valid_basins:
            logger.warning("No valid basins to save")
            return
        
        output_dir = self.pipeline.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as text file (one basin per line)
        txt_file = output_dir / "valid_basins.txt"
        with open(txt_file, 'w') as f:
            for basin_id in sorted(self.valid_basins):
                f.write(f"{basin_id}\n")
        
        # Save as JSON with metadata
        json_file = output_dir / "valid_basins.json"
        metadata = {
            'generated_date': pd.Timestamp.now().isoformat(),
            'total_valid_basins': len(self.valid_basins),
            'min_coverage': self.processor.min_coverage,
            'study_period': {
                'start': self.processor.study_start.isoformat(),
                'end': self.processor.study_end.isoformat()
            },
            'valid_basins': sorted(self.valid_basins)
        }
        
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Valid basins list saved to: {txt_file}")
        logger.info(f"Valid basins metadata saved to: {json_file}")