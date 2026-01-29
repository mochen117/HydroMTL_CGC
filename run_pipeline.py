#!/usr/bin/env python3
"""
Hydro Data Processing Pipeline - Main Entry Point
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import custom logging
from hydro_data_processor.utils.logging_config import setup_logging, log_section
from hydro_data_processor.config.settings import ProjectConfig, ProcessingConfig
from hydro_data_processor.pipeline.main_pipeline import HydroDataPipeline


def validate_data_directory(data_root: Path) -> bool:
    """Validate that the data directory exists."""
    if not data_root.exists():
        logging.error(f"Data directory does not exist: {data_root}")
        return False
    
    return True


def load_basin_list(file_path: Optional[Path]) -> Optional[List[str]]:
    """Load basin IDs from a file and validate format."""
    if not file_path or not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        basins = []
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if ';' in line:
                parts = line.split(';')
                basin_part = parts[0].strip()
                
                import re
                match = re.search(r'\b\d{8}\b', basin_part)
                if match:
                    basins.append(match.group(0))
                else:
                    logging.debug(f"Line {i} does not contain 8-digit basin ID: {line[:50]}...")
            else:
                if line.isdigit() and len(line) == 8:
                    basins.append(line)
                else:
                    logging.debug(f"Line {i} is not an 8-digit basin ID: {line}")
        
        basins = sorted(set(basins))
        
        if not basins:
            logging.error(f"No valid basin IDs found in {file_path}")
            return None
        
        return basins
        
    except Exception as e:
        logging.error(f"Failed to load basin list from {file_path}: {e}")
        return None


def print_config_summary(config: Dict[str, Any]) -> None:
    """Print configuration summary in a clean format."""
    logging.info("\nConfiguration Summary:")
    logging.info("-" * 30)
    
    for key, value in config.items():
        if key == 'selected_basins' and value:
            logging.info(f"{key:20} {len(value)} basins specified")
        elif key == 'selected_basins':
            logging.info(f"{key:20} Not specified")
        elif isinstance(value, Path):
            logging.info(f"{key:20} {value}")
        elif isinstance(value, list):
            logging.info(f"{key:20} {len(value)} items")
        else:
            logging.info(f"{key:20} {value}")


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(
        description='Hydro Data Processing Pipeline'
    )
    
    # Required arguments
    parser.add_argument('--data-root', required=True, type=Path,
                       help='Root directory containing hydrological data')
    
    # Optional arguments
    parser.add_argument('--max-basins', type=int, default=None,
                       help='Maximum number of basins to process')
    
    parser.add_argument('--basin-list', type=Path,
                       help='File containing list of basin IDs (one per line)')
    
    parser.add_argument('--output-dir', type=Path, default=Path('./output'),
                       help='Output directory for processed data')
    
    parser.add_argument('--start-date', default='2001-01-01',
                       help='Start date for data processing (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', default='2021-09-30',
                       help='End date for data processing (YYYY-MM-DD)')
    
    parser.add_argument('--min-coverage', type=float, default=0.95,
                       help='Minimum streamflow data coverage (0.0-1.0)')
    
    parser.add_argument('--output-format', choices=['netcdf', 'parquet', 'hdf5'], 
                       default='netcdf', help='Output file format')
    
    # Mode flags
    parser.add_argument('--explore-only', action='store_true',
                       help='Only explore data structure without processing')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without actually doing it')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging (debug level)')
    
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output files')
    
    parser.add_argument('--test-basin', type=str,
                       help='Test processing for a single basin ID')
    
    args = parser.parse_args()
    
    # Setup logging first
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    log_section("HYDRO DATA PROCESSING PIPELINE", logger)
    
    # Test single basin mode
    if args.test_basin:
        logger.info(f"Testing single basin: {args.test_basin}")
    
    # Validate data directory
    if not validate_data_directory(args.data_root):
        sys.exit(1)
    
    # Load basin list if provided
    selected_basins = None
    if args.basin_list:
        selected_basins = load_basin_list(args.basin_list)
        if selected_basins:
            logger.info(f"Loaded {len(selected_basins)} basins from {args.basin_list}")
    
    # Test single basin mode
    if args.test_basin:
        selected_basins = [args.test_basin]
    
    try:
        # Create processing configuration
        processing_config = ProcessingConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            min_streamflow_coverage=args.min_coverage,
            output_format=args.output_format,
            overwrite_existing=args.overwrite
        )
        
        # Create project configuration directly
        config = ProjectConfig(
            data_root=args.data_root,
            output_dir=args.output_dir,
            max_basins=args.max_basins if args.max_basins else 10,
            selected_basins=selected_basins if selected_basins else [],
            processing_config=processing_config
        )
        
        # Ensure output directory exists
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Print configuration summary
        config_summary = {
            'Data root': str(config.data_root),
            'Output directory': str(config.output_dir),
            'Start date': config.processing_config.start_date,
            'End date': config.processing_config.end_date,
            'Min coverage': config.processing_config.min_streamflow_coverage,
            'Output format': config.processing_config.output_format,
            'Max basins': config.max_basins,
            'Selected basins': selected_basins,
            'Overwrite existing': config.processing_config.overwrite_existing
        }
        
        print_config_summary(config_summary)
        
        # Initialize pipeline
        log_section("INITIALIZING PIPELINE", logger)
        pipeline = HydroDataPipeline(config)
        
        # Run pipeline based on mode
        if args.dry_run:
            log_section("DRY RUN MODE", logger)
            logger.info("Showing what would be processed...")
            pipeline.explore_data_structure()
            
        elif args.explore_only:
            log_section("EXPLORATION MODE", logger)
            pipeline.explore_data_structure()
            
        else:
            log_section("RUNNING PIPELINE", logger)
            pipeline.run()
        
        # Success message
        log_section("PIPELINE COMPLETED", logger)
        logger.info("All processing steps completed successfully.")
        
    except ImportError as e:
        log_section("IMPORT ERROR", logger)
        logger.error(f"Failed to import required module: {e}")
        logger.error("Please check that all dependencies are installed.")
        logger.error("You may need to run: pip install -r requirements.txt")
        sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user.")
        sys.exit(0)
        
    except Exception as e:
        log_section("PIPELINE FAILED", logger)
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()