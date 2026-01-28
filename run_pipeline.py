#!/usr/bin/env python3
"""
Hydro Data Processing Pipeline - Main Entry Point
Optimized version with cleaner logging
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class PipelineLogger:
    """Custom logger for pipeline with clean output format"""
    
    @staticmethod
    def setup(verbose: bool = False) -> logging.Logger:
        """Configure logging with clean format."""
        level = logging.DEBUG if verbose else logging.INFO
        
        # Create custom formatter
        class PipelineFormatter(logging.Formatter):
            def format(self, record):
                # Simplify format for cleaner output
                if record.levelno == logging.INFO:
                    return f"{record.getMessage()}"
                elif record.levelno == logging.WARNING:
                    return f"WARNING: {record.getMessage()}"
                elif record.levelno == logging.ERROR:
                    return f"ERROR: {record.getMessage()}"
                elif record.levelno == logging.DEBUG:
                    return f"[DEBUG] {record.getMessage()}"
                else:
                    return super().format(record)
        
        # Get root logger
        logger = logging.getLogger()
        logger.setLevel(level)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Add console handler with custom formatter
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(PipelineFormatter())
        logger.addHandler(console_handler)
        
        return logger
    
    @staticmethod
    def log_section(title: str, logger: logging.Logger, level: str = "info"):
        """Log a section header."""
        log_method = getattr(logger, level.lower())
        log_method(f"\n{'=' * 50}")
        log_method(f"{title}")
        log_method(f"{'=' * 50}")


def validate_data_directory(data_root: Path) -> bool:
    """Validate that the data directory exists."""
    if not data_root.exists():
        print(f"ERROR: Data directory does not exist: {data_root}")
        print(f"Please create the directory or check the path.")
        return False
    
    # Check for expected subdirectories
    expected_dirs = ['attributes', 'streamflow', 'forcing', 'et', 'smap']
    missing_dirs = []
    
    for subdir in expected_dirs:
        if not (data_root / subdir).exists():
            missing_dirs.append(subdir)
    
    if missing_dirs:
        print(f"WARNING: Missing expected subdirectories: {missing_dirs}")
        print(f"Some data sources may not be available.")
    
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
            
            # For this specific format, split by semicolon and take first part
            if ';' in line:
                parts = line.split(';')
                basin_part = parts[0].strip()
                
                # Extract 8-digit number
                import re
                match = re.search(r'\b\d{8}\b', basin_part)
                if match:
                    basins.append(match.group(0))
                else:
                    print(f"WARNING: Line {i} does not contain 8-digit basin ID: {line[:50]}...")
            else:
                # Assume it's just the basin ID
                if line.isdigit() and len(line) == 8:
                    basins.append(line)
                else:
                    print(f"WARNING: Line {i} is not an 8-digit basin ID: {line}")
        
        # Remove duplicates and sort
        basins = sorted(set(basins))
        
        if not basins:
            print(f"ERROR: No valid basin IDs found in {file_path}")
            return None
        
        return basins
        
    except Exception as e:
        print(f"ERROR: Failed to load basin list from {file_path}: {e}")
        return None


def print_config_summary(config: Dict[str, Any]) -> None:
    """Print configuration summary in a clean format."""
    print("\nConfiguration Summary:")
    print("-" * 30)
    
    for key, value in config.items():
        if key == 'selected_basins' and value:
            print(f"{key:20} {len(value)} basins specified")
        elif key == 'selected_basins':
            print(f"{key:20} Not specified")
        elif isinstance(value, Path):
            print(f"{key:20} {value}")
        elif isinstance(value, list):
            print(f"{key:20} {len(value)} items")
        else:
            print(f"{key:20} {value}")


def inspect_attribute_file(attribute_path: Path, num_rows: int = 5):
    """Inspect the attribute file format."""
    if not attribute_path.exists():
        print(f"ERROR: Attribute file not found: {attribute_path}")
        return
    
    print(f"\nInspecting attribute file: {attribute_path}")
    print("-" * 40)
    
    try:
        # Try different formats
        if attribute_path.suffix.lower() == '.csv':
            df = pd.read_csv(attribute_path, nrows=num_rows)
        elif attribute_path.suffix.lower() == '.txt':
            # Try reading as CSV with semicolon separator
            try:
                df = pd.read_csv(attribute_path, sep=';', nrows=num_rows)
            except:
                # Try reading as fixed width or space separated
                df = pd.read_csv(attribute_path, sep='\s+', nrows=num_rows)
        else:
            print(f"Unknown file format: {attribute_path.suffix}")
            return
        
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst {num_rows} rows:")
        print(df.head(num_rows))
        
        # Check basin_id column
        if 'basin_id' in df.columns:
            print(f"\nBasin ID column sample:")
            for i, val in enumerate(df['basin_id'].head(num_rows)):
                print(f"  Row {i}: {val}")
        
    except Exception as e:
        print(f"Error reading attribute file: {e}")
        # Try to read as raw text
        try:
            with open(attribute_path, 'r') as f:
                lines = [next(f).strip() for _ in range(min(num_rows, 10))]
            print(f"\nFirst {len(lines)} lines as text:")
            for i, line in enumerate(lines):
                print(f"  Line {i}: {line}")
        except Exception as e2:
            print(f"Also failed to read as text: {e2}")


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(
        description='Hydro Data Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data-root ./data --max-basins 10
  %(prog)s --data-root ./data --basin-list basins.txt
  %(prog)s --data-root ./data --explore-only --verbose
  %(prog)s --data-root ./data --inspect-attributes
        """
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
    
    parser.add_argument('--inspect-attributes', action='store_true',
                       help='Inspect attribute file format and exit')
    
    parser.add_argument('--test-basin', type=str,
                       help='Test processing for a single basin ID')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = PipelineLogger.setup(args.verbose)
    PipelineLogger.log_section("HYDRO DATA PROCESSING PIPELINE", logger)
    
    # Validate data directory
    if not validate_data_directory(args.data_root):
        sys.exit(1)
    
    # Load basin list if provided
    selected_basins = None
    if args.basin_list:
        selected_basins = load_basin_list(args.basin_list)
        if selected_basins:
            logger.info(f"Loaded {len(selected_basins)} basins from {args.basin_list}")
        else:
            logger.warning(f"No valid basins found in {args.basin_list}")
            logger.warning("Will use all available basins.")
    
    # Test single basin mode
    if args.test_basin:
        selected_basins = [args.test_basin]
        logger.info(f"Testing single basin: {args.test_basin}")
    
    # Inspect attribute file if requested
    if args.inspect_attributes:
        import pandas as pd
        attribute_path = args.data_root / "attributes" / "basin_attributes.txt"
        if not attribute_path.exists():
            # Try other common names
            for fname in ["basin_attributes.csv", "attributes.txt", "attributes.csv"]:
                test_path = args.data_root / "attributes" / fname
                if test_path.exists():
                    attribute_path = test_path
                    break
        
        inspect_attribute_file(attribute_path)
        sys.exit(0)
    
    try:
        # Import pipeline components
        logger.debug("Importing configuration classes...")
        
        from hydro_data_processor.config.settings import (
            ProjectConfig, 
            ProcessingConfig, 
            DataSourceConfig
        )
        from hydro_data_processor.pipeline.main_pipeline import HydroDataPipeline
        
        # Create processing configuration
        processing_config = ProcessingConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            min_streamflow_coverage=args.min_coverage,
            output_format=args.output_format,
            overwrite_existing=args.overwrite
        )
        
        # Create project configuration
        project_config = ProjectConfig(
            data_root=args.data_root,
            max_basins=args.max_basins,
            output_dir=args.output_dir,
            processing_config=processing_config,
            selected_basins=selected_basins  # Pass selected basins if provided
        )
        
        # Ensure output directory exists
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Print configuration summary
        config_summary = {
            'Data root': str(project_config.data_root),
            'Output directory': str(project_config.output_dir),
            'Start date': processing_config.start_date,
            'End date': processing_config.end_date,
            'Min coverage': processing_config.min_streamflow_coverage,
            'Output format': processing_config.output_format,
            'Max basins': project_config.max_basins,
            'Selected basins': selected_basins,
            'Overwrite existing': processing_config.overwrite_existing
        }
        
        print_config_summary(config_summary)
        
        # Initialize pipeline
        PipelineLogger.log_section("INITIALIZING PIPELINE", logger)
        pipeline = HydroDataPipeline(project_config)
        
        # Run pipeline based on mode
        if args.dry_run:
            PipelineLogger.log_section("DRY RUN MODE", logger)
            logger.info("Showing what would be processed...")
            pipeline.explore_data_structure()
            
        elif args.explore_only:
            PipelineLogger.log_section("EXPLORATION MODE", logger)
            pipeline.explore_data_structure()
            
        else:
            PipelineLogger.log_section("RUNNING PIPELINE", logger)
            pipeline.run()
        
        # Success message
        PipelineLogger.log_section("PIPELINE COMPLETED", logger)
        logger.info("All processing steps completed successfully.")
        
    except ImportError as e:
        PipelineLogger.log_section("IMPORT ERROR", logger, "error")
        logger.error(f"Failed to import required module: {e}")
        logger.error("Please check that all dependencies are installed.")
        logger.error("You may need to run: pip install -r requirements.txt")
        sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user.")
        sys.exit(0)
        
    except Exception as e:
        PipelineLogger.log_section("PIPELINE FAILED", logger, "error")
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()