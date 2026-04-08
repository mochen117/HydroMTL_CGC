#!/usr/bin/env python3
"""
Hydro Data Processing Pipeline - Main Entry Point
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List
import traceback

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hydro_data_processor.utils.logging_config import setup_logging, log_section
from hydro_data_processor.config.settings import ProjectConfig, ProcessingConfig
from hydro_data_processor.pipeline.main_pipeline import HydroDataPipeline


def validate_data_directory(data_root: Path) -> bool:
    if not data_root.exists():
        logging.error(f"Data directory does not exist: {data_root}")
        return False
    return True


def load_basin_list(file_path: Optional[Path]) -> Optional[List[str]]:
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


def main():
    parser = argparse.ArgumentParser(
        description='Hydro Data Processing Pipeline - Process CAMELS data with multiple sources'
    )
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Root directory containing hydrological data')
    parser.add_argument('--max-basins', type=int, default=None,
                        help='Maximum number of basins to process (None=all)')
    parser.add_argument('--basin-list', type=Path,
                        help='File containing list of basin IDs (one per line)')
    parser.add_argument('--output-dir', type=Path, default=Path('./output'),
                        help='Output directory for processed data (default: ./output)')
    parser.add_argument('--start-date', default='2001-01-01',
                        help='Start date for data processing (YYYY-MM-DD, default: 2001-01-01)')
    parser.add_argument('--end-date', default='2021-09-30',
                        help='End date for data processing (YYYY-MM-DD, default: 2021-09-30)')
    parser.add_argument('--min-coverage', type=float, default=0.95,
                        help='Minimum streamflow data coverage (0.0-1.0, default: 0.95)')
    parser.add_argument('--output-format', choices=['netcdf', 'parquet', 'hdf5'],
                        default='netcdf', help='Output file format (default: netcdf)')
    parser.add_argument('--explore-only', action='store_true',
                        help='Only explore data structure without processing')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be processed without actually doing it')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging (debug level)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress info messages, only show warnings and errors')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output files')
    parser.add_argument('--test-basin', type=str,
                        help='Test processing for a single basin ID')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Set logging level (default: INFO)')

    args = parser.parse_args()

    # Determine effective log level
    if args.quiet:
        log_level = 'WARNING'
    elif args.verbose:
        log_level = 'DEBUG'
    else:
        log_level = args.log_level

    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)

    log_section("HYDRO DATA PROCESSING PIPELINE", logger)

    if not validate_data_directory(args.data_root):
        sys.exit(1)

    selected_basins = []
    if args.test_basin:
        selected_basins = [args.test_basin]
    elif args.basin_list:
        selected_basins = load_basin_list(args.basin_list)

    try:
        processing_config = ProcessingConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            min_streamflow_coverage=args.min_coverage,
            output_format=args.output_format,
            overwrite_existing=args.overwrite
        )

        config = ProjectConfig(
            data_root=args.data_root,
            output_dir=args.output_dir,
            max_basins=args.max_basins,
            selected_basins=selected_basins,
            processing_config=processing_config
        )

        config.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("\nConfiguration Summary:")
        logger.info("-" * 50)
        logger.info(f"Data root                 {config.data_root}")
        logger.info(f"Output directory          {config.output_dir}")
        logger.info(f"Max basins                {'ALL' if config.max_basins is None else config.max_basins}")
        logger.info(f"Selected basins count     {len(config.selected_basins)} specified" if config.selected_basins else "Selected basins count     Not specified")
        logger.info(f"Start date                {config.processing_config.start_date}")
        logger.info(f"End date                  {config.processing_config.end_date}")
        logger.info(f"Min coverage              {config.processing_config.min_streamflow_coverage:.0%}")
        logger.info(f"Output format             {config.processing_config.output_format}")
        logger.info(f"Overwrite existing        {config.processing_config.overwrite_existing}")
        logger.info(f"Log level                 {log_level}")

        log_section("INITIALIZING PIPELINE", logger)
        pipeline = HydroDataPipeline(config)

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

        log_section("PIPELINE COMPLETED", logger)
        logger.info("All processing steps completed successfully.")

    except ImportError as e:
        log_section("IMPORT ERROR", logger)
        logger.error(f"Failed to import required module: {e}")
        logger.error("Please check that all dependencies are installed.")
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