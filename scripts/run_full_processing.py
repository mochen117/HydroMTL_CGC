#!/usr/bin/env python3
"""
HydroMTL Full Processing Runner
"""

import subprocess
import time
import sys
import argparse
from pathlib import Path
import signal
import os
import shutil
import psutil
import yaml

class FullProcessingRunner:
    def __init__(self, data_root=None, output_dir=None, max_basins=None, config_file=None):
        """
        Initialize the processing runner.
        
        Args:
            data_root: Data root directory (if None, will be read from config or default)
            output_dir: Output directory (if None, will use default)
            max_basins: Maximum number of basins to process
            config_file: Path to configuration YAML file (optional)
        """
        self.config_file = Path(config_file) if config_file else None
        self.max_basins = max_basins
        self.log_file = Path("full_processing.log")
        self.process = None
        
        # Load configuration if provided
        self.config = self._load_config() if config_file else {}
        
        # Determine data root with priority: cmd line > config > default
        if data_root:
            self.data_root = Path(data_root)
        elif self.config and 'data_root' in self.config:
            self.data_root = Path(self.config['data_root'])
        else:
            # Default data root if nothing specified
            self.data_root = Path("/home/mochen/hydro_data")
        
        # Determine output directory with priority: cmd line > config > default
        if output_dir:
            self.output_dir = Path(output_dir)
        elif self.config and 'output_dir' in self.config:
            self.output_dir = Path(self.config['output_dir'])
        else:
            self.output_dir = Path("./output_all_basins")
    
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config file {self.config_file}: {e}")
            return {}
    
    def _build_pipeline_command(self):
        """Build command for run_pipeline.py based on configuration."""
        cmd = [
            "python", "run_pipeline.py",
            "--data-root", str(self.data_root),
            "--output-dir", str(self.output_dir),
        ]
        
        # Add max_basins if specified (command line takes priority)
        if self.max_basins is not None:
            cmd.extend(["--max-basins", str(self.max_basins)])
        elif self.config and 'max_basins' in self.config and self.config['max_basins'] is not None:
            cmd.extend(["--max-basins", str(self.config['max_basins'])])
        
        # Add processing configuration from config file
        if self.config and 'processing_config' in self.config:
            pc = self.config['processing_config']
            cmd.extend(["--start-date", pc.get('start_date', '2001-01-01')])
            cmd.extend(["--end-date", pc.get('end_date', '2021-09-30')])
            cmd.extend(["--min-coverage", str(pc.get('min_streamflow_coverage', 0.95))])
            cmd.extend(["--output-format", pc.get('output_format', 'netcdf')])
            
            if pc.get('overwrite_existing', False):
                cmd.append("--overwrite")
        
        # Always use verbose for detailed logging
        cmd.append("--verbose")
        
        return cmd
    
    def validate_environment(self):
        """Validate the processing environment."""
        print("Validating processing environment...")
        
        # Check data directory exists
        if not self.data_root.exists():
            print(f"ERROR: Data root directory does not exist: {self.data_root}")
            return False
        
        # Check required subdirectories exist
        required_dirs = [
            self.data_root / "camels" / "camels_us",
            self.data_root / "nldas4camels",
            self.data_root / "modiset4camels",
            self.data_root / "smap4camels"
        ]
        
        missing_dirs = []
        for req_dir in required_dirs:
            if not req_dir.exists():
                missing_dirs.append(str(req_dir))
        
        if missing_dirs:
            print("WARNING: Some required directories do not exist:")
            for missing in missing_dirs:
                print(f"  - {missing}")
            response = input("Continue anyway? (yes/no): ")
            if response.lower() != 'yes':
                return False
        
        # Check if run_pipeline.py exists
        if not Path("run_pipeline.py").exists():
            print("ERROR: run_pipeline.py not found in current directory")
            return False
        
        return True
    
    def setup_directories(self, force_delete=False):
        """Setup necessary directories."""
        print("Setting up directories...")
        
        # Clean and create output directory
        if self.output_dir.exists():
            print(f"Output directory already exists: {self.output_dir}")
            
            if force_delete:
                response = "yes"
            else:
                response = input("Delete existing output? (yes/no): ")
            
            if response.lower() == 'yes':
                try:
                    shutil.rmtree(self.output_dir)
                    print("Deleted existing output directory")
                except Exception as e:
                    print(f"Error deleting directory: {e}")
                    return False
        
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Output directory ready: {self.output_dir}")
            return True
        except Exception as e:
            print(f"Error creating output directory: {e}")
            return False
    
    def check_resources(self):
        """Check system resources before starting."""
        print("\nChecking system resources...")
        
        # Check disk space on output directory's partition
        try:
            total, used, free = shutil.disk_usage(str(self.output_dir))
            print(f"Disk space on {self.output_dir}:")
            print(f"  Total: {total // (2**30)} GB")
            print(f"  Used: {used // (2**30)} GB")
            print(f"  Free: {free // (2**30)} GB")
            
            # Estimate required disk space
            # Each basin: ~5-10MB for NetCDF + metadata
            basins_to_process = self.max_basins if self.max_basins else 671
            estimated_space = basins_to_process * 10 * 1024 * 1024  # 10MB per basin
            
            if free < estimated_space * 1.5:
                print(f"WARNING: May not have enough disk space")
                print(f"  Estimated required: {estimated_space // (2**20)} MB")
                print(f"  Available: {free // (2**20)} MB")
                response = input("Continue anyway? (yes/no): ")
                if response.lower() != 'yes':
                    return False
        except Exception as e:
            print(f"Warning: Could not check disk space: {e}")
        
        # Check memory
        try:
            memory = psutil.virtual_memory()
            print(f"Memory:")
            print(f"  Total: {memory.total // (2**30)} GB")
            print(f"  Available: {memory.available // (2**30)} GB")
            print(f"  Percent used: {memory.percent}%")
            
            if memory.percent > 90:
                print("WARNING: High memory usage")
                response = input("Continue anyway? (yes/no): ")
                if response.lower() != 'yes':
                    return False
        except Exception as e:
            print(f"Warning: Could not check memory: {e}")
        
        return True
    
    def signal_handler(self, sig, frame):
        """Handle interrupt signals."""
        print(f"\nReceived signal {sig}. Stopping processing...")
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
                print("Process terminated")
            except subprocess.TimeoutExpired:
                print("Process did not terminate gracefully, killing...")
                self.process.kill()
        sys.exit(0)
    
    def estimate_processing_time(self):
        """Estimate total processing time based on basin count."""
        # Based on previous runs: ~30 seconds per basin
        basins_to_process = self.max_basins if self.max_basins else 671
        estimated_seconds = basins_to_process * 30
        
        hours = estimated_seconds // 3600
        minutes = (estimated_seconds % 3600) // 60
        seconds = estimated_seconds % 60
        
        print(f"\nEstimated processing time:")
        print(f"  Basins to process: {basins_to_process}")
        print(f"  Estimated time: {hours}h {minutes}m {seconds}s")
        
        if hours > 2:
            print("  WARNING: This will take several hours. Consider using fewer basins.")
    
    def run(self, force_delete=False, skip_setup=False, skip_checks=False):
        """Run the full processing."""
        print("=" * 70)
        print("HydroMTL Full Processing Runner")
        print("=" * 70)
        
        # Setup signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Validate environment
        if not skip_checks and not self.validate_environment():
            print("Environment validation failed. Exiting...")
            return 1
        
        # Setup directories
        if not skip_setup:
            if not self.setup_directories(force_delete):
                print("Directory setup failed. Exiting...")
                return 1
        
        # Check resources
        if not skip_checks and not self.check_resources():
            print("Resource check failed. Exiting...")
            return 1
        
        # Estimate processing time
        if not skip_checks:
            self.estimate_processing_time()
            response = input("\nContinue with processing? (yes/no): ")
            if response.lower() != 'yes':
                print("Processing cancelled by user")
                return 0
        
        # Build command based on configuration
        cmd = self._build_pipeline_command()
        
        print(f"\nStarting full processing...")
        print(f"Command: {' '.join(cmd)}")
        print(f"Data root: {self.data_root}")
        print(f"Output directory: {self.output_dir}")
        print(f"Log file: {self.log_file}")
        print(f"\nProcessing started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Execute
        start_time = time.time()
        with open(self.log_file, "w") as log_f:
            log_f.write(f"HydroMTL Processing Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write(f"Command: {' '.join(cmd)}\n")
            log_f.write(f"Data root: {self.data_root}\n")
            log_f.write(f"Output directory: {self.output_dir}\n")
            log_f.write("=" * 70 + "\n\n")
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output to both console and log file
            line_count = 0
            for line in self.process.stdout:
                print(line, end='')
                log_f.write(line)
                line_count += 1
                
                # Periodic status updates
                if line_count % 100 == 0:
                    elapsed = time.time() - start_time
                    basins_processed = line_count // 500  # Rough estimate
                    print(f"\n[Progress] Processed ~{basins_processed} basins, elapsed: {elapsed:.1f}s")
        
        # Wait for completion
        return_code = self.process.wait()
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n{'='*70}")
        print(f"Processing completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total processing time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        print(f"Return code: {return_code}")
        
        # Summary
        with open(self.log_file, "a") as log_f:
            log_f.write(f"\n{'='*70}\n")
            log_f.write(f"Processing completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write(f"Total processing time: {total_time:.2f} seconds\n")
            log_f.write(f"Return code: {return_code}\n")
        
        if return_code == 0:
            print("\n✅ Processing completed successfully!")
            # Count generated files
            nc_files = list(self.output_dir.glob("gage_*.nc"))
            print(f"   Generated {len(nc_files)} NetCDF files")
        else:
            print("\n❌ Processing completed with errors")
            print(f"   Check log file for details: {self.log_file}")
        
        return return_code


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HydroMTL Full Processing Runner - Process multiple basins"
    )
    
    # Data location arguments
    data_group = parser.add_argument_group('Data Locations')
    data_group.add_argument(
        "--data-root", "--data",
        type=str,
        help="Path to hydro_data directory"
    )
    
    data_group.add_argument(
        "--output-dir", "--output",
        type=str,
        default="./output_all_basins",
        help="Output directory for processed data (default: ./output_all_basins)"
    )
    
    data_group.add_argument(
        "--config-file", "--config",
        type=str,
        help="Path to optional configuration YAML file"
    )
    
    # Processing control arguments
    process_group = parser.add_argument_group('Processing Control')
    process_group.add_argument(
        "--max-basins",
        type=int,
        default=None,
        help="Maximum number of basins to process (default: all)"
    )
    
    # Execution control arguments
    exec_group = parser.add_argument_group('Execution Control')
    exec_group.add_argument(
        "--force-delete",
        action="store_true",
        help="Automatically delete existing output directory without prompting"
    )
    
    exec_group.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip directory setup"
    )
    
    exec_group.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip all checks (environment, resources, etc.)"
    )
    
    exec_group.add_argument(
        "--log-file",
        type=str,
        help="Log file path (default: full_processing.log)"
    )
    
    # Add help text
    parser.epilog = """
Examples:
  # Process all basins using config file
  python %(prog)s --config-file config.yaml
  
  # Process 10 basins with command line arguments
  python %(prog)s --data-root /path/to/data --max-basins 10
  
  # Force delete existing output
  python %(prog)s --data-root /path/to/data --force-delete
  
  # Skip all checks (quick start)
  python %(prog)s --data-root /path/to/data --skip-checks --skip-setup
  
Priority (highest to lowest):
  1. Command line arguments
  2. Configuration file values
  3. Default values
"""
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Create runner with specified parameters
    runner = FullProcessingRunner(
        data_root=args.data_root,
        output_dir=args.output_dir,
        max_basins=args.max_basins,
        config_file=args.config_file
    )
    
    # Override log file if specified
    if args.log_file:
        runner.log_file = Path(args.log_file)
    
    # Run processing
    sys.exit(runner.run(
        force_delete=args.force_delete,
        skip_setup=args.skip_setup,
        skip_checks=args.skip_checks
    ))