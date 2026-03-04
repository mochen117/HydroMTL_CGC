# monitor_and_validate.py
import time
import sys
from pathlib import Path
import json
import os
import subprocess

class MonitorAndValidate:
    def __init__(self, output_dir="./output_all_basins"):
        self.output_dir = Path(output_dir)
        self.log_file = Path("full_processing.log")
        self.expected_total = 671
        
    def get_current_status(self):
        """Get current processing status."""
        status = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'output_dir': str(self.output_dir),
            'expected_basins': self.expected_total
        }
        
        # Count processed basins
        nc_files = list(self.output_dir.glob("gage_*.nc"))
        status['processed_basins'] = len(nc_files)
        status['progress_percent'] = (len(nc_files) / self.expected_total) * 100
        
        # Check summary file
        summary_file = self.output_dir / "final_processing_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                status['summary'] = summary
                status['processing_complete'] = True
            except:
                status['processing_complete'] = False
        else:
            status['processing_complete'] = False
        
        # Check log file for errors
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    lines = f.readlines()
                    status['log_lines'] = len(lines)
                    
                    # Count errors and warnings
                    errors = [l for l in lines if 'ERROR' in l]
                    warnings = [l for l in lines if 'WARNING' in l]
                    status['error_count'] = len(errors)
                    status['warning_count'] = len(warnings)
                    
                    # Get recent errors
                    if errors:
                        status['recent_errors'] = errors[-5:]
            except:
                pass
        
        # Check file sizes
        if nc_files:
            total_size = sum(f.stat().st_size for f in nc_files)
            status['total_size_mb'] = total_size / (1024 * 1024)
            status['avg_size_mb'] = total_size / len(nc_files) / (1024 * 1024) if nc_files else 0
        
        return status
    
    def display_status(self, status):
        """Display current status."""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 70)
        print("HydroMTL Processing Monitor")
        print("=" * 70)
        print(f"Timestamp: {status['timestamp']}")
        print(f"Output directory: {status['output_dir']}")
        print()
        
        print("Processing Status:")
        print(f"  Processed basins: {status['processed_basins']}/{status['expected_basins']}")
        print(f"  Progress: {status['progress_percent']:.1f}%")
        print(f"  Remaining: {status['expected_basins'] - status['processed_basins']}")
        
        if 'total_size_mb' in status:
            print(f"  Total size: {status['total_size_mb']:.1f} MB")
            print(f"  Average file size: {status['avg_size_mb']:.1f} MB")
        
        if 'error_count' in status:
            print(f"  Errors in log: {status.get('error_count', 0)}")
            print(f"  Warnings in log: {status.get('warning_count', 0)}")
        
        if status.get('processing_complete'):
            print(f"\nProcessing complete!")
            if 'summary' in status:
                stats = status['summary'].get('statistics', {})
                print(f"  Success rate: {stats.get('success_rate', 0)*100:.1f}%")
                print(f"  Valid basins: {stats.get('valid_gages', 0)}")
                print(f"  Failed basins: {stats.get('failed_gages', 0)}")
                print(f"  Skipped basins: {stats.get('skipped_gages', 0)}")
        else:
            # Show recent activity
            print(f"\nRecent activity (checking for new files)...")
            try:
                nc_files = list(self.output_dir.glob("gage_*.nc"))
                if nc_files:
                    recent_files = sorted(nc_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]
                    for f in recent_files:
                        file_time = time.strftime('%H:%M:%S', time.localtime(f.stat().st_mtime))
                        print(f"  - {f.stem} ({file_time})")
            except:
                pass
        
        print()
        print("=" * 70)
        print("Press Ctrl+C to stop monitoring")
    
    def quick_validate(self):
        """Quick validation of processed data."""
        print("\n" + "=" * 70)
        print("Quick Data Validation")
        print("=" * 70)
        
        nc_files = list(self.output_dir.glob("gage_*.nc"))
        if not nc_files:
            print("No NetCDF files found for validation")
            return
        
        print(f"Found {len(nc_files)} NetCDF files")
        
        # Check first 3 files
        test_files = nc_files[:3]
        print(f"Testing first {len(test_files)} files...")
        
        for nc_file in test_files:
            try:
                result = subprocess.run(
                    ["ncdump", "-h", str(nc_file)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    print(f"  ✓ {nc_file.stem}: NetCDF header readable")
                    # Count variables
                    lines = result.stdout.split('\n')
                    var_count = sum(1 for line in lines if ':' in line and '(' in line)
                    print(f"    Variables: {var_count}")
                else:
                    print(f"  ✗ {nc_file.stem}: Error reading header")
            except Exception as e:
                print(f"  ✗ {nc_file.stem}: Validation failed - {e}")
        
        # Check file sizes
        print(f"\nFile size statistics:")
        sizes = [f.stat().st_size / (1024 * 1024) for f in nc_files]  # MB
        print(f"  Min: {min(sizes):.1f} MB")
        print(f"  Max: {max(sizes):.1f} MB")
        print(f"  Avg: {sum(sizes)/len(sizes):.1f} MB")
        
        print("\nQuick validation complete!")
    
    def run_monitor(self, interval=60):
        """Run monitoring loop."""
        print(f"Starting monitor for {self.output_dir}")
        print(f"Update interval: {interval} seconds\n")
        
        try:
            while True:
                status = self.get_current_status()
                self.display_status(status)
                
                # Check if processing is complete
                if status.get('processing_complete'):
                    print("\nProcessing complete. Running quick validation...")
                    time.sleep(2)
                    self.quick_validate()
                    
                    print("\nWould you like to:")
                    print("  1. Continue monitoring")
                    print("  2. Exit")
                    choice = input("Enter choice (1 or 2): ")
                    if choice == '2':
                        break
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            
            # Ask if user wants to run validation
            response = input("Run quick validation on current data? (yes/no): ")
            if response.lower() == 'yes':
                self.quick_validate()
    
    def generate_report(self):
        """Generate a final report."""
        print("\n" + "=" * 70)
        print("Generating Final Report")
        print("=" * 70)
        
        report = {
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'output_directory': str(self.output_dir.resolve())
        }
        
        # Count files
        nc_files = list(self.output_dir.glob("gage_*.nc"))
        json_files = list(self.output_dir.glob("gage_*_metadata.json"))
        
        report['netcdf_files'] = len(nc_files)
        report['metadata_files'] = len(json_files)
        report['expected_basins'] = self.expected_total
        
        # Check summary file
        summary_file = self.output_dir / "final_processing_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                report['processing_summary'] = summary
            except Exception as e:
                report['summary_error'] = str(e)
        
        # Calculate disk usage
        if nc_files:
            total_size = sum(f.stat().st_size for f in nc_files)
            report['total_disk_usage_mb'] = total_size / (1024 * 1024)
            report['average_file_size_mb'] = total_size / len(nc_files) / (1024 * 1024)
        
        # Save report
        report_file = self.output_dir / "processing_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to: {report_file}")
        print(f"\nSummary:")
        print(f"  NetCDF files: {report['netcdf_files']}/{self.expected_total}")
        print(f"  Metadata files: {report['metadata_files']}")
        if 'total_disk_usage_mb' in report:
            print(f"  Total disk usage: {report['total_disk_usage_mb']:.1f} MB")
        
        return report

if __name__ == "__main__":
    # Set output directory
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./output_all_basins"
    
    monitor = MonitorAndValidate(output_dir)
    
    print("=" * 70)
    print("HydroMTL Monitor and Validation Tool")
    print("=" * 70)
    print("\nOptions:")
    print("  1. Monitor processing progress")
    print("  2. Quick validate current data")
    print("  3. Generate final report")
    print("  4. Exit")
    
    choice = input("\nEnter choice (1-4): ")
    
    if choice == '1':
        interval = input("Enter update interval in seconds (default 60): ")
        interval = int(interval) if interval.isdigit() else 60
        monitor.run_monitor(interval)
    elif choice == '2':
        monitor.quick_validate()
    elif choice == '3':
        monitor.generate_report()
    elif choice == '4':
        print("Exiting...")
    else:
        print("Invalid choice")