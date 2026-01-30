"""
Monitor the progress of Hydro-MTL pipeline processing.
"""

import time
import json
from pathlib import Path
import xarray as xr


def monitor_processing():
    """Monitor processing progress by checking output directory."""
    output_dir = Path("/home/mochen/code/HydroMTL_CGC/output")
    
    print("=== Hydro-MTL Pipeline Progress Monitor ===\n")
    
    start_time = time.time()
    last_count = 0
    
    while True:
        try:
            # Check if processing is complete
            summary_file = output_dir / "final_processing_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    
                print("Processing Complete!")
                print(f"Total basins analyzed: {summary['statistics']['total_gages']}")
                print(f"Valid basins (≥95% coverage): {summary['statistics']['valid_gages']}")
                print(f"Success rate: {summary['statistics']['success_rate']:.1%}")
                print(f"Average streamflow coverage: {summary.get('coverage_statistics', {}).get('avg_streamflow_coverage', 0):.1%}")
                break
            
            # Count current files
            nc_files = list(output_dir.glob("gage_*.nc"))
            json_files = list(output_dir.glob("gage_*_metadata.json"))
            
            current_count = len(nc_files)
            current_time = time.time()
            
            # Calculate processing rate
            if current_count > last_count and last_count > 0:
                time_per_basin = (current_time - start_time) / current_count
                remaining_basins = 591 - current_count
                estimated_time_remaining = remaining_basins * time_per_basin
                
                print(f"Processed: {current_count}/591 basins ({current_count/591*100:.1f}%)")
                print(f"Rate: {1/time_per_basin:.1f} basins/hour")
                print(f"Estimated time remaining: {estimated_time_remaining/3600:.1f} hours")
            else:
                print(f"Processed: {current_count}/591 basins ({current_count/591*100:.1f}%)")
            
            last_count = current_count
            
            # Display recent activity
            recent_files = []
            for nc_file in nc_files[-5:]:  # Last 5 files
                mtime = nc_file.stat().st_mtime
                if time.time() - mtime < 300:  # Modified in last 5 minutes
                    try:
                        ds = xr.open_dataset(nc_file)
                        gage_id = ds.attrs.get('gage_id', 'unknown')
                        coverage = ds.attrs.get('coverage_achieved', 0)
                        ds.close()
                        recent_files.append(f"{gage_id} ({coverage:.1%})")
                    except:
                        recent_files.append(nc_file.name)
            
            if recent_files:
                print(f"Recent basins: {', '.join(recent_files)}")
            
            print("-" * 50)
            
            # Wait before next check
            time.sleep(60)  # Check every minute
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
            break
        except Exception as e:
            print(f"Error in monitoring: {e}")
            time.sleep(30)


if __name__ == "__main__":
    monitor_processing()