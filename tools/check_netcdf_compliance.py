#!/usr/bin/env python3
"""
NetCDF Compliance Checker for HydroMTL Project
Checks if generated NetCDF files meet paper data requirements
"""

import xarray as xr
import numpy as np
import os
import glob
import json
from datetime import datetime
import pandas as pd
import argparse

def check_netcdf_files(output_dir):
    """Check NetCDF files for compliance with paper data requirements"""
    
    print("=" * 60)
    print("NETCDF FILE COMPLIANCE CHECK")
    print("=" * 60)
    
    # Find all NetCDF files
    nc_files = glob.glob(os.path.join(output_dir, "*.nc"))
    
    if not nc_files:
        print("No NetCDF files found!")
        print("File extensions checked: .nc")
        print(f"Search directory: {output_dir}")
        return []
    
    print(f"Found {len(nc_files)} NetCDF file(s)")
    
    compliance_results = []
    
    # Check each file
    for nc_file in nc_files:
        print(f"\n{'='*60}")
        print(f"Checking file: {os.path.basename(nc_file)}")
        print(f"Full path: {nc_file}")
        print(f"{'='*60}")
        
        file_result = {
            'filename': os.path.basename(nc_file),
            'path': nc_file,
            'compliant': False,
            'issues': [],
            'warnings': [],
            'metadata': {}
        }
        
        try:
            # Open NetCDF file
            ds = xr.open_dataset(nc_file)
            
            # 1. Check basic metadata
            file_size = os.path.getsize(nc_file) / (1024*1024)  # MB
            file_result['metadata']['file_size_mb'] = round(file_size, 2)
            file_result['metadata']['variable_count'] = len(ds.data_vars)
            file_result['metadata']['dimension_count'] = len(ds.dims)
            
            print("\n1. BASIC METADATA:")
            print(f"   File size: {file_size:.2f} MB")
            print(f"   Variables: {len(ds.data_vars)}")
            print(f"   Dimensions: {len(ds.dims)}")
            
            # 2. Check time dimension
            print("\n2. TIME DIMENSION:")
            if 'time' in ds.dims:
                time_len = len(ds.time)
                file_result['metadata']['time_points'] = time_len
                
                # Check time range
                if hasattr(ds.time, 'values'):
                    time_values = ds.time.values
                    try:
                        start_date = pd.Timestamp(time_values[0]).strftime('%Y-%m-%d')
                        end_date = pd.Timestamp(time_values[-1]).strftime('%Y-%m-%d')
                        
                        file_result['metadata']['time_start'] = start_date
                        file_result['metadata']['time_end'] = end_date
                        
                        print(f"   Time points: {time_len}")
                        print(f"   Time range: {start_date} to {end_date}")
                        
                        # Check against expected range (2001-01-01 to 2021-09-30)
                        expected_start = datetime(2001, 1, 1)
                        expected_end = datetime(2021, 9, 30)
                        expected_days = (expected_end - expected_start).days + 1
                        
                        actual_start = pd.Timestamp(time_values[0]).to_pydatetime()
                        actual_end = pd.Timestamp(time_values[-1]).to_pydatetime()
                        actual_days = time_len
                        
                        # Check if dates match expected
                        if actual_start.date() == expected_start.date():
                            print(f"   ✓ Start date matches expected (2001-01-01)")
                        else:
                            message = f"   ✗ Start date mismatch: expected 2001-01-01, got {actual_start.date()}"
                            print(message)
                            file_result['issues'].append(message)
                        
                        if actual_end.date() == expected_end.date():
                            print(f"   ✓ End date matches expected (2021-09-30)")
                        else:
                            message = f"   ✗ End date mismatch: expected 2021-09-30, got {actual_end.date()}"
                            print(message)
                            file_result['issues'].append(message)
                        
                        # Check coverage
                        coverage_ratio = actual_days / expected_days
                        file_result['metadata']['time_coverage_ratio'] = round(coverage_ratio, 3)
                        
                        if coverage_ratio >= 0.95:
                            print(f"   ✓ Time coverage meets requirement (≥95%): {coverage_ratio*100:.1f}%")
                        else:
                            message = f"   ✗ Time coverage insufficient: {coverage_ratio*100:.1f}% (<95%)"
                            print(message)
                            file_result['issues'].append(message)
                            
                    except Exception as e:
                        message = f"   Time parsing error: {e}"
                        print(message)
                        file_result['issues'].append(message)
            else:
                message = "No time dimension found"
                print(f"   ✗ {message}")
                file_result['issues'].append(message)
            
            # 3. Check key variables
            print("\n3. KEY VARIABLES:")
            
            # Expected variables for hydrological studies
            expected_vars = {
                'required': ['streamflow', 'precipitation', 'temperature'],
                'recommended': ['evapotranspiration', 'soil_moisture', 'potential_evapotranspiration']
            }
            
            found_vars = {}
            available_vars = list(ds.data_vars)
            
            for var in available_vars:
                var_name = str(var).lower()
                
                # Classify variable
                if 'streamflow' in var_name or 'q' == var_name or 'discharge' in var_name:
                    found_vars['streamflow'] = var
                elif 'precip' in var_name or 'prcp' in var_name or 'rain' in var_name:
                    found_vars['precipitation'] = var
                elif 'temp' in var_name or 'tmean' in var_name or 'tmax' in var_name or 'tmin' in var_name:
                    found_vars['temperature'] = var
                elif 'et' in var_name or 'evapotranspiration' in var_name or 'evapo' in var_name:
                    found_vars['evapotranspiration'] = var
                elif 'soil' in var_name or 'sm' == var_name or 'moisture' in var_name:
                    found_vars['soil_moisture'] = var
                elif 'pet' in var_name or 'potential_et' in var_name:
                    found_vars['potential_evapotranspiration'] = var
            
            # Report found variables
            for var_type, var_name in found_vars.items():
                print(f"   Found {var_type}: {var_name}")
            
            # Check required variables
            missing_required = []
            for req_var in expected_vars['required']:
                if req_var not in found_vars:
                    missing_required.append(req_var)
                    message = f"   ✗ Missing required variable: {req_var}"
                    print(message)
                    file_result['issues'].append(message)
                else:
                    print(f"   ✓ Required variable present: {req_var}")
            
            # Check variable quality
            for var_name, var_key in found_vars.items():
                var_data = ds[var_key]
                print(f"   Variable {var_key}: shape={var_data.shape}, dtype={var_data.dtype}")
                
                # Check for NaN values
                nan_count = np.isnan(var_data.values).sum()
                total_values = np.prod(var_data.shape)
                nan_percentage = (nan_count / total_values * 100) if total_values > 0 else 0
                
                if nan_count > 0:
                    message = f"      {nan_count} NaN values ({nan_percentage:.1f}%)"
                    print(message)
                    if nan_percentage > 5:
                        file_result['warnings'].append(f"High NaN percentage in {var_key}: {nan_percentage:.1f}%")
                else:
                    print(f"      ✓ No NaN values")
                
                # Check data range
                if hasattr(var_data, 'values'):
                    data = var_data.values.flatten()
                    valid_data = data[~np.isnan(data)]
                    if len(valid_data) > 0:
                        data_min = valid_data.min()
                        data_max = valid_data.max()
                        print(f"      Data range: [{data_min:.4f}, {data_max:.4f}]")
                        
                        # Basic range validation
                        if var_name == 'streamflow' and data_max > 10000:
                            file_result['warnings'].append(f"Streamflow values unusually high: max={data_max:.2f}")
                        elif var_name == 'temperature' and (data_min < -100 or data_max > 100):
                            file_result['warnings'].append(f"Temperature values out of expected range: [{data_min:.1f}, {data_max:.1f}]")
            
            # 4. Check global attributes
            print("\n4. GLOBAL ATTRIBUTES:")
            if ds.attrs:
                file_result['metadata']['global_attributes'] = len(ds.attrs)
                for key, value in ds.attrs.items():
                    print(f"   {key}: {value}")
                
                # Check for important attributes
                important_attrs = ['title', 'institution', 'source', 'history', 'Conventions']
                missing_attrs = [attr for attr in important_attrs if attr not in ds.attrs]
                if missing_attrs:
                    message = f"   Missing important attributes: {missing_attrs}"
                    print(message)
                    file_result['warnings'].append(message)
            else:
                message = "No global attributes found"
                print(f"   ✗ {message}")
                file_result['issues'].append(message)
            
            # 5. Check data quality
            print("\n5. DATA QUALITY:")
            
            # Check time continuity
            if 'time' in ds.dims and len(ds.time) > 1:
                time_diffs = np.diff(ds.time.values)
                try:
                    unique_diffs = np.unique(time_diffs)
                    if len(unique_diffs) == 1:
                        print(f"   ✓ Consistent time intervals")
                    else:
                        message = f"   ✗ Inconsistent time intervals: {unique_diffs}"
                        print(message)
                        file_result['issues'].append(message)
                except:
                    pass
            
            # Check spatial information
            spatial_dims = [dim for dim in ds.dims if dim != 'time']
            if spatial_dims:
                print(f"   Spatial dimensions: {spatial_dims}")
                file_result['metadata']['spatial_dimensions'] = spatial_dims
            
            ds.close()
            
            # Determine overall compliance
            if len(file_result['issues']) == 0:
                file_result['compliant'] = True
                print(f"\n✓ File is COMPLIANT with requirements")
            else:
                print(f"\n✗ File has {len(file_result['issues'])} compliance issue(s)")
            
            compliance_results.append(file_result)
            
        except Exception as e:
            message = f"Error reading file: {e}"
            print(f"   {message}")
            file_result['issues'].append(message)
            compliance_results.append(file_result)
    
    return compliance_results

def check_valid_basins(output_dir):
    """Check valid basins metadata"""
    print(f"\n{'='*60}")
    print("VALID BASINS METADATA CHECK")
    print(f"{'='*60}")
    
    json_file = os.path.join(output_dir, "valid_basins.json")
    txt_file = os.path.join(output_dir, "valid_basins.txt")
    
    basin_info = {}
    
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            basin_data = json.load(f)
        basin_count = len(basin_data)
        print(f"Valid basins in JSON: {basin_count}")
        
        if basin_data:
            # Get first basin as example
            first_basin = list(basin_data.keys())[0]
            basin_info['example'] = {first_basin: basin_data[first_basin]}
            print(f"Example basin: {first_basin}")
            print(f"  Attributes: {list(basin_data[first_basin].keys())}")
    else:
        print("No valid_basins.json file found")
    
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as f:
            basin_ids = [line.strip() for line in f if line.strip()]
        print(f"Valid basins in TXT: {len(basin_ids)}")
        basin_info['count_txt'] = len(basin_ids)
    else:
        print("No valid_basins.txt file found")
    
    return basin_info

def generate_compliance_report(output_dir, results, basin_info):
    """Generate a comprehensive compliance report"""
    
    report_file = os.path.join(output_dir, "compliance_report.txt")
    
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("NETCDF COMPLIANCE REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        # Summary statistics
        compliant_files = [r for r in results if r.get('compliant', False)]
        f.write(f"SUMMARY:\n")
        f.write(f"  Total NetCDF files: {len(results)}\n")
        f.write(f"  Compliant files: {len(compliant_files)}\n")
        f.write(f"  Compliance rate: {len(compliant_files)/len(results)*100:.1f}%\n\n")
        
        # File details
        for i, result in enumerate(results, 1):
            f.write(f"FILE {i}: {result['filename']}\n")
            f.write(f"  Path: {result['path']}\n")
            f.write(f"  Compliant: {'YES' if result['compliant'] else 'NO'}\n")
            f.write(f"  Metadata: {result.get('metadata', {})}\n")
            
            if result['issues']:
                f.write(f"  Issues ({len(result['issues'])}):\n")
                for issue in result['issues']:
                    f.write(f"    - {issue}\n")
            
            if result['warnings']:
                f.write(f"  Warnings ({len(result['warnings'])}):\n")
                for warning in result['warnings']:
                    f.write(f"    - {warning}\n")
            
            f.write("\n")
        
        # Basin information
        if basin_info:
            f.write("BASIN INFORMATION:\n")
            for key, value in basin_info.items():
                if key == 'example':
                    f.write(f"  Example basin details:\n")
                    for basin_id, attrs in value.items():
                        f.write(f"    {basin_id}: {attrs}\n")
                else:
                    f.write(f"  {key}: {value}\n")
    
    print(f"\nCompliance report saved to: {report_file}")
    return report_file

def main():
    parser = argparse.ArgumentParser(description='Check NetCDF file compliance with paper requirements')
    parser.add_argument('--output-dir', '-o', default='./output',
                       help='Output directory containing NetCDF files')
    parser.add_argument('--generate-report', '-r', action='store_true',
                       help='Generate detailed compliance report')
    
    args = parser.parse_args()
    
    # Convert relative path to absolute
    output_dir = os.path.abspath(args.output_dir)
    
    if not os.path.exists(output_dir):
        print(f"Error: Output directory does not exist: {output_dir}")
        return
    
    # Check NetCDF files
    results = check_netcdf_files(output_dir)
    
    # Check valid basins
    basin_info = check_valid_basins(output_dir)
    
    # Generate report if requested
    if args.generate_report and results:
        report_file = generate_compliance_report(output_dir, results, basin_info)
        print(f"\nDetailed report available at: {report_file}")
    
    # Summary
    if results:
        compliant_count = sum(1 for r in results if r.get('compliant', False))
        total_count = len(results)
        print(f"\n{'='*60}")
        print("FINAL COMPLIANCE SUMMARY:")
        print(f"  Files checked: {total_count}")
        print(f"  Compliant files: {compliant_count}")
        print(f"  Compliance rate: {compliant_count/total_count*100:.1f}%")
        print(f"{'='*60}")
    else:
        print("\nNo NetCDF files were checked.")

if __name__ == "__main__":
    main()