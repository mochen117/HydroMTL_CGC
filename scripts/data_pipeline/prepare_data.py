# scripts/prepare_data.py
"""
Prepare data for multi-task hydrological modeling
"""

import yaml
import glob
from pathlib import Path
from typing import List
import logging
from data.data_loader import DataLoaderFactory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_nc_files(data_dir: str, pattern: str = "gage_*.nc") -> List[str]:
    """
    Find all NetCDF files in the data directory
    
    Args:
        data_dir: Directory containing NetCDF files
        pattern: File pattern to match
        
    Returns:
        List of file paths
    """
    data_path = Path(data_dir)
    nc_files = list(data_path.glob(pattern))
    
    if not nc_files:
        raise FileNotFoundError(f"No files found matching {pattern} in {data_dir}")
    
    # Convert to strings and sort
    nc_files = sorted([str(f) for f in nc_files])
    
    logger.info(f"Found {len(nc_files)} NetCDF files")
    
    return nc_files


def extract_basin_ids(nc_files: List[str]) -> List[str]:
    """
    Extract basin IDs from file names
    
    Args:
        nc_files: List of NetCDF file paths
        
    Returns:
        List of basin IDs
    """
    basin_ids = []
    
    for nc_file in nc_files:
        # Extract basin ID from file name (e.g., "gage_10023000.nc" -> "10023000")
        file_name = Path(nc_file).stem  # Remove extension
        basin_id = file_name.replace("gage_", "")
        basin_ids.append(basin_id)
    
    return basin_ids


def main():
    """Main function to prepare data"""
    
    # Load configuration
    with open("configs/data.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Find NetCDF files
    data_dir = "./output_all_basins"  # Your data directory
    nc_files = find_nc_files(data_dir)
    
    # Limit to 50 basins for testing
    nc_files = nc_files[:50]
    
    # Extract basin IDs
    basin_ids = extract_basin_ids(nc_files)
    
    logger.info(f"Using {len(basin_ids)} basins: {basin_ids[:5]}...")
    
    # Create data loaders
    train_loader, val_loader, test_loader, metadata = DataLoaderFactory.create_dataloaders(
        nc_files=nc_files,
        basin_ids=basin_ids,
        config=config['data'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    
    # Test the data loaders
    logger.info("Testing data loaders...")
    
    # Get a batch from training loader
    train_batch = next(iter(train_loader))
    
    logger.info(f"Batch features shape: {train_batch['features'].shape}")
    logger.info(f"Batch streamflow shape: {train_batch['streamflow'].shape}")
    logger.info(f"Batch evapotranspiration shape: {train_batch['evapotranspiration'].shape}")
    logger.info(f"Batch basin indices shape: {train_batch['basin_idx'].shape}")
    
    # Print metadata
    logger.info(f"Input dimension: {metadata['input_dim']}")
    logger.info(f"Number of basins: {metadata['num_basins']}")
    
    # Save metadata
    import json
    with open("./data/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Data preparation complete!")
    
    return train_loader, val_loader, test_loader, metadata


if __name__ == "__main__":
    train_loader, val_loader, test_loader, metadata = main()