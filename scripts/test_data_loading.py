# scripts/test_data_loading.py
"""
Test script for data loading with correct imports
"""

import sys
import os
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import from the correct location: mtl_cgc.data.data_set
try:
    from mtl_cgc.data.data_set import HydroBasinDataset, MultiBasinDataset
    print("Successfully imported from mtl_cgc.data.data_set")
    
    # Also check if we can import the data loader
    from mtl_cgc.data.data_loader import DataLoaderFactory
    print("Successfully imported from mtl_cgc.data.data_loader")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("\nLet's check what's in mtl_cgc.data:")
    data_dir = os.path.join(project_root, "mtl_cgc/data")
    if os.path.exists(data_dir):
        print(f"Contents of {data_dir}:")
        for item in os.listdir(data_dir):
            print(f"  {item}")
    sys.exit(1)


def test_single_basin():
    """Test loading a single basin"""
    
    # Test with one basin
    nc_file = os.path.join(project_root, "output_all_basins/gage_10023000.nc")
    basin_id = "10023000"
    
    print(f"\nTesting single basin:")
    print(f"  NC file: {nc_file}")
    print(f"  File exists: {os.path.exists(nc_file)}")
    
    if not os.path.exists(nc_file):
        print("  ERROR: NC file not found!")
        return None
    
    try:
        # Create dataset with shorter sequence for testing
        dataset = HydroBasinDataset(
            nc_file=nc_file,
            basin_id=basin_id,
            sequence_length=30,  # Shorter for testing
            prediction_horizon=1,
            normalize=True
        )
        
        print(f"  Dataset created successfully!")
        print(f"  Dataset size: {len(dataset)}")
        
        if len(dataset) == 0:
            print("  WARNING: Dataset has 0 sequences")
            return None
        
        # Get first sample
        sample = dataset[0]
        
        print(f"\n  Sample details:")
        print(f"    Features shape: {sample['features'].shape}")
        print(f"    Streamflow target shape: {sample['streamflow'].shape}")
        print(f"    Evapotranspiration target shape: {sample['evapotranspiration'].shape}")
        print(f"    Basin ID: {sample['basin_id']}")
        
        # Check feature dimensions
        print(f"\n  Feature information:")
        print(f"    Dynamic features: {dataset.dynamic_features}")
        print(f"    Static features: {dataset.static_features}")
        print(f"    Target features: {dataset.target_features}")
        print(f"    Total input features: {len(dataset.dynamic_features) + len(dataset.static_features)}")
        
        return dataset
        
    except Exception as e:
        print(f"  ERROR creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multiple_basins():
    """Test loading multiple basins with MultiBasinDataset"""
    
    print("\n" + "="*60)
    print("Testing multiple basins with MultiBasinDataset")
    print("="*60)
    
    # Find NC files
    data_dir = os.path.join(project_root, "output_all_basins")
    nc_files = []
    
    # Try to find a few basins
    for i in range(3):
        basin_num = 10023000 + i
        nc_file = os.path.join(data_dir, f"gage_{basin_num}.nc")
        if os.path.exists(nc_file):
            nc_files.append(nc_file)
    
    # If not found, try to find any NC files
    if len(nc_files) == 0:
        import glob
        all_nc_files = glob.glob(os.path.join(data_dir, "gage_*.nc"))
        nc_files = all_nc_files[:3]  # Use first 3
    
    if len(nc_files) == 0:
        print("ERROR: No NC files found!")
        return None
    
    # Extract basin IDs from file names
    basin_ids = []
    for nc_file in nc_files:
        basename = os.path.basename(nc_file)
        basin_id = basename.replace("gage_", "").replace(".nc", "")
        basin_ids.append(basin_id)
    
    print(f"\nFound {len(nc_files)} basins:")
    for basin_id, nc_file in zip(basin_ids, nc_files):
        print(f"  {basin_id}: {os.path.basename(nc_file)}")
    
    try:
        # Create MultiBasinDataset for training
        dataset = MultiBasinDataset(
            nc_files=nc_files,
            basin_ids=basin_ids,
            sequence_length=30,  # Shorter for testing
            prediction_horizon=1,
            normalize=True,
            train_ratio=0.7,
            val_ratio=0.15,
            mode='train'
        )
        
        print(f"\nMultiBasinDataset created successfully!")
        print(f"  Total sequences: {len(dataset)}")
        
        if len(dataset) == 0:
            print("  WARNING: Dataset has 0 sequences")
            return None
        
        # Get a sample
        sample = dataset[0]
        
        print(f"\n  Sample details:")
        print(f"    Features shape: {sample['features'].shape}")
        print(f"    Streamflow target shape: {sample['streamflow'].shape}")
        print(f"    Evapotranspiration target shape: {sample['evapotranspiration'].shape}")
        print(f"    Basin ID: {sample['basin_id']}")
        print(f"    Basin index: {sample['basin_idx']}")
        
        # Get dataset statistics
        stats = dataset.get_dataset_stats()
        print(f"\n  Dataset statistics:")
        print(f"    Number of basins: {stats['num_basins']}")
        print(f"    Sequence length: {stats['sequence_length']}")
        print(f"    Prediction horizon: {stats['prediction_horizon']}")
        
        return dataset
        
    except Exception as e:
        print(f"  ERROR creating MultiBasinDataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_dataloader_factory():
    """Test DataLoaderFactory"""
    
    print("\n" + "="*60)
    print("Testing DataLoaderFactory")
    print("="*60)
    
    # Find NC files
    data_dir = os.path.join(project_root, "output_all_basins")
    import glob
    all_nc_files = glob.glob(os.path.join(data_dir, "gage_*.nc"))
    
    if len(all_nc_files) == 0:
        print("ERROR: No NC files found!")
        return None
    
    # Use first 5 basins for testing
    nc_files = all_nc_files[:5]
    basin_ids = []
    for nc_file in nc_files:
        basename = os.path.basename(nc_file)
        basin_id = basename.replace("gage_", "").replace(".nc", "")
        basin_ids.append(basin_id)
    
    print(f"\nUsing {len(basin_ids)} basins for testing")
    
    # Create configuration
    config = {
        'sequence_length': 30,
        'prediction_horizon': 1,
        'dynamic_features': [
            'total_precipitation',
            'temperature', 
            'specific_humidity',
            'shortwave_radiation',
            'potential_energy'
        ],
        'static_features': [
            'elev_mean',
            'slope_mean',
            'area_gages2',
            'frac_forest',
            'lai_max'
        ],
        'target_features': ['streamflow', 'evapotranspiration'],
        'normalize': True,
        'train_ratio': 0.7,
        'val_ratio': 0.15
    }
    
    try:
        # Create dataloaders
        train_loader, val_loader, test_loader, metadata = DataLoaderFactory.create_dataloaders(
            nc_files=nc_files,
            basin_ids=basin_ids,
            config=config,
            batch_size=8,  # Small batch for testing
            num_workers=0,  # No multiprocessing for testing
            shuffle_train=True
        )
        
        print(f"\nDataLoaderFactory test successful!")
        print(f"  Input dimension: {metadata['input_dim']}")
        print(f"  Number of basins: {metadata['num_basins']}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Get a batch from training loader
        train_batch = next(iter(train_loader))
        
        print(f"\n  Batch details:")
        print(f"    Features shape: {train_batch['features'].shape}")
        print(f"    Streamflow shape: {train_batch['streamflow'].shape}")
        print(f"    Evapotranspiration shape: {train_batch['evapotranspiration'].shape}")
        print(f"    Basin indices shape: {train_batch['basin_idx'].shape}")
        print(f"    Number of basin IDs: {len(train_batch['basin_id'])}")
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"  ERROR in DataLoaderFactory: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    print("Starting comprehensive data loading tests...")
    print(f"Project root: {project_root}")
    
    # Test 1: Single basin
    print("\n" + "="*60)
    print("TEST 1: Single Basin Dataset")
    print("="*60)
    single_dataset = test_single_basin()
    
    # Test 2: Multiple basins
    print("\n" + "="*60)
    print("TEST 2: Multi Basin Dataset")
    print("="*60)
    multi_dataset = test_multiple_basins()
    
    # Test 3: DataLoaderFactory
    print("\n" + "="*60)
    print("TEST 3: DataLoaderFactory")
    print("="*60)
    train_loader, val_loader, test_loader = test_dataloader_factory()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    tests_passed = 0
    total_tests = 3
    
    if single_dataset is not None:
        print(f"✓ Test 1 (Single Basin): PASSED")
        tests_passed += 1
    else:
        print(f"✗ Test 1 (Single Basin): FAILED")
    
    if multi_dataset is not None:
        print(f"✓ Test 2 (Multi Basin): PASSED")
        tests_passed += 1
    else:
        print(f"✗ Test 2 (Multi Basin): FAILED")
    
    if train_loader is not None:
        print(f"✓ Test 3 (DataLoaderFactory): PASSED")
        tests_passed += 1
    else:
        print(f"✗ Test 3 (DataLoaderFactory): FAILED")
    
    print(f"\n{tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("\nAll tests passed successfully! Data loading is working correctly.")
    else:
        print("\nSome tests failed. Check the error messages above.")