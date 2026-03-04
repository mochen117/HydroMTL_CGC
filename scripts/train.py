"""
Training script for HydroMTL_CGC
Simplified script for training only
"""
import os
import sys
import ctypes
import argparse
from pathlib import Path
import multiprocessing as mp

# Force LD_LIBRARY_PATH to point to conda environment's lib directory
conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix:
    lib_path = os.path.join(conda_prefix, 'lib')
    os.environ['LD_LIBRARY_PATH'] = lib_path + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    print(f"[ENV] LD_LIBRARY_PATH set to: {os.environ['LD_LIBRARY_PATH']}")
else:
    print("[ERROR] CONDA_PREFIX not set. Are you in the conda environment?")
    sys.exit(1)

# Preload conda's libstdc++.so.6 before any other imports
libstdcxx_path = os.path.join(conda_prefix, 'lib', 'libstdc++.so.6')
try:
    ctypes.CDLL(libstdcxx_path, mode=ctypes.RTLD_GLOBAL)
    print(f"[PRELOAD] Successfully preloaded {libstdcxx_path}")
except Exception as e:
    print(f"[PRELOAD] Failed: {e}")

# Set multiprocessing start method
try:
    mp.set_start_method('fork', force=True)
except RuntimeError:
    pass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from mtl_cgc.main import setup_experiment, load_data, build_model, train_model


def train_only(config_path: str, args: argparse.Namespace):
    """
    Train model only (without full evaluation)

    Args:
        config_path: Path to configuration file
        args: Command line arguments
    """
    # Setup experiment
    config = setup_experiment(config_path, args)

    # Load data
    data = load_data(config)

    # Build model
    model = build_model(config, args.device if hasattr(args, 'device') else 'auto')

    # Train model
    training_result = train_model(model, data, config)

    print(f"\nTraining complete!")
    print(f"Results saved to: {config.experiment['save_dir']}")

    return training_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HydroMTL_CGC model')

    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--experiment_name', type=str,
                       help='Name of the experiment')
    parser.add_argument('--data_root', type=str,
                       help='Root directory of data')
    parser.add_argument('--batch_size', type=int,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: cuda, cpu, or auto')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        import torch
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_only(args.config, args)