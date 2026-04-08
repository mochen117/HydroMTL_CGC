import os
import sys
import ctypes
import argparse
import logging
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
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import yaml

from mtl_cgc.utils.config_parser import ConfigParser
from mtl_cgc.utils.logger import setup_logger
from mtl_cgc.data.data_loader import create_data_loaders
from mtl_cgc.core.baseline.baseline_mtl import HardSharingLSTM
from mtl_cgc.core.training.trainer import HydroTrainer

def main():
    parser = argparse.ArgumentParser(description='Train baseline hard-sharing LSTM model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=None, help='Override training epochs')
    parser.add_argument('--learning_rate', type=float, default=None, help='Override learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use: cuda, cpu, auto')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Load configuration
    config_parser = ConfigParser(args.config)
    config = config_parser.load_config(args.config)

    # Override config if provided
    if args.epochs:
        config.training.epochs = args.epochs
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Load data: get basin_ids from data_root
    from pathlib import Path
    data_root = Path(config.data.data_root)
    nc_files = list(data_root.glob("gage_*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No gage_*.nc files found in {data_root}")
    basin_ids = [f.stem.replace("gage_", "") for f in nc_files]
    logger.info(f"Found {len(basin_ids)} basins")

    # Create data loaders (pass basin_ids)
    data_loaders = create_data_loaders(config.data, basin_ids)
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']

    # Determine input dimension from a sample batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch['features'].shape[-1]
    logger.info(f"Input dimension: {input_dim}")

    # Build baseline model
    model = HardSharingLSTM(
        input_dim=input_dim,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2
    )
    logger.info(f"Baseline model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Setup trainer (reuses existing HydroTrainer)
    trainer = HydroTrainer(
        model=model,
        config=config,
        device=device,
        use_wandb=False
    )

    # Train
    history = trainer.fit(train_loader, val_loader)

    # Save final model
    save_dir = Path(config.experiment['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / 'baseline_model.pth')
    logger.info(f"Model saved to {save_dir / 'baseline_model.pth'}")

    # Print final validation metrics
    final_metrics = history['val_metrics'][-1]
    logger.info("Final validation metrics:")
    for key, value in final_metrics.items():
        logger.info(f"  {key}: {value:.4f}")

if __name__ == '__main__':
    main()