"""
Main entry point for HydroMTL_CGC
Orchestrates the complete training and evaluation pipeline
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import numpy as np
import yaml

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Add the parent directory of mtl_cgc to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set environment variables for MKL
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Now import using absolute paths
try:
    from mtl_cgc.utils.config_parser import ConfigParser
    from mtl_cgc.utils.logger import setup_logger, setup_file_logger
    from mtl_cgc.data.data_loader import create_data_loaders
    from mtl_cgc.core.cgc_models.mtl_model import HydroMTL_CGC
    from mtl_cgc.core.training.trainer import HydroTrainer
    from mtl_cgc.core.evaluation.evaluator import HydroEvaluator
    from mtl_cgc.core.evaluation.visualizer import HydroVisualizer
except ImportError as e:
    # Fallback: try relative imports
    try:
        from .utils.config_parser import ConfigParser
        from .utils.logger import setup_logger, setup_file_logger
        from .data.data_loader import create_data_loaders
        from .core.cgc_models.mtl_model import HydroMTL_CGC
        from .core.training.trainer import HydroTrainer
        from .core.evaluation.evaluator import HydroEvaluator
        from .core.evaluation.visualizer import HydroVisualizer
    except ImportError as ie:
        raise ImportError(f"Cannot import required modules: {ie}\nOriginal error: {e}")


def setup_experiment(config_path: str, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Setup experiment from configuration file

    Args:
        config_path: Path to configuration file
        args: Command line arguments

    Returns:
        Complete configuration dictionary
    """
    # Load configuration
    config_parser = ConfigParser(config_path)
    config = config_parser.load_config(config_path)

    # Override config with command line arguments
    if args.experiment_name:
        config.experiment['name'] = args.experiment_name

    if args.data_root:
        config.data.data_root = args.data_root

    if args.batch_size:
        config.data.batch_size = args.batch_size

    if args.learning_rate:
        config.training.learning_rate = args.learning_rate

    if args.epochs:
        config.training.epochs = args.epochs

    # Add device assignment (args.device is already resolved to 'cuda' or 'cpu' in main)
    if args.device:
        config.training.device = args.device

    if hasattr(args, 'shared_experts') and args.shared_experts:
        config.model.cgc['shared_experts'] = args.shared_experts

    if hasattr(args, 'task_experts') and args.task_experts:
        config.model.cgc['task_experts'] = args.task_experts

    # Save updated configuration
    config_parser.save_config()

    # Setup logging
    log_level = getattr(logging, config.logging['level'].upper())
    setup_logger('__main__', log_level=log_level)

    if config.logging.get('file', True):
        log_file = Path(config.experiment['save_dir']) / 'experiment.log'
        setup_file_logger('__main__', log_file, log_level)

    logger = logging.getLogger(__name__)
    logger.info(f"Experiment setup complete: {config.experiment['name']}")
    logger.info(f"Save directory: {config.experiment['save_dir']}")

    return config


def load_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load and prepare data

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary containing data loaders and metadata
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading data...")

    from pathlib import Path
    data_root = Path(config.data.data_root)
    nc_files = list(data_root.glob("gage_*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No gage_*.nc files found in {data_root}")

    basin_ids = [f.stem.replace("gage_", "") for f in nc_files]
    logger.info(f"Found {len(basin_ids)} basin files: {basin_ids[:5]}...")

    data_loaders = create_data_loaders(config.data, basin_ids)

    if not data_loaders:
        raise ValueError("No data loaders created. Check data configuration.")

    logger.info(f"Created data loaders: {list(data_loaders.keys())}")
    for split in ['train', 'val', 'test']:
        loader = data_loaders.get(split)
        if loader is not None:
            logger.info(f"  {split}: {len(loader.dataset)} samples")

    basin_scalers = data_loaders.get('basin_scalers', None)
    if basin_scalers is None:
        logger.warning("No basin_scalers found in data_loaders. Inverse transform may fail.")

    return {
        'loaders': data_loaders,
        'basin_ids': basin_ids,
        'feature_names': config.data.static_features + config.data.dynamic_features,
        'target_names': [t['name'] for t in config.data.targets],
        'basin_scalers': basin_scalers
    }


def build_model(config: Dict[str, Any], device: str) -> HydroMTL_CGC:
    """
    Build the HydroMTL_CGC model

    Args:
        config: Configuration dictionary
        device: Device to place model on

    Returns:
        Initialized model
    """
    logger = logging.getLogger(__name__)
    logger.info("Building model...")

    # Construct config dictionary expected by the model
    config_dict = {
        'data': config.data.__dict__,
        'experiment': config.experiment
    }
    # Promote all model fields to top-level (e.g., task_towers, cgc, encoder, physics_constraints)
    config_dict.update(config.model.__dict__)

    # Build model
    model = HydroMTL_CGC(config_dict)

    # Print model summary
    summary = model.get_model_summary()
    logger.info(f"Model summary:")
    logger.info(f"  Total parameters: {summary['total_parameters']:,}")
    logger.info(f"  Trainable parameters: {summary['trainable_parameters']:,}")
    logger.info(f"  Encoder type: {summary['encoder_type']}")
    logger.info(f"  Number of tasks: {summary['num_tasks']}")
    if 'cgc_shared_experts' in summary:
        logger.info(f"  CGC shared experts: {summary['cgc_shared_experts']}")
    if 'cgc_task_experts' in summary:
        logger.info(f"  CGC task experts: {summary['cgc_task_experts']}")
    if 'physics_constraints_enabled' in summary:
        logger.info(f"  Physics constraints: {summary['physics_constraints_enabled']}")

    return model


def train_model(model: HydroMTL_CGC, data: Dict[str, Any],
                config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train the model

    Args:
        model: HydroMTL_CGC model
        data: Data dictionary containing loaders
        config: Configuration dictionary

    Returns:
        Training history
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")

    # Setup device
    device = torch.device(getattr(config.training, 'device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    # ========== Normalization check (streaming, full dataset) ==========
    print("\n"+" Checking normalization of training targets ".center(80, "="))
    train_loader = data['loaders']['train']
    sum_sf = 0.0; sum_sf_sq = 0.0; cnt_sf = 0
    sum_et = 0.0; sum_et_sq = 0.0; cnt_et = 0
    for batch in train_loader:
        sf = batch['streamflow'].cpu().numpy().flatten()
        et = batch['evapotranspiration'].cpu().numpy().flatten()
        m_sf = ~np.isnan(sf)
        m_et = ~np.isnan(et)
        if m_sf.any():
            v = sf[m_sf]
            sum_sf += v.sum()
            sum_sf_sq += (v ** 2).sum()
            cnt_sf += len(v)
        if m_et.any():
            v = et[m_et]
            sum_et += v.sum()
            sum_et_sq += (v ** 2).sum()
            cnt_et += len(v)
    if cnt_sf > 0:
        mean_sf = sum_sf / cnt_sf
        std_sf = np.sqrt(sum_sf_sq / cnt_sf - mean_sf ** 2)
        print(f"Train streamflow (standardized) - mean: {mean_sf:.4f}, std: {std_sf:.4f}")
    if cnt_et > 0:
        mean_et = sum_et / cnt_et
        std_et = np.sqrt(sum_et_sq / cnt_et - mean_et ** 2)
        print(f"Train ET (standardized) - mean: {mean_et:.4f}, std: {std_et:.4f}")
    print("=" * 80 + "\n")
    # ========== End normalization check ==========

    # Setup trainer
    trainer = HydroTrainer(
        model=model,
        config=config,
        device=device,
        use_wandb=config.logging.get('wandb', False),
        basin_scalers=data.get('basin_scalers')
    )

    # Train model
    train_loader = data['loaders']['train']
    val_loader = data['loaders']['val']

    history = trainer.fit(train_loader, val_loader)

    logger.info("Model training complete")

    return {
        'history': history,
        'trainer': trainer,
        'device': device
    }


def evaluate_model(model: HydroMTL_CGC, data: Dict[str, Any],
                   training_result: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate the trained model

    Args:
        model: Trained model
        data: Data dictionary
        training_result: Training results
        config: Configuration dictionary

    Returns:
        Evaluation results
    """
    logger = logging.getLogger(__name__)
    logger.info("Evaluating model...")

    # Setup evaluator and visualizer
    save_dir = Path(config.experiment['save_dir']) / 'evaluation'
    save_dir.mkdir(parents=True, exist_ok=True)
    evaluator = HydroEvaluator(config.evaluation, save_dir=str(save_dir))
    visualizer = HydroVisualizer(save_dir=str(save_dir))

    # Generate predictions
    trainer = training_result['trainer']
    test_loader = data['loaders']['test']

    logger.info("Generating predictions...")
    predictions = trainer.predict(test_loader, return_analysis=True)

    # Get ground truth
    logger.info("Collecting ground truth...")
    all_targets = {name: [] for name in data['target_names']}

    for features, targets in test_loader:
        for name in data['target_names']:
            if name in targets:
                all_targets[name].append(targets[name].numpy())

    # Concatenate targets
    for name in all_targets:
        if all_targets[name]:
            all_targets[name] = np.concatenate(all_targets[name], axis=0)

    # Compute metrics
    logger.info("Computing metrics...")
    metrics = evaluator.compute_all_metrics(
        predictions=predictions,
        targets=all_targets,
        basin_ids=np.repeat(data['basin_ids'], len(test_loader.dataset) // len(data['basin_ids'])),
        save_results=True
    )

    # Generate detailed analysis
    logger.info("Performing detailed analysis...")
    analysis = evaluator.analyze_predictions(
        predictions=predictions,
        targets=all_targets,
        task_names=data['target_names']
    )

    # Generate and save report
    report = evaluator.generate_report(metrics, analysis)
    report_file = save_dir / 'evaluation_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)

    logger.info(f"Evaluation report saved to {report_file}")
    print("\n" + report)

    # Generate visualizations
    logger.info("Creating visualizations...")

    # Plot predictions vs observations
    visualizer.plot_predictions_vs_observations(
        predictions=predictions,
        observations=all_targets,
        task_names=data['target_names'],
        basin_ids=np.repeat(data['basin_ids'], len(test_loader.dataset) // len(data['basin_ids'])),
        save=True
    )

    # Plot training history
    visualizer.plot_training_history(
        history=training_result['history'],
        save=True
    )

    # Plot gate analysis if available
    if 'gate_analysis' in predictions:
        visualizer.plot_gate_analysis(
            gate_analysis=predictions['gate_analysis'],
            task_names=data['target_names'],
            save=True
        )

    logger.info("Evaluation complete")

    return {
        'metrics': metrics,
        'analysis': analysis,
        'predictions': predictions,
        'targets': all_targets
    }


def save_final_results(config: Dict[str, Any],
                       training_result: Dict[str, Any],
                       evaluation_result: Dict[str, Any]) -> None:
    """
    Save final experiment results

    Args:
        config: Configuration dictionary
        training_result: Training results
        evaluation_result: Evaluation results
    """
    import pickle

    save_dir = Path(config.experiment['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save final model
    model_file = save_dir / 'final_model.pth'
    torch.save(training_result['trainer'].model.state_dict(), model_file)

    # Save training history
    history_file = save_dir / 'training_history.pkl'
    with open(history_file, 'wb') as f:
        pickle.dump(training_result['history'], f)

    # Save evaluation results
    eval_file = save_dir / 'evaluation_results.pkl'
    with open(eval_file, 'wb') as f:
        pickle.dump(evaluation_result, f)

    # Save configuration
    config_file = save_dir / 'final_config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger = logging.getLogger(__name__)
    logger.info(f"Final results saved to {save_dir}")


def run_experiment(config_path: str, args: argparse.Namespace) -> None:
    """
    Run complete experiment pipeline

    Args:
        config_path: Path to configuration file
        args: Command line arguments
    """
    print("=" * 60)
    print("DEBUG: Entered run_experiment")
    print("=" * 60)
    try:
        # 1. Setup experiment
        config = setup_experiment(config_path, args)

        # 2. Load data
        data = load_data(config)

        # 3. Build model
        model = build_model(config, args.device if hasattr(args, 'device') else 'auto')

        # 4. Train model
        training_result = train_model(model, data, config)

        # 5. Evaluate model
        evaluation_result = evaluate_model(model, data, training_result, config)

        # 6. Save final results
        save_final_results(config, training_result, evaluation_result)

        print(f"\n{'='*60}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {config.experiment['save_dir']}")
        print(f"{'='*60}")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Experiment failed with error: {e}")
        raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='HydroMTL_CGC: Multi-task hydrological modeling')

    # Configuration
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to configuration file')

    # Experiment
    parser.add_argument('--experiment_name', type=str,
                       help='Name of the experiment')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate', 'predict'],
                       help='Mode to run: train, evaluate, or predict')

    # Data
    parser.add_argument('--data_root', type=str,
                       help='Root directory of data')
    parser.add_argument('--batch_size', type=int,
                       help='Batch size for training')

    # Model
    parser.add_argument('--shared_experts', type=int,
                       help='Number of shared experts in CGC')
    parser.add_argument('--task_experts', type=int, nargs='+',
                       help='Number of task-specific experts for each task')

    # Training
    parser.add_argument('--learning_rate', type=float,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: cuda, cpu, or auto')

    # Evaluation
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint for evaluation')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Run experiment
    run_experiment(args.config, args)


if __name__ == '__main__':
    main()