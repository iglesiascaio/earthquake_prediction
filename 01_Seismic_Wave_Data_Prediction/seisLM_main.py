"""
Main script for earthquake magnitude prediction project.
This script orchestrates the model training and evaluation process.
"""
print("Start")
import os
import yaml
import torch
import argparse
from datetime import datetime
import sys
import importlib

import sys, pathlib, os
INNER = pathlib.Path("~/toto/toto").expanduser()
if str(INNER) not in sys.path:
    sys.path.insert(0, str(INNER))
os.environ["HF_HUB_OFFLINE"] = "1"     # stay offline


# --- locate 02_Functions and add to path --------------------------------
from pathlib import Path
NB_ROOT = Path.cwd()

FUNC_DIR  = NB_ROOT / "02_Functions"                          # ./02_Functions
SEISLM_INNER = FUNC_DIR / "seisLM" / "seisLM"                 # ./02_Functions/seisLM/seisLM

for p in (FUNC_DIR, SEISLM_INNER):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

print("Added to sys.path:")
for p in (FUNC_DIR, SEISLM_INNER):
    print("  •", p if p.is_dir() else "(missing)", p.is_dir())

# Make outer name 'seisLM' point to the inner package that *does* have .model
if "seisLM" not in sys.modules:
    inner_pkg = importlib.import_module("seisLM.seisLM")
    sys.modules["seisLM"] = inner_pkg

TOTO_INNER = pathlib.Path.home() / "toto" / "toto"
if TOTO_INNER.is_dir() and str(TOTO_INNER) not in sys.path:
    sys.path.insert(0, str(TOTO_INNER))

# alias:  sys.modules["model"]  ->  toto.model  (and its sub-modules)
if "model" not in sys.modules:
    sys.modules["model"] = importlib.import_module("toto.model")

print("✓ aliased toto.model  ➜  top-level 'model'")

# Import project modules
from Model_Trainer import ModelTrainer
from Model_Evaluator import ModelEvaluator
from Dataset_creation import create_train_dataset_new, get_data_loader
print("Finished Importing")

# find config.yaml right next to the notebook / script
PROJECT_DIR = Path.cwd()                           # current working dir
CFG_CANDIDATE = PROJECT_DIR / "config.yaml"

# if we launched one level up, fall back to sub-dir path
if not CFG_CANDIDATE.is_file():
    CFG_CANDIDATE = PROJECT_DIR / "01_Seismic_Wave_Data_Prediction" / "config.yaml"
DEFAULT_CFG = str(CFG_CANDIDATE)


def parse_args_jupyter_safe() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training / evaluation")
    parser.add_argument(
        "--config",
        default=DEFAULT_CFG,         # ← here
        help="Path to YAML config file")
    parser.add_argument(
        "--mode", choices=["train", "evaluate", "both"], default="both")
    parser.add_argument("--checkpoint", default=None)

    # Ignore stray notebook flags
    if "ipykernel_launcher" in sys.argv[0]:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()
    return args


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    #print(config)
    return config


def main():
    """Main function to orchestrate training and evaluation."""
    # Parse arguments and load configuration
    args = parse_args_jupyter_safe()
    config = load_config(args.config)
    
    # Setup output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['paths']['output_dir'], f"{config['model']['model_type']}_{config['model']['aggregation_type']}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration to output directory
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
        
    # Check if using tabular features
    use_tabular_features = config['model'].get('use_tabular_features', False)
    
    # Create dataset and data loader
    #print("Creating dataset...")
    train_data = create_train_dataset_new(
        config['paths']['earthquake_parquet'],
        config['paths']['combined_stream_dir']
    )
    
    data_loader = get_data_loader(
        train_data,
        config['paths']['earthquake_parquet'],  # For tabular features if needed
        window_size_days=config['data']['window_size_days'],
        batch_size=config['training']['batch_size'],
        shuffle=True,
        use_tabular_features=use_tabular_features,
        downsampling_rate = config['data']['downsampling_rate']
    )
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train model if requested
    if args.mode in ['train', 'both']:
        trainer = ModelTrainer(config, output_dir, device)
        model, checkpoint_path = trainer.train(data_loader)
    
    # Evaluate model if requested
    if args.mode in ['evaluate', 'both']:
        # If we're only evaluating, load the model from checkpoint
        if args.mode == 'evaluate':
            checkpoint_path = args.checkpoint
            if checkpoint_path is None:
                print("Error: Checkpoint path required for evaluation mode")
                return
            
            trainer = ModelTrainer(config, output_dir, device)
            model = trainer.load_model(checkpoint_path)
        
        # Setup evaluator
        evaluator = ModelEvaluator(
            model=model,
            data_loader=data_loader,
            device=device,
            class_names=config['model']['class_names'],
            output_dir=os.path.join(output_dir, 'evaluation')
        )
        
        # Run evaluation
        results = evaluator.evaluate()
        print(f"Evaluation completed. Results saved to {os.path.join(output_dir, 'evaluation')}")
    
    print("Done!")

    
def only_evaluate():
    checkpoint_path= '/home/gridsan/mknuth/01_Seismic_Wave_Data_Prediction/03_Results/seislm_toto_20250807_114200/checkpoints/checkpoint_epoch1.pth'
    args = parse_args_jupyter_safe()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"Using device: {device}")
    # Setup output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['paths']['output_dir'], f"{config['model']['model_type']}_{config['model']['aggregation_type']}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    trainer = ModelTrainer(config, output_dir, device)

    model = trainer.load_model(checkpoint_path)

    # Create dataset and data loader
    #print("Creating dataset...")
    train_data = create_train_dataset_new(
            config['paths']['earthquake_parquet'],
            config['paths']['combined_stream_dir']
        )
    use_tabular_features = config['model'].get('use_tabular_features', False)
    data_loader = get_data_loader(
            train_data,
            config['paths']['earthquake_parquet'],  # For tabular features if needed
            window_size_days=config['data']['window_size_days'],
            batch_size=config['training']['batch_size'],
            shuffle=True,
            use_tabular_features=use_tabular_features,
            downsampling_rate = config['data']['downsampling_rate']
        )     
    # Setup evaluator
    evaluator = ModelEvaluator(
        model=model,
        data_loader=data_loader,
        device=device,
        class_names=config['model']['class_names'],
        output_dir=os.path.join(output_dir, 'evaluation')
    )

    # Run evaluation
    results = evaluator.evaluate()
    print(f"Evaluation completed. Results saved to {os.path.join(output_dir, 'evaluation')}")


if __name__ == "__main__":
    #main()
    only_evaluate()