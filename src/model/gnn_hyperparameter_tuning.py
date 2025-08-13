#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GNN Hyperparameter Tuning Script for Earthquake Prediction

This script systematically tests different GNN architectures and configurations:
- 25 different model variations
- 3 embedding scenarios per variation
- Incremental result saving to prevent data loss
- Comprehensive logging for supercloud execution
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.loader import DataLoader

# Force printing and disable buffering for supercloud
import builtins
original_print = builtins.print
def force_print(*args, **kwargs):
    original_print(*args, **kwargs)
    sys.stdout.flush()
builtins.print = force_print

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .utils import load_and_split_data, evaluate
import src.model.utils.gnn_helper as gh

warnings.filterwarnings("ignore")

# Force CUDA to be deterministic for reproducible results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import random
random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"Random seed set to: {RANDOM_SEED}")
print(f"CUDA deterministic: {torch.backends.cudnn.deterministic}")

# ==========================================================
# 1. MODEL ARCHITECTURES
# ==========================================================

class StationGNN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, n_layers: int = 3, n_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.convs = torch.nn.ModuleList([SAGEConv(in_dim, hidden)])
        self.convs.extend([SAGEConv(hidden, hidden) for _ in range(n_layers - 2)])
        self.convs.append(SAGEConv(hidden, hidden))
        
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, data):
        x, edge = data.x, data.edge_index
        for conv in self.convs:
            x = torch.relu(conv(x, edge))
        return self.head(x)


class StationGAT(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, n_layers: int = 3, n_heads: int = 4, 
                 n_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        assert n_layers >= 2, "Need at least 2 GAT layers"

        # First layer: in_dim → hidden
        self.convs = torch.nn.ModuleList([
            GATConv(in_dim, hidden, heads=n_heads, concat=True, dropout=dropout)
        ])

        # Middle layers keep same hidden size
        for _ in range(n_layers - 2):
            self.convs.append(
                GATConv(hidden * n_heads, hidden, heads=n_heads, concat=True, dropout=dropout)
            )

        # Last GAT layer keeps dimension but no head concatenation
        self.convs.append(
            GATConv(hidden * n_heads, hidden, heads=1, concat=False, dropout=dropout)
        )

        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, data):
        x, edge = data.x, data.edge_index
        for conv in self.convs:
            x = torch.relu(conv(x, edge))
        return self.head(x)


# ==========================================================
# 2. HYPERPARAMETER CONFIGURATIONS
# ==========================================================

def generate_hyperparameter_configs() -> List[Dict[str, Any]]:
    """Generate 25 different hyperparameter configurations."""
    
    configs = []
    
    # BASELINE: Original configuration from gnn_experiment.py
    baseline_config = {
        "config_id": "baseline",
        "model_type": "StationGNN",
        "hidden": 16,
        "n_layers": 3,
        "lr": 1e-3,
        "batch_size": 16,
        "dropout": 0.1,
        "description": "Original baseline from gnn_experiment.py - EXACT COPY"
    }
    configs.append(baseline_config)
    
    # Base configurations
    base_configs = [
        # Small models (good for both GNN and GAT)
        {"hidden": 32, "n_layers": 2, "lr": 1e-3, "batch_size": 32, "dropout": 0.1},
        {"hidden": 64, "n_layers": 2, "lr": 1e-3, "batch_size": 32, "dropout": 0.1},
        {"hidden": 32, "n_layers": 3, "lr": 1e-3, "batch_size": 32, "dropout": 0.1},
        
        # Medium models (GNN-friendly)
        {"hidden": 64, "n_layers": 3, "lr": 1e-3, "batch_size": 16, "dropout": 0.1},
        {"hidden": 128, "n_layers": 3, "lr": 1e-3, "batch_size": 16, "dropout": 0.1},
        {"hidden": 64, "n_layers": 4, "lr": 1e-3, "batch_size": 16, "dropout": 0.1},
        {"hidden": 128, "n_layers": 4, "lr": 1e-3, "batch_size": 16, "dropout": 0.1},
        
        # Large models (GNN-friendly, but may be too big for GAT)
        {"hidden": 128, "n_layers": 3, "lr": 5e-4, "batch_size": 8, "dropout": 0.2},
        {"hidden": 256, "n_layers": 3, "lr": 5e-4, "batch_size": 8, "dropout": 0.2},
        {"hidden": 128, "n_layers": 4, "lr": 5e-4, "batch_size": 8, "dropout": 0.2},
        
        # High learning rate models
        {"hidden": 64, "n_layers": 3, "lr": 5e-3, "batch_size": 16, "dropout": 0.1},
        {"hidden": 128, "n_layers": 3, "lr": 5e-3, "batch_size": 16, "dropout": 0.1},
        
        # Low learning rate models
        {"hidden": 64, "n_layers": 3, "lr": 1e-4, "batch_size": 16, "dropout": 0.1},
        {"hidden": 128, "n_layers": 3, "lr": 1e-4, "batch_size": 16, "dropout": 0.1},
        
        # High dropout models
        {"hidden": 64, "n_layers": 3, "lr": 1e-3, "batch_size": 16, "dropout": 0.3},
        {"hidden": 128, "n_layers": 3, "lr": 1e-3, "batch_size": 16, "dropout": 0.3},
        
        # GAT-specific configurations (optimized for attention models)
        {"hidden": 32, "n_layers": 3, "lr": 1e-3, "batch_size": 32, "dropout": 0.1, "n_heads": 2},
        {"hidden": 64, "n_layers": 3, "lr": 1e-3, "batch_size": 16, "dropout": 0.1, "n_heads": 2},
        {"hidden": 64, "n_layers": 3, "lr": 1e-3, "batch_size": 16, "dropout": 0.1, "n_heads": 4},
        {"hidden": 64, "n_layers": 4, "lr": 1e-3, "batch_size": 16, "dropout": 0.1, "n_heads": 2},
        {"hidden": 64, "n_layers": 4, "lr": 1e-3, "batch_size": 16, "dropout": 0.1, "n_heads": 4},
        
        # GAT with different learning rates (attention models can be sensitive to LR)
        {"hidden": 64, "n_layers": 3, "lr": 5e-4, "batch_size": 16, "dropout": 0.1, "n_heads": 4},
        {"hidden": 64, "n_layers": 3, "lr": 2e-3, "batch_size": 16, "dropout": 0.1, "n_heads": 4},
        
        # Extreme configurations (GNN only - too big for GAT)
        {"hidden": 512, "n_layers": 5, "lr": 1e-4, "batch_size": 4, "dropout": 0.3},
        {"hidden": 32, "n_layers": 6, "lr": 1e-2, "batch_size": 64, "dropout": 0.05},
    ]
    
    # Create configurations with different model types and feature selections
    for i, base_config in enumerate(base_configs):
        # StationGNN configurations (can handle larger models)
        configs.append({
            "config_id": f"config_{i+1:02d}_GNN",
            "model_type": "StationGNN",
            **base_config
        })
        
        # StationGAT configurations (optimized for attention models)
        gat_config = base_config.copy()
        if "n_heads" not in gat_config:
            gat_config["n_heads"] = 4  # Default to 4 attention heads
        
        # Skip extremely large configurations for GAT (too many parameters)
        total_params_estimate = gat_config["hidden"] * gat_config["n_layers"] * gat_config["n_heads"]
        if total_params_estimate <= 10000:  # Reasonable parameter limit for GAT
            configs.append({
                "config_id": f"config_{i+1:02d}_GAT",
                "model_type": "StationGAT",
                **gat_config
            })
        else:
            print(f"Skipping GAT config {i+1}: too many parameters ({total_params_estimate})")
    
    # Ensure we have exactly 25 configurations (baseline + 24 others)
    configs = configs[:25]
    
    return configs


# ==========================================================
# 3. TRAINING FUNCTIONS
# ==========================================================

def logits_and_labels(loader):
    """Helper function to extract logits and labels from dataloader (same as gnn_experiment.py)."""
    model.eval()
    outs, ys = [], []
    with torch.no_grad():
        for d in loader:
            d = d.to(DEVICE)
            outs.append(model(d).cpu().numpy())
            ys.append(d.y.cpu().numpy())
    return np.concatenate(outs), np.concatenate(ys)


# ==========================================================
# 4. MAIN HYPERPARAMETER TUNING LOOP
# ==========================================================

def main():
    # Configuration
    DATA = {
        "daily": "./data/features/earthquake_features.parquet",
        "seismic": "./data/embeddings/Embeddings_192142.pkl",
        "meta": "./data/raw/earthquake_data.parquet",
    }
    
    TOP_FEATS = [
        "time_since_class_3",
        "rolling_T_value", 
        "daily_count_30d_sum",
        "daily_b_value",
        "rolling_dE_half",
        "daily_etas_intensity",
        "time_since_class_2",
        "daily_count_7d_sum",
    ]
    
    CUT = "2024-01-01"
    RADIUS = 100.0
    EPOCHS = 60
    PATIENCE = 12
    
    # TEST MODE: Set to True for quick testing with only 2 models
    TEST_MODE = False  # Change to False for full 25-model run
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = Path(f"results/gnn_hyperparameter_tuning_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = {
        "timestamp": timestamp,
        "data_paths": DATA,
        "top_features": TOP_FEATS,
        "cutoff_date": CUT,
        "radius_km": RADIUS,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "device": str(DEVICE),
        "test_mode": TEST_MODE
    }
    
    with open(results_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Starting GNN hyperparameter tuning experiment")
    print(f"Results will be saved to: {results_dir}")
    print(f"Configuration saved to: {results_dir}/experiment_config.json")
    print(f"Device: {DEVICE}")
    print(f"TEST MODE: {TEST_MODE}")
    print("=" * 80)
    
    # Generate hyperparameter configurations
    configs = generate_hyperparameter_configs()
    if TEST_MODE:
        # In test mode, include both GNN and GAT to verify both work
        # Take first config (baseline GNN) + first GAT config
        baseline_config = configs[0]  # baseline GNN
        first_gat_config = None
        
        # Find the first GAT configuration
        for config in configs:
            if config["model_type"] == "StationGAT":
                first_gat_config = config
                break
        
        if first_gat_config:
            configs = [baseline_config, first_gat_config]
            print(f"TEST MODE: Using {len(configs)} configurations")
            print(f"  - {baseline_config['config_id']} ({baseline_config['model_type']})")
            print(f"  - {first_gat_config['config_id']} ({first_gat_config['model_type']})")
        else:
            # Fallback if no GAT config found
            configs = configs[:2]
            print(f"TEST MODE: No GAT config found, using first 2 configs")
    else:
        print(f"Full run: Using {len(configs)} configurations")
    
    # Initialize results storage
    results = []
    partial_results_file = results_dir / "partial_results.csv"
    
    # Test each configuration
    for config_idx, hp_config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Testing Configuration {config_idx + 1}/{len(configs)}: {hp_config['config_id']}")
        print(f"Model: {hp_config['model_type']}")
        print(f"Hidden: {hp_config['hidden']}, Layers: {hp_config['n_layers']}")
        print(f"LR: {hp_config['lr']}, Batch: {hp_config['batch_size']}, Dropout: {hp_config['dropout']}")
        print(f"{'='*60}")
        
        # Test embedding scenarios (fast ones first, slow ones last)
        embedding_scenarios = [
            {"merge_embeddings": True, "keep_embeddings": False, "name": "no_embeddings_limited_data"},
            {"merge_embeddings": True, "keep_embeddings": True, "name": "embeddings_limited_data"},
            {"merge_embeddings": False, "keep_embeddings": False, "name": "no_embeddings_full_data"}
        ]
        
        # In test mode, skip the slow full_data scenario
        if TEST_MODE:
            embedding_scenarios = embedding_scenarios[:2]  # Only first 2 (fast scenarios)
            print(f"TEST MODE: Skipping slow 'no_embeddings_full_data' scenario")
        
        # Explicitly order scenarios (fast first, slow last) instead of sorting
        # This ensures no_embeddings_full_data runs last
        if not TEST_MODE:
            # Reorder: fast scenarios first, slow scenario last
            fast_scenarios = [s for s in embedding_scenarios if s["merge_embeddings"]]
            slow_scenarios = [s for s in embedding_scenarios if not s["merge_embeddings"]]
            embedding_scenarios = fast_scenarios + slow_scenarios
        
        print(f"  Scenario execution order (fast → slow):")
        for i, scenario in enumerate(embedding_scenarios):
            speed = "⚡ FAST" if scenario["merge_embeddings"] else "🐌 SLOW"
            print(f"    {i+1}. {scenario['name']} ({speed})")
        
        # Test both feature selection strategies for each scenario
        feature_strategies = [
            {"name": "top_features", "select_top_feats": True, "top_features": TOP_FEATS},
            {"name": "all_features", "select_top_feats": False, "top_features": None}
        ]
        
        for scenario in embedding_scenarios:
            for feature_strategy in feature_strategies:
                scenario_name = f"{scenario['name']}_{feature_strategy['name']}"
                print(f"\n--- Testing scenario: {scenario_name} ---")
                print(f"  Feature strategy: {feature_strategy['name']}")
                
                try:
                    # Load data for this scenario and feature strategy
                    print("Loading data...")
                    _, _, _, _, df_full = load_and_split_data(
                        DATA["daily"],
                        DATA["seismic"],
                        select_top_feats=feature_strategy["select_top_feats"],
                        top_features=feature_strategy["top_features"],
                        merge_embeddings=scenario["merge_embeddings"],
                        keep_embeddings=scenario["keep_embeddings"],
                        cutoff_date=CUT,
                    )
                    
                    # Get feature columns based on strategy
                    if feature_strategy["select_top_feats"]:
                        if scenario["merge_embeddings"] and scenario["keep_embeddings"]:
                            feats = TOP_FEATS + [c for c in df_full.columns if c.startswith("emb_")]
                        else:
                            feats = TOP_FEATS
                    else:
                        # Use all available features (excluding target columns)
                        exclude_cols = ["max_mag_next_30d", "target_class", "date", "station_code"]
                        feats = [c for c in df_full.columns if c not in exclude_cols]
                    
                    print(f"Feature dimension: {len(feats)}")
                    print(f"Feature strategy: {feature_strategy['name']}")
                    if feature_strategy["select_top_feats"]:
                        print(f"Using top {len(feats)} features")
                    else:
                        print(f"Using all {len(feats)} available features")
                    
                    # For baseline, ensure we're using the same features as original
                    if hp_config["config_id"] == "baseline":
                        print(f"BASELINE: Using {len(feats)} features")
                        if scenario["name"] == "no_embeddings_limited_data" and feature_strategy["name"] == "top_features":
                            print(f"BASELINE: This should match gnn_experiment.py exactly")
                            print(f"BASELINE: TOP_FEATS = {TOP_FEATS}")
                            print(f"BASELINE: merge_embeddings = {scenario['merge_embeddings']}")
                            print(f"BASELINE: keep_embeddings = {scenario['keep_embeddings']}")
                    
                    # Build graph
                    print("Building graph...")
                    meta = gh.load_station_metadata(DATA["meta"])
                    edge_index = gh.build_radius_graph(
                        meta.station_latitude.values, 
                        meta.station_longitude.values, 
                        RADIUS
                    )
                    
                    # Create dataloaders
                    print("Creating dataloaders...")
                    dl_tr, dl_te, n_classes, med, mu, sig = gh.make_dataloaders(
                        df_full, meta, feats, edge_index, 
                        batch_size=hp_config["batch_size"], 
                        cutoff=CUT
                    )
                    
                    # Create model
                    if hp_config["model_type"] == "StationGNN":
                        model = StationGNN(
                            in_dim=len(feats) * 2,  # t-1 + t features
                            hidden=hp_config["hidden"],
                            n_layers=hp_config["n_layers"],
                            n_classes=n_classes,
                            dropout=hp_config["dropout"]
                        )
                    else:  # StationGAT
                        n_heads = hp_config.get("n_heads", 4)
                        model = StationGAT(
                            in_dim=len(feats) * 2,  # t-1 + t features
                            hidden=hp_config["hidden"],
                            n_layers=hp_config["n_layers"],
                            n_heads=n_heads,
                            n_classes=n_classes,
                            dropout=hp_config["dropout"]
                        )
                    
                    model = model.to(DEVICE)
                    
                    # Setup training
                    optimizer = torch.optim.Adam(model.parameters(), lr=hp_config["lr"])
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="min", patience=5, factor=0.5, verbose=False
                    )
                    loss_fn = torch.nn.CrossEntropyLoss()
                    
                    # Training loop (exactly as in gnn_experiment.py)
                    print("Training model...")
                    start_time = time.time()
                    
                    best_loss = float("inf")  # lowest validation CE seen so far
                    wait, patience_epochs = 0, PATIENCE  # stop if no improvement for `patience` epochs
                    best_state = None  # keep the weights that achieved best_loss
                    
                    def run_epoch(loader, train: bool):
                        model.train() if train else model.eval()
                        tot, nodes = 0.0, 0
                        for data in loader:
                            data = data.to(DEVICE)
                            if train:
                                optimizer.zero_grad()
                            out = model(data)
                            loss = loss_fn(out, data.y)
                            if train:
                                loss.backward()
                                optimizer.step()
                            tot += loss.item() * data.num_nodes
                            nodes += data.num_nodes
                        return tot / nodes  # average cross-entropy per node
                    
                    for ep in range(1, EPOCHS + 1):
                        tr_ce = run_epoch(dl_tr, train=True)
                        val_ce = run_epoch(dl_te, train=False)
                        
                        scheduler.step(val_ce)  # ↓ LR if plateau
                        improved = val_ce < best_loss - 1e-6  # tiny tolerance
                        if improved:
                            best_loss, wait = val_ce, 0
                            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                        else:
                            wait += 1
                        
                        if ep % 10 == 0:  # Print every 10 epochs to avoid spam
                            print(f"  Epoch {ep:3} │ Train CE {tr_ce:.4f} │ Val CE {val_ce:.4f} │ LR {optimizer.param_groups[0]['lr']:.2e}")
                        
                        if wait >= patience_epochs:
                            print(f"  ⏹️  Early stopping triggered after {ep} epochs.")
                            break
                    
                    # Reload best weights (exactly as in gnn_experiment.py)
                    if best_state:
                        model.load_state_dict(best_state)
                        print(f"  🗄️  Restored weights with best Val CE = {best_loss:.4f}")
                    
                    training_time = time.time() - start_time
                    
                    # Evaluation (exactly as in gnn_experiment.py)
                    print("Evaluating model...")
                    
                    def logits_and_labels(loader):
                        model.eval()
                        outs, ys = [], []
                        with torch.no_grad():
                            for d in loader:
                                d = d.to(DEVICE)
                                outs.append(model(d).cpu().numpy())
                                ys.append(d.y.cpu().numpy())
                        return np.concatenate(outs), np.concatenate(ys)
                    
                    log_tr, y_tr = logits_and_labels(dl_tr)
                    log_te, y_te = logits_and_labels(dl_te)
                    
                    # Use the same evaluate function as gnn_experiment.py and capture its output
                    # Capture stdout to get the AUC values
                    import io
                    import sys
                    
                    # Capture stdout from evaluate function
                    old_stdout = sys.stdout
                    sys.stdout = io.StringIO()
                    
                    evaluate(log_tr.argmax(1), log_tr, pd.DataFrame(), y_tr, "Train", results_dir)
                    train_output = sys.stdout.getvalue()
                    
                    evaluate(log_te.argmax(1), log_te, pd.DataFrame(), y_te, "Test", results_dir)
                    test_output = sys.stdout.getvalue()
                    
                    # Restore stdout
                    sys.stdout = old_stdout
                    
                    # Extract AUC values from the captured output
                    import re
                    
                    train_auc_match = re.search(r'Train Average ROC AUC: ([\d.]+)', train_output)
                    test_auc_match = re.search(r'Test Average ROC AUC: ([\d.]+)', test_output)
                    
                    train_auc = float(train_auc_match.group(1)) if train_auc_match else np.nan
                    test_auc = float(test_auc_match.group(1)) if test_auc_match else np.nan
                    
                    # Extract accuracy values
                    train_acc_match = re.search(r'Train Accuracy: ([\d.]+)', train_output)
                    test_acc_match = re.search(r'Test Accuracy: ([\d.]+)', test_output)
                    
                    train_acc = float(train_acc_match.group(1)) if train_acc_match else np.nan
                    test_acc = float(test_acc_match.group(1)) if test_acc_match else np.nan
                    
                    print(f"  Extracted - Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}")
                    print(f"  Extracted - Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
                    
                    metrics = {
                        'train_accuracy': train_acc,
                        'val_accuracy': test_acc,
                        'train_auc': train_auc,
                        'val_auc': test_auc
                    }
                    
                    # Store results as a flat dictionary for CSV
                    result_row = {
                        "config_id": hp_config["config_id"],
                        "model_type": hp_config["model_type"],
                        "scenario": scenario_name,
                        "hidden": hp_config["hidden"],
                        "n_layers": hp_config["n_layers"],
                        "lr": hp_config["lr"],
                        "batch_size": hp_config["batch_size"],
                        "dropout": hp_config["dropout"],
                        "n_heads": hp_config.get("n_heads", None),  # For GAT models
                        # Main metrics (what you care about most)
                        "train_accuracy": metrics['train_accuracy'],
                        "train_auc": metrics['train_auc'],
                        "test_accuracy": metrics['val_accuracy'],  # Renamed for clarity
                        "test_auc": metrics['val_auc'],           # Renamed for clarity
                        # Additional metrics
                        "best_val_loss": best_loss,
                        "training_time": training_time,
                        "n_features": len(feats),
                        "n_classes": n_classes,
                        "n_train_samples": len(dl_tr.dataset),
                        "n_val_samples": len(dl_te.dataset),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    results.append(result_row)
                    
                    # Save partial results as CSV after each configuration
                    df_partial = pd.DataFrame(results)
                    df_partial.to_csv(partial_results_file, index=False)
                    
                    print(f"✅ Completed: {scenario_name}")
                    print(f"   Train Acc: {metrics['train_accuracy']:.4f}, Val Acc: {metrics['val_accuracy']:.4f}")
                    print(f"   Train AUC: {metrics['train_auc']:.4f}, Val AUC: {metrics['val_auc']:.4f}")
                    print(f"   Best Val Loss: {best_loss:.4f}")
                    print(f"   Training Time: {training_time:.1f}s")
                    
                    # Debug: show what's being saved to CSV
                    print(f"  CSV Debug - train_auc: {result_row['train_auc']}")
                    print(f"  CSV Debug - test_auc: {result_row['test_auc']}")
                    print(f"  CSV Debug - train_accuracy: {result_row['train_accuracy']}")
                    print(f"  CSV Debug - test_accuracy: {result_row['test_accuracy']}")
                    
                except Exception as e:
                    print(f"❌ Error in scenario {scenario_name}: {str(e)}")
                    # Save error result as a row
                    error_row = {
                        "config_id": hp_config["config_id"],
                        "model_type": hp_config["model_type"],
                        "scenario": scenario_name,
                        "hidden": hp_config["hidden"],
                        "n_layers": hp_config["n_layers"],
                        "lr": hp_config["lr"],
                        "batch_size": hp_config["batch_size"],
                        "dropout": hp_config["dropout"],
                        "n_heads": hp_config.get("n_heads", None),
                        # Main metrics (what you care about most)
                        "train_accuracy": np.nan,
                        "train_auc": np.nan,
                        "test_accuracy": np.nan,
                        "test_auc": np.nan,
                        # Additional metrics
                        "best_val_loss": np.nan,
                        "training_time": np.nan,
                        "n_features": np.nan,
                        "n_classes": np.nan,
                        "n_train_samples": np.nan,
                        "n_val_samples": np.nan,
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e)
                    }
                    results.append(error_row)
                    
                    # Save partial results as CSV
                    df_partial = pd.DataFrame(results)
                    df_partial.to_csv(partial_results_file, index=False)
                    
                    continue
        
        print(f"\n✅ Completed configuration {config_idx + 1}/{len(configs)}")
    
    # Save final results
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED!")
    print(f"Results saved to: {results_dir}")
    
    # Create final CSV with all results
    df_final = pd.DataFrame(results)
    final_csv_path = results_dir / "final_results.csv"
    df_final.to_csv(final_csv_path, index=False)
    print(f"Final results saved to: {final_csv_path}")
    
    # Create summary table (filter out error rows)
    summary_data = df_final[df_final["error"].isna()].copy()
    if not summary_data.empty:
        summary_csv_path = results_dir / "results_summary.csv"
        summary_data.to_csv(summary_csv_path, index=False)
        print(f"Summary table saved to: {summary_csv_path}")
        
        # Print top 5 configurations by validation accuracy
        print("\nTop 5 configurations by test accuracy:")
        top_configs = summary_data.nlargest(5, "test_accuracy")[["config_id", "model_type", "scenario", "test_accuracy", "test_auc"]]
        print(top_configs.to_string(index=False))
    
    # Save configuration as JSON (keep this for reference)
    with open(results_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfiguration saved to: {results_dir}/experiment_config.json")
    print(f"Partial results saved to: {results_dir}/partial_results.csv")
    print(f"Final results saved to: {results_dir}/final_results.csv")


if __name__ == "__main__":
    main() 