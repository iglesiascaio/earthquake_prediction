# GNN Hyperparameter Tuning for Earthquake Prediction

This directory contains a comprehensive hyperparameter tuning system for Graph Neural Networks (GNNs) applied to earthquake prediction.

## Overview

The system tests **25 different GNN architectures** across **3 embedding scenarios**, resulting in **75 total experiments**:

### Model Variations
- **Baseline**: Original configuration from `gnn_experiment.py` (always included first)
- **StationGNN**: GraphSAGE-based models
- **StationGAT**: Graph Attention Network models
- **Hyperparameters**: hidden size, layers, learning rate, batch size, dropout, attention heads

### Embedding Scenarios (Optimized Order)
1. **`no_embeddings_limited_data`**: `merge_embeddings=True, keep_embeddings=False` ⚡ **FASTEST**
2. **`embeddings_limited_data`**: `merge_embeddings=True, keep_embeddings=True` 🚀 **MEDIUM SPEED**  
3. **`no_embeddings_full_data`**: `merge_embeddings=False, keep_embeddings=False` 🐌 **SLOWEST (runs last)**

**Note**: Scenarios are automatically reordered to run fast ones first, slow ones last for better resource utilization.

## Quick Testing

For quick testing, enable **TEST MODE** in the script:

```python
# In gnn_hyperparameter_tuning.py
TEST_MODE = True  # Change to False for full 25-model run
```

**Test Mode Benefits**:
- Always includes **baseline configuration** (your original setup)
- Only runs **2 model configurations** instead of 25
- **4 total experiments** (2 configs × 2 fast scenarios only)
- **Estimated runtime**: 15-30 minutes
- Perfect for validating the pipeline before full runs
- **Skips slow scenarios**: No `no_embeddings_full_data` in test mode

## Files

- **`gnn_hyperparameter_tuning.py`**: Main Python script with all configuration
- **`gnn_hyperparameter_tuning.sh`**: SLURM job submission script
- **`README_hyperparameter_tuning.md`**: This file

## Usage

### 1. Quick Test (Recommended First)

```bash
# Edit the script to set TEST_MODE = True
# Then submit job
sbatch gnn_hyperparameter_tuning.sh
```

### 2. Full Run

```bash
# Edit the script to set TEST_MODE = False
# Then submit job
sbatch gnn_hyperparameter_tuning.sh
```

### 3. Monitor Progress

```bash
# Check job status
squeue -u $USER

# Monitor output
tail -f logs/gnn_hyperparameter_tuning_<JOB_ID>.out

# Check for errors
tail -f logs/gnn_hyperparameter_tuning_<JOB_ID>.err
```

### 4. View Results

Results are saved incrementally to prevent data loss:

- **`partial_results.json`**: Updated after each configuration
- **`final_results.json`**: Complete results when finished
- **`results_summary.csv`**: Summary table for easy analysis
- **`experiment_config.json`**: Experiment configuration

## Output Structure

Each result contains:
- Configuration details (model type, hyperparameters)
- Performance metrics (accuracy, AUC, loss)
- Training time and feature dimensions
- Error information (if any)

## Configuration

Edit `gnn_hyperparameter_tuning.py` to adjust:

### **Test Mode**
```python
# TEST MODE: Set to True for quick testing with only 2 models
TEST_MODE = True  # Change to False for full 25-model run
```

### **Data Paths**
```python
DATA = {
    "daily": "./data/features/earthquake_features.parquet",
    "seismic": "./data/embeddings/Embeddings_192142.pkl",
    "meta": "./data/raw/earthquake_data.parquet",
}
```

### **Training Parameters**
```python
CUT = "2024-01-01"      # Cutoff date for train/test split
RADIUS = 100.0          # Graph construction radius (km)
EPOCHS = 60             # Maximum training epochs
PATIENCE = 12           # Early stopping patience
```

### **Feature Selection**
```python
TOP_FEATS = [
    "time_since_class_3", "rolling_T_value", "daily_count_30d_sum",
    "daily_b_value", "rolling_dE_half", "daily_etas_intensity",
    "time_since_class_2", "daily_count_7d_sum",
]
```

## Key Features

- **Incremental saving**: Results saved after each configuration
- **Error handling**: Continues even if individual runs fail
- **Comprehensive logging**: Force-printed output for supercloud
- **Reproducible**: Deterministic CUDA settings
- **Flexible**: Easy to modify hyperparameter ranges
- **Optimized ordering**: Fast scenarios run first, slow ones last
- **Test mode**: Quick validation with 2 models

## Expected Runtime

### Test Mode (2 configurations)
- **4 total experiments** (2 configs × 2 fast scenarios only)
- **Each experiment**: ~5-15 minutes
- **Total estimated time**: 15-30 minutes
- **Memory usage**: 64GB
- **Scenarios skipped**: `no_embeddings_full_data` (too slow for testing)

### Full Mode (25 configurations)
- **75 total experiments** (25 configs × 3 scenarios)
- **Each experiment**: ~5-15 minutes (depending on model size)
- **Total estimated time**: 6-18 hours
- **Memory usage**: 64GB (configurable in SLURM script)

## Analysis

After completion, analyze results using the summary CSV:

```python
import pandas as pd
df = pd.read_csv("results/gnn_hyperparameter_tuning_<timestamp>/results_summary.csv")

# Top configurations by validation accuracy
top_configs = df.nlargest(10, "val_acc")

# Best per scenario
best_per_scenario = df.loc[df.groupby("scenario")["val_acc"].idxmax()]

# Compare embedding scenarios
scenario_performance = df.groupby("scenario")[["val_acc", "val_auc"]].mean()
```

## Troubleshooting

- **Out of memory**: Reduce batch sizes or hidden dimensions
- **Long training**: Reduce epochs or increase patience
- **Crashes**: Check partial results and restart from last saved point
- **Poor performance**: Adjust learning rates or add regularization
- **Want to test first**: Enable TEST_MODE for quick validation 