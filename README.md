# Earthquake AI: Predicting Maximum Earthquake Magnitude in California

[![Research Report](https://img.shields.io/badge/Research-Report%20PDF-blue)](Research_Report-4.pdf)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📋 Project Overview

This repository implements a **hybrid modeling framework** for forecasting the maximum earthquake magnitude in the Los Angeles region over the next 30 days. The methodology integrates two complementary approaches:

1. **Seismic Waveform-Based Model**: Leverages continuous waveform recordings and deep representation learning using the SeisLM foundation model
2. **Event-Based Probabilistic Model**: Grounded in Gutenberg-Richter relations and historical seismicity rates

The system combines these approaches to enhance predictive robustness and accuracy, providing early warning capabilities for seismic risk assessment.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EARTHQUAKE PREDICTION PIPELINE                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │   WAVEFORM      │    │    TABULAR      │    │     PREDICTION          │  │
│  │   PIPELINE      │    │    PIPELINE     │    │     SYSTEM              │  │
│  │                 │    │                 │    │                         │  │
│  │ • 30-day 3-comp │    │ • Earthquake    │    │ • Multi-class           │  │
│  │   waveforms     │    │   catalogs      │    │   classification        │  │
│  │ • SeisLM        │    │ • Rolling       │    │ • Magnitude bins        │  │
│  │   embeddings    │    │   features      │    │ • 30-day horizon        │  │
│  │ • LSTM/Toto     │    │ • ETAS, B-values│    │ • Hybrid fusion         │  │
│  │   aggregation   │    │ • Energy stats  │    │                         │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │
│           │                       │                       │                 │
│           └───────────────────────┼───────────────────────┘                 │
│                                   │                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    FEATURE FUSION & MODELING                        │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │    │
│  │  │  Gradient   │  │    GNN      │  │      Evaluation &           │  │    │
│  │  │  Boosting   │  │  (GAT/SAGE) │  │      Deployment             │  │    │
│  │  │             │  │             │  │                             │  │    │
│  │  │ • LightGBM  │  │ • Spatial   │  │ • Multi-metric              │  │    │
│  │  │ • Tabular   │  │   awareness │  │   assessment                │  │    │
│  │  │ • Efficient │  │ • Station   │  │ • ROC-AUC, Accuracy         │  │    │
│  │  │             │  │   networks  │  │ • Confusion matrices        │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────────────────────-┘    │
└────────────────────────────────────────────────────────────────────────────-┘
```

## 🎯 Research Objectives

- **Primary Goal**: Predict maximum earthquake magnitude in LA region within 30 days
- **Classification**: 9 magnitude classes (M<1.0 to M≥8.0)
- **Input**: Continuous seismic waveforms + historical earthquake catalogs
- **Output**: Discrete magnitude risk assessment

## 📁 Repository Structure

```
earthquake-ai/
├── 📊 01_Seismic_Wave_Data_Prediction/     # Main SeisLM + Toto pipeline
│   ├── seisLM_main.py                      # Main orchestration script
│   ├── seisLM_main.sh                      # SLURM job submission script
│   ├── seisLM_main_new.ipynb               # Jupyter notebook version
│   ├── config.yaml                         # Configuration parameters
│   ├── TESTING.ipynb                       # Comprehensive testing notebook
│   ├── merge_waveform_streams.sh           # Shell script for waveform merging
│   ├── merge_waveform_streams.py           # Python script for waveform merging
│   ├── 02_Functions/                       # Core model implementations
│   │   ├── Model_Trainer.py               # Training orchestration
│   │   ├── Model_Evaluator.py             # Evaluation framework
│   │   ├── Dataset_creation.py            # Waveform dataset creation
│   │   ├── SeisLM_train.py                # SeisLM model definitions
│   │   ├── __init__.py                    # Package initialization
│   │   ├── 00_Archive/                    # Archived/legacy implementations
│   │   │   ├── Dataset_creation_old.py   # Previous dataset creation version
│   │   │   ├── SeisLM_train.py           # Previous SeisLM training version
│   │   │   └── Seismic_event_data_preprocessing.py # Legacy preprocessing
│   │   └── seisLM/                        # SeisLM foundation model
│   ├── 01_Data/                           # Data storage and processing
│   │   ├── 01_Seismic_Wave_Data/          # Processed waveform streams
│   │   │   ├── 2020/                      # 2020 waveform data
│   │   │   ├── 2021/                      # 2021 waveform data
│   │   │   ├── 2022/                      # 2022 waveform data
│   │   │   ├── 2023/                      # 2023 waveform data
│   │   │   ├── 2024/                      # 2024 waveform data
│   │   │   ├── Combined_Processed_Streams_30_new/ # 30-day combined streams
│   │   │   ├── Combined_Processed_Streams_30_new_2025_07_28/ # 30-day streams (more data)
│   │   │   └── Combined_Processed_Streams_50_new_2025_07_28/ # 50-day combined streams (more data)
│   │   └── 02_Seismic_Event_Data/         # Earthquake catalogs
│   │       ├── earthquake_features.parquet # Main earthquake features 
│   │       ├── curr_earthquake_features.parquet # Current earthquake features
│   │       ├── earthquake_events_2020_2025_Caio.parquet # Caio's earthquake dataset 
│   ├── 03_Results/                         # Model outputs and checkpoints
│   │   ├── seislm_toto_*/                 # Toto-based model results 
│   │   ├── seislm_lstm_*/                 # LSTM-based model results 
│   │   └── [Timestamped experiment directories] # All experiment outputs
│   ├── 04_Trace_Preprocessing/             # Waveform preprocessing pipeline
│   │   ├── process_waveforms.ipynb         # Main preprocessing notebook
│   │   ├── merge_waveform_streams.py       # Waveform merging utility
│   │   ├── 01_Data/                       # Raw waveform data
│   │   ├── 02_Logs/                       # Processing logs
│   │   └── 03_Waveform_Analysis/          # Station-specific analysis notebooks
│   │       ├── PASC_waveform_analysis.ipynb    # PASC station analysis
│   │       ├── MAN_waveform_analysis.ipynb     # MAN station analysis
│   │       └── [Additional station notebooks]  # Other station analyses
│   ├── toto_models/                        # Toto model checkpoints
│
├── 🧠 02_Full_Model/                       # GNN-based spatial modeling
│   ├── src/                                # Source code directory
│   │   ├── model/                          # GNN model implementations
│   │   │   ├── gnn_experiment.py           # Main GNN experiment
│   │   │   ├── gnn_hyperparameter_tuning.py # 75-architecture tuning
│   │   │   ├── matching_experiment.py      # Feature matching experiments
│   │   │   ├── model_train.py              # GNN training utilities
│   │   │   └── utils/                      # GNN utilities and helpers
│   │   │       ├── __init__.py             # Utils package initialization
│   │   │       ├── evaluation.py           # Evaluation utilities
│   │   │       ├── gnn_helper.py           # GNN-specific helper functions
│   │   │       └── load_data.py            # Data loading utilities
│   │   └── data_prep/                      # Data preparation pipeline
│   │       ├── __init__.py                 # Data prep package initialization
│   │       ├── raw/                        # Raw data download utilities
│   │       │   ├── __init__.py             # Raw package initialization
│   │       │   ├── download_data.py        # Earthquake catalog download
│   │       │   └── download_data.sh        # Download script for SLURM
│   │       └── features/                   # Feature engineering pipeline
│   │           ├── __init__.py             # Features package initialization
│   │           └── create_features.py      # Seismological feature creation
│   ├── config/                             # Configuration files
│   │   ├── 00-download-config.yaml         # Data download configuration
│   │   └── 10-features-config.yaml         # Feature engineering configuration
│   ├── notebooks/                          # Jupyter notebooks
│   │   ├── data_exploration.ipynb          # Data exploration and analysis
│   │   ├── model_exploration.ipynb         # Model architecture exploration
│   │   └── next_steps.ipynb                # Future development planning
│   ├── data/                               # Processed feature data
│   │   ├── features/                       # Engineered features
│   │   │   └── earthquake_features.parquet # Main feature dataset (5.5MB)
│   │   └── raw/                            # Raw earthquake data
│   ├── results/                             # GNN experiment results
│   ├── activate                             # Environment activation script
│   ├── gnn_experiment.sh                   # SLURM job for GNN experiments
│   ├── gnn_hyperparameter_tuning.sh        # SLURM job for hyperparameter tuning
│   ├── model_train_experiment.sh            # SLURM job for model training
│   └── matching_experiment.sh               # SLURM job for matching experiments
│
├── 🚀 toto/                                # Toto foundation model source
├── 📚 Toto-Open-Base-1.0/                 # Pre-trained Toto weights
├── 📄 Research_Report-4.pdf                # Comprehensive research documentation
├── 📋 requirements.txt                     # Python dependencies and versions
├── 📖 README.md                            # This documentation file
├── 📝 .gitignore                           # Git ignore patterns
```

## 🔬 Key Components

### 1. Seismic Waveform Processing (`01_Seismic_Wave_Data_Prediction/`)

#### **Data Sources**
- **SCEDC**: Southern California Earthquake Data Center via AWS
- **Coverage**: 25+ broadband stations, 3 components (BHZ, BHE, BHN)
- **Format**: MiniSEED files with continuous recordings


#### **Waveform Preprocessing Pipeline**
- **`process_waveforms.ipynb`**: Main preprocessing workflow for all daily waveforms of stations
- **`merge_waveform_streams.py`**: Utility for combining daily waveform files to multiple day waveforms
- **Station-Specific Analysis**: Individual notebooks for each seismic station
- **Data Quality Control**: Gap detection, zero-filling, and validation
- **Multi-Day Streams**: Configurable windowing for temporal analysis

#### **SeisLM Integration**
- **Foundation Model**: Pre-trained SeisLM with frozen backbone (130MB checkpoint)
- **Embeddings**: 256-dimensional representations per time window
- **Aggregation**: Multiple temporal aggregation strategies:
  - **LSTM + Attention**: Sequential processing with learned weighting
  - **Toto Head**: Transformer-based foundation model integration

#### **Embedding Generation and Storage**
- **`seisLM_main.py`**: Creates SeisLM embeddings from seismic waveforms
- **Embedding Storage**: Saves embeddings in `03_Results/` folder for reuse
- **Performance Evaluation**: Tests embeddings alone for magnitude prediction
- **Baseline Results**: Establishes performance benchmark using only waveform embeddings
- **Reusable Features**: Embeddings can be loaded for different downstream tasks



### 2. Tabular Feature Engineering (`02_Full_Model/`)

#### **Data Preparation Pipeline**
- **`download_data.py`**: Automated earthquake catalog download from FDSN services
- **`download_data.sh`**: SLURM-compatible download script for batch processing
- **`create_features.py`**: Comprehensive feature engineering from raw catalogs
- **Multi-station Support**: Handles 50+ stations with automated processing

#### **Seismological Features**
- **Basic Statistics**: Daily max/min/mean magnitude, event counts
- **Energy Measures**: Gutenberg-Richter energy scaling
- **Recurrence Features**: Time since last event by magnitude class
- **Advanced Indicators**: B-values, T-values, magnitude deficits
- **ETAS Modeling**: Aftershock intensity and clustering

#### **Feature Aggregation**
```python
# Daily station-level features
- Rolling windows (7-30 days)
- Spatial proximity filtering (50km radius)
- Magnitude-dependent calculations
- Temporal lag features for context
```

### 3. Graph Neural Network Modeling (`02_Full_Model/src/model/`)

#### **Architecture Variants**
- **StationGNN**: GraphSAGE-based spatial modeling
- **StationGAT**: Graph Attention Network with learned neighbor importance
- **Spatial Context**: 100km radius for station connectivity

#### **Hyperparameter Tuning**
- **25 Model Configurations**: Hidden sizes, layers, attention heads
- **3 Embedding Scenarios**: Feature availability optimization
- **75 Total Experiments**: Systematic architecture evaluation

#### **Exploration and Analysis Tools**
- **`data_exploration.ipynb`**: Comprehensive data analysis and visualization
- **`model_exploration.ipynb`**: Model architecture experimentation
- **`next_steps.ipynb`**: Future development planning and roadmap
- **Configuration Files**: YAML-based setup for data download and feature engineering

### 4. Multi-Modal Fusion

#### **Feature Combination Strategies**
1. **Waveform Only**: SeisLM embeddings + temporal aggregation
   - **Baseline Performance**: Embeddings alone achieve significant magnitude prediction accuracy
   - **Stored Results**: Embeddings saved in results folder for analysis and reuse
   - **Temporal Context**: LSTM/Toto aggregation captures sequential patterns
2. **Tabular Only**: Engineered seismological features
3. **Hybrid Approach**: Combined waveform + tabular features

#### **Modeling Approaches**
- **Gradient Boosting**: LightGBM for efficient tabular learning
- **Graph Neural Networks**: Spatial-aware station modeling
- **Ensemble Methods**: Combining multiple approaches

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd earthquake-ai

# Activate conda environment
conda activate toto-310

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

### 2. Configuration

Edit `01_Seismic_Wave_Data_Prediction/config.yaml`:

```yaml
# Data paths
paths:
  combined_stream_dir: "/path/to/processed/waveforms"
  earthquake_parquet: "/path/to/earthquake/catalog"
  output_dir: "/path/to/results"

# Model configuration
model:
  model_type: "seislm"           # Options: "seislm", "scattering"
  aggregation_type: "toto"        # Options: "toto", "lstm", "transformer"
  toto:
    regime: "partial"             # Options: "head", "partial", "full"
    pretrained_name: "/path/to/toto/model"

# Training parameters
training:
  num_epochs: 10
  batch_size: 4
  learning_rate: 0.0005
```

### 3. Data Preparation (Optional)

#### **Download Seismic Waveforms**
```python
import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from obspy import UTCDateTime

# Configuration
bucket_name = "scedc-pds"
station = "JVA"  # Example station
channels = ["BHZ", "BHE", "BHN"]
year = 2020      # Example year
base_dir = f"01_Data/01_Seismic_Wave_Data/{station}_BB_{year}_s3"

# S3 client (unsigned access)
s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
bucket = s3.Bucket(bucket_name)

# Loop through each day and channel
start_day = UTCDateTime(f"{year}-01-01")
end_day = UTCDateTime(f"{year}-12-31")
day = start_day

while day <= end_day:
    jday = f"{day.julday:03d}"
    date_str = day.strftime("%Y-%m-%d")

    for channel in channels:
        # Construct S3 key and local path
        key = f"continuous_waveforms/{year}/{year}_{jday}/CI{station}__{channel}___{year}{jday}.ms"
        channel_dir = os.path.join(base_dir, channel)
        os.makedirs(channel_dir, exist_ok=True)
        filename = f"{station}_{channel}_{date_str}.mseed"
        local_path = os.path.join(channel_dir, filename)

        try:
            bucket.download_file(key, local_path)
            print(f"✅ Saved: {local_path}")
        except Exception as e:
            print(f"❌ Failed: {key} — {e}")

    day += 86400  # move to next day

print("🎉 All files downloaded and saved in structured folders.")
```

**Important**: After downloading, you must process the waveforms before training:

1. **Preprocess Waveforms**: Use `process_waveforms.ipynb` to clean and prepare the data
2. **Merge Streams**: Use `merge_waveform_streams.py` to combine daily files into longer segments (e.g., 30-day streams)
3. **Quality Control**: Detect gaps, fill missing data, and validate stream integrity
4. **Then Train**: Only after preprocessing can you run the training pipeline

#### **Download Earthquake Catalogs**
```bash
# Download earthquake catalogs and create features
cd 02_Full_Model
python src/data_prep/raw/download_data.py
python src/data_prep/features/create_features.py

# Or use SLURM scripts for large datasets
sbatch src/data_prep/raw/download_data.sh
```

### 4. Run Training

```bash
# SeisLM + Toto pipeline
cd 01_Seismic_Wave_Data_Prediction
python seisLM_main.py --mode train

# Or submit as SLURM job
sbatch seisLM_main.sh
```

**Training Workflow**:
1. **Generates embeddings** from seismic waveforms using SeisLM
2. **Saves embeddings** in `03_Results/` folder for reuse
3. **Evaluates baseline performance** using embeddings alone for magnitude prediction
4. **Trains full model** with temporal aggregation (LSTM/Toto)

# GNN hyperparameter tuning
cd 02_Full_Model
sbatch gnn_hyperparameter_tuning.sh

# Other GNN experiments
sbatch gnn_experiment.sh           # Single GNN experiment
sbatch model_train_experiment.sh   # Model training experiment
sbatch matching_experiment.sh      # Feature matching experiment
```

**HPC Integration**: All major experiments include SLURM job scripts for cluster execution:
- **`seisLM_main.sh`**: Main SeisLM + Toto training pipeline
- **`gnn_hyperparameter_tuning.sh`**: 75-architecture systematic evaluation
- **`gnn_experiment.sh`**: Single GNN architecture testing
- **`model_train_experiment.sh`**: Model training experiments
- **`matching_experiment.sh`**: Feature matching and comparison studies

### 5. Evaluate Results

```bash
# Model evaluation
python seisLM_main.py --mode evaluate

# View GNN results
cd 02_Full_Model/results
python -c "import json; print(json.dumps(json.load(open('final_results.json')), indent=2))"
```

## 📊 Current Performance

### **Preliminary Results** (Limited Data, Top Features)
- **Baseline (No Embeddings)**: ROC-AUC: 0.5825, Accuracy: 0.6436
- **With SeisLM Embeddings**: ROC-AUC: 0.6583 (+0.0758), Accuracy: 0.6277
- **Best Performance**: Classes 3-4 (M 3.0-5.0) with AUC 0.77-0.74

### **Data Availability**
- **Complete Coverage**: PASC, SDG, SYN stations (2020-2024)
- **Partial Coverage**: Multiple stations with missing years
- **Limited Data**: Many stations missing 2020-2022 data

### **Data Organization**
- **Waveform Data**: 5 years of continuous recordings (2020-2024)
- **Combined Streams**: 30-day and 50-day processed waveform segments
- **Earthquake Catalogs**: Multiple parquet datasets with varying completeness
- **Feature Datasets**: Engineered seismological features (5.5MB+)
- **Model Checkpoints**: Pre-trained SeisLM model (130MB)
- **Experiment Results**: 90+ timestamped experiment directories

## 🔗 Related Work

- **SeisLM**: [Foundation Model for Seismic Waveforms](https://arxiv.org/abs/2410.15765)
- **Toto**: [Time Series Optimized Transformer](https://arxiv.org/abs/2505.14766)
- **SCEDC**: [Southern California Earthquake Data Center](https://scedc.caltech.edu/)

## 📞 Contact

- **Maximilian Knuth**: mknuth@mit.edu
- **Caio Iglesias**: caiopigl@mit.edu

