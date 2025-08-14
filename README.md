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
│  │ • 30-day 3-comp │     │ • Earthquake   │    │ • Multi-class           │  │
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
- **Output**: Discrete magnitude risk assessment with uncertainty quantification

## 📁 Repository Structure

```
earthquake-ai/
├── 📊 01_Seismic_Wave_Data_Prediction/     # Main SeisLM + Toto pipeline
│   ├── seisLM_main.py                      # Main orchestration script
│   ├── config.yaml                         # Configuration parameters
│   ├── 02_Functions/                       # Core model implementations
│   │   ├── Model_Trainer.py               # Training orchestration
│   │   ├── Model_Evaluator.py             # Evaluation framework
│   │   ├── Dataset_creation.py            # Waveform dataset creation
│   │   ├── SeisLM_train.py                # SeisLM model definitions
│   │   └── seisLM/                        # SeisLM foundation model
│   ├── 01_Data/                           # Data storage and processing
│   │   ├── 01_Seismic_Wave_Data/          # Processed waveform streams
│   │   └── 02_Seismic_Event_Data/         # Earthquake catalogs
│   └── 03_Results/                         # Model outputs and checkpoints
│
├── 🧠 02_Full_Model/                       # GNN-based spatial modeling
│   ├── src/model/                          # GNN model implementations
│   │   ├── gnn_experiment.py              # Main GNN experiment
│   │   ├── gnn_hyperparameter_tuning.py   # 75-architecture tuning
│   │   └── utils/                          # GNN utilities
│   ├── config/                             # Configuration files
│   ├── data/                               # Processed feature data
│   └── results/                            # GNN experiment results
│
├── 🚀 toto/                                # Toto foundation model
├── 📚 Toto-Open-Base-1.0/                 # Pre-trained Toto weights
└── 📄 Research_Report-4.pdf               # Comprehensive research documentation
```

## 🔬 Key Components

### 1. Seismic Waveform Processing (`01_Seismic_Wave_Data_Prediction/`)

#### **Data Sources**
- **SCEDC**: Southern California Earthquake Data Center via AWS
- **Coverage**: 25+ broadband stations, 3 components (BHZ, BHE, BHN)
- **Format**: MiniSEED files with continuous recordings

#### **Preprocessing Pipeline**
```python
# Waveform processing steps
1. Gap detection and zero-filling
2. Multi-trace merging for 24-hour coverage
3. Downsampling (factor 16) for computational efficiency
4. Multi-day stream creation with sliding windows
5. Normalization (zero-mean, unit variance)
```

#### **SeisLM Integration**
- **Foundation Model**: Pre-trained SeisLM with frozen backbone
- **Embeddings**: 256-dimensional representations per time window
- **Aggregation**: Multiple temporal aggregation strategies:
  - **LSTM + Attention**: Sequential processing with learned weighting
  - **Toto Head**: Transformer-based foundation model integration

### 2. Tabular Feature Engineering (`02_Full_Model/`)

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

### 4. Multi-Modal Fusion

#### **Feature Combination Strategies**
1. **Waveform Only**: SeisLM embeddings + temporal aggregation
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

# Install dependencies
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

### 3. Run Training

```bash
# SeisLM + Toto pipeline
cd 01_Seismic_Wave_Data_Prediction
python seisLM_main.py --mode train

# GNN hyperparameter tuning
cd 02_Full_Model
sbatch gnn_hyperparameter_tuning.sh
```

### 4. Evaluate Results

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

## 🔗 Related Work

- **SeisLM**: [Foundation Model for Seismic Waveforms](https://arxiv.org/abs/2410.15765)
- **Toto**: [Time Series Optimized Transformer](https://arxiv.org/abs/2505.14766)
- **SCEDC**: [Southern California Earthquake Data Center](https://scedc.caltech.edu/)

## 📞 Contact

- **Maximilian Knuth**: mknuth@mit.edu
- **Caio Iglesias**: caiopigl@mit.edu

