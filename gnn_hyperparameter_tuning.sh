#!/bin/bash -l

#SBATCH --job-name=gnn_hyperparameter_tuning
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gnn_hyperparameter_tuning_%j.out
#SBATCH --error=logs/gnn_hyperparameter_tuning_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load the required modules
module load anaconda/Python-ML-2025a


# Set environment variables for better logging
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Print job information
echo "Starting GNN Hyperparameter Tuning Experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Run the hyperparameter tuning script
python -m src.model.gnn_hyperparameter_tuning

# Print completion information
echo "=========================================="
echo "Experiment completed at: $(date)"
echo "Job finished with exit code: $?" 