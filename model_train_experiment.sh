#!/bin/bash -l

#SBATCH --cpus-per-task=4            # Number of CPUs per task
#SBATCH --mem=64G                    # Total memory per node

# Load the required modules
module load anaconda/Python-ML-2025a

# Run the Python download script
python -m src.model.model_train

