#!/bin/bash -l

#SBATCH -p download
#SBATCH --cpus-per-task=1            # Number of CPUs per task
#SBATCH --mem=32G                    # Total memory per node

# Load the required modules
module load anaconda/Python-ML-2025a
pip install --user obspy

# Run the Python download script
python matching_experiment.py

