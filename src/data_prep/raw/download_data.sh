#!/bin/bash -l

#SBATCH -p download

# Load the required modules
module load anaconda/Python-ML-2025a
pip install --user obspy

# Run the Python download script
python download_data.py

