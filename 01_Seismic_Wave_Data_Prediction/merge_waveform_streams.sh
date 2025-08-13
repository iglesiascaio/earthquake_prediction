#!/bin/bash

#SBATCH --mem=180G                    # Total memory per node

# Run the script
python 04_Trace_Preprocessing/merge_waveform_streams.py
