#!/bin/bash

#SBATCH --mem=180G                    # Total memory per node
#SBATCH --gres=gpu:volta:1

ulimit -s unlimited

# Run the script
python /home/gridsan/mknuth/src/model/model_train.py