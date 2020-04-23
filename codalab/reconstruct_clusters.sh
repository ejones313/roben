#!/bin/bash
# Script to reconstruct agglomerative clusters

# Activate at-env conda environment
source activate atenv
# Missing package?
pip install query
# conda list

# Set required environment variables
# Make directory where clusterers will be stored
mkdir $HOME/clusterers

# Needed for CodaLab as roben is mounted one level down
cd roben || exit 1

echo 'Reconstructing...'
python reconstruct_clusters.py --save_path $HOME/clusterers/vocab100000_ed1_gamma0.3.pkl --file_paths $HOME/agglom1/clusterers/vocab100000_ed1_gamma0.3/job0outof2  $HOME/agglom2/clusterers/vocab100000_ed1_gamma0.3/job1outof2