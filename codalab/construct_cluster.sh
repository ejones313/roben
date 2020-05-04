#!/bin/bash
# Script to construct connected component clusters

# Activate at-env conda environment
source activate erik-cert
# Missing package?

# Set required environment variables
export CLUSTERER_PATH=$HOME/clusterers

# Make directory CLUSTERER_PATH
mkdir $CLUSTERER_PATH

# Needed for CodaLab as roben is mounted one level down
cd roben || exit 1

# Make clusterer
echo 'Constructing connected component clusters. Attack type: '$1
python construct_clusters.py --vocab_size 100000 --perturb_type $1 --data_dir $HOME/data --output_dir $CLUSTERER_PATH
