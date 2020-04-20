#!/bin/bash
# Script to construct agglomerative clusters

echo 'job_id: '$1
echo 'gamma: '$1

# Activate at-env conda environment
source activate atenv
# Missing package?
pip install query
# conda list

# Set required environment variables
export CLUSTERER_PATH=$HOME/clusterers/vocab100000_ed1_gamma$2

# Make directory CLUSTERER_PATH
mkdir $HOME/clusterers
mkdir $CLUSTERER_PATH

# Needed for CodaLab as roben is mounted one level down
cd roben || exit 1

# Make clusterer
echo 'Constructing a clusterer...'
python construct_clusters.py --vocab_size 100000 --perturb_type ed1 --data_dir $HOME/data --output_dir $CLUSTERER_PATH

# We will now construct our more complicated clusters, the agglomerative clusters
echo 'Building agglomerative cluster...'
python agglom_clusters.py --gamma $2 --clusterer_path $CLUSTERER_PATH/vocab100000_ed1.pkl --job_id $1 --num_jobs 2
