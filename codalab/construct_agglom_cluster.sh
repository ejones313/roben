#!/bin/bash
# Script to construct agglomerative clusters

echo 'job_id: '$1
echo 'gamma: '$2

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

# Make clusterer
echo 'Constructing a connected component clusters'
python construct_clusters.py --vocab_size 100000 --perturb_type ed1 --data_dir $HOME/data --output_dir $HOME/clusterers

#Now set the clusterer path
export CLUSTERER_PATH=$HOME/clusterers/vocab100000_ed1.pkl

#Make directory where partial agglomerative clusters will be saved
mkdir $HOME/clusterers/vocab100000_ed1_gamma$2

# We will now construct our more complicated clusters, the agglomerative clusters
echo 'Building agglomerative clusters (one of two jobs)...'
python agglom_clusters.py --gamma $2 --clusterer_path $CLUSTERER_PATH --job_id $1 --num_jobs 2
