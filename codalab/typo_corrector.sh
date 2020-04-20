#!/bin/bash
# Script to run typo corrector training
# Possible tasks are RTE, MRPC, SST-2, QNLI, MNLI, QQP

echo 'Running typo corrector training for task: $1...'

# Activate at-env conda environment
source activate atenv
# Missing package?
pip install query
# conda list

# Set required environment variables
export TASK_NAME=$1
export CLUSTERER_PATH=$HOME/clusterers/vocab100000_ed1.pkl
export GLUE_DIR=$HOME/data/glue_data
export TC_DIR=$HOME/tc_data

# Make directory TC_DIR
mkdir $TC_DIR
mkdir $TC_DIR/glue_tc_preprocessed

# Needed for CodaLab as typo-embeddings is mounted one level down
cd typo-embeddings || exit 1

# Store preprocessed data, vocabularies, and models
echo 'Storing preprocessed data, vocabularies, and models...'
python preprocess_tc.py --glue_dir $GLUE_DIR --save_dir $TC_DIR/glue_tc_preprocessed

# Change directory to scRNN
cd scRNN || exit 1

# Train a typo-corrector based on random perturbations to the task data
echo 'Training a typo-corrector based on random perturbations to the data...'
python train.py --task-name $TASK_NAME --preprocessed_glue_dir $TC_DIR/glue_tc_preprocessed --tc_dir $TC_DIR