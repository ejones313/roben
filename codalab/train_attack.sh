#!/bin/bash
# Script to run typo corrector training
# Possible tasks are RTE, MRPC, SST-2, QNLI, MNLI, QQP

echo 'Running traing and attack for task: $1...'

# Activate at-env conda environment
source activate atenv
# Missing package?
pip install query

# Set required environment variables
export TASK_NAME=$1
export CLUSTERER_PATH=$HOME/clusterers/vocab100000_ed1.pkl
export GLUE_DIR=$HOME/data/glue_data
export TC_DIR=$HOME/tc_data

# Train and Attack
echo 'Training and then attacking...'
python run_glue.py --log_stdout_only --tc_dir $TC_DIR --task_name $TASK_NAME --do_lower_case --do_train --do_eval --data_dir $GLUE_DIR/$TASK_NAME --output_dir $HOME/model_output/$TASK_NAME --save_results --save_dir $HOME/train --recoverer scrnn --augmentor identity --run_test
python run_glue.py --log_stdout_only --tc_dir $TC_DIR --task_name $TASK_NAME --do_lower_case --do_eval --data_dir $GLUE_DIR/$TASK_NAME --output_dir $HOME/model_output/$TASK_NAME  --save_results --save_dir $HOME/attack --recoverer scrnn --augmentor identity --run_test --model_name_or_path $HOME/model_output/$TASK_NAME  --attack --new_attack --attacker beam-search --beam_width 5 --attack_name LongDeleteShortAll --attack_type ed1