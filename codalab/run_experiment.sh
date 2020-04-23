#!/bin/bash
# Script to run standard training + attack in CodaLab
# # Activate at-env conda environment
source activate atenv
# Missing package?
pip install query
# # conda list
# # Set required environment variables
export TASK_NAME=$1
export RECOVERER=$2
export AUGMENTOR=$3
export DO_ROBUST=''
export ATTACK_TYPE='ed1'
# Add --do_robust to compute robust accuracy when using clusters to defend
if [ $RECOVERER = 'clust-rep' ]; then
  export DO_ROBUST='--do_robust'
elif [ $RECOVERER = 'clust-intprm' ]; then
  export DO_ROBUST='--do_robust'
  export ATTACK_TYPE='intprm'
fi
export CLUSTERER_PATH=$HOME/clusterers/vocab100000_ed1.pkl
# When the fourth argument is present, change the cluster path to $HOME/$4
if [ "$#" -ge 3 ]; then
  export CLUSTERER_PATH=$HOME/$4
fi
export GLUE_DIR=$HOME/data/glue_data
export TC_DIR=$HOME/tc_data
# # Train and Attack
echo 'Training and then attacking...'
python run_glue.py --log_stdout_only --tc_dir $TC_DIR --task_name $TASK_NAME --do_lower_case --do_train --do_eval --data_dir $GLUE_DIR/$TASK_NAME --output_dir $HOME/model_output/$TASK_NAME --overwrite_output_dir --save_results --save_dir $HOME/train --recoverer $RECOVERER --augmentor $AUGMENTOR --run_test $DO_ROBUST
python run_glue.py --log_stdout_only --tc_dir $TC_DIR --task_name $TASK_NAME --do_lower_case --do_eval --data_dir $GLUE_DIR/$TASK_NAME --output_dir $HOME/model_output/$TASK_NAME  --save_results --save_dir $HOME/attack --recoverer $RECOVERER --augmentor $AUGMENTOR --run_test --model_name_or_path $HOME/model_output/$TASK_NAME  --attack --new_attack --attacker beam-search --beam_width 5 --attack_name LongDeleteShortAll --attack_type $ATTACK_TYPE
