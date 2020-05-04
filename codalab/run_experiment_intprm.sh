# Script to run standard training + attack in CodaLab
# Use: bash run_experiment.sh task_name recoverer augmentor [clusterer_path/tc_dir]
# Fourth argument is optional: use tc_dir when recoverer is scrnn, clusterer_path when recoverer starts with clust.

# Activate at-env conda environment
source activate erik-cert

# Missing package?

# Set required environment variables
export TASK_NAME=$1
export RECOVERER=$2
export AUGMENTOR=$3
export DO_ROBUST=''
export ATTACK_TYPE='intprm'
export DO_ATTACK='true'

# Add --do_robust to compute robust accuracy when using clusters to defend
if [ "$RECOVERER" = 'clust-intprm' ]; then
  export DO_ROBUST='--do_robust'
  #Attack does not change performance, since can sort interior of string...
  export DO_ATTACK='false'
fi

export CLUSTERER_PATH=$HOME/$4
# When the fourth argument is present, change the cluster path to $HOME/$5
# WARNING: to change clusterer path, recoverer can't be scrnn (wouldn't make a difference anyways')
export GLUE_DIR=$HOME/data/glue_data

# Needed for CodaLab as roben is mounted one level down
cd roben_intprm || exit 1

# Training + attacking
echo 'Training and then attacking...'
python run_glue.py --log_stdout_only --task_name $TASK_NAME --do_lower_case --do_train --do_eval --data_dir $GLUE_DIR/$TASK_NAME --output_dir $HOME/model_output/$TASK_NAME --overwrite_output_dir --save_results --save_dir $HOME/train --recoverer $RECOVERER --augmentor $AUGMENTOR --run_test $DO_ROBUST --clusterer_path $CLUSTERER_PATH
if [ "$DO_ATTACK" = 'true' ]; then
  python run_glue.py --log_stdout_only --task_name $TASK_NAME --do_lower_case --do_eval --data_dir $GLUE_DIR/$TASK_NAME --output_dir $HOME/model_output/$TASK_NAME  --save_results --save_dir $HOME/attack --recoverer $RECOVERER --augmentor $AUGMENTOR --run_test --clusterer_path $CLUSTERER_PATH --model_name_or_path $HOME/model_output/$TASK_NAME  --attack --new_attack --attacker beam-search --beam_width 5 --attack_name RandomPerturbationAttack --attack_type $ATTACK_TYPE
fi
