# Script to run standard training + attack in CodaLab

# Activate at-env conda environment
source activate atenv
# Missing package?
pip install query
# conda list

# Needed for CodaLab as typo-embeddings is mounted one level down
cd typo-embeddings || exit 1

# Set required environment variables
export TASK_NAME=MRPC
export CLUSTERER_PATH=$HOME/clusterers/vocab100000_ed1.pkl
export GLUE_DIR=$HOME/data/glue_data

# Make clusterer
echo 'Constructing a clusterer...'
python construct_clusters.py --vocab_size 100000 --perturb_type ed1 --data_dir $HOME/data --output_dir $HOME/clusterers

# Train and Attack
echo 'Training and then attacking...'
python run_glue.py --task_name $TASK_NAME --do_lower_case --do_train --do_eval --data_dir $GLUE_DIR/$TASK_NAME --output_dir $HOME/model_output/$TASK_NAME --save_results --save_dir $HOME/train --recoverer identity --augmentor identity --run_test
python run_glue.py --task_name $TASK_NAME --do_lower_case --do_eval --data_dir $GLUE_DIR/$TASK_NAME --output_dir $HOME/model_output/$TASK_NAME  --save_results --save_dir $HOME/attack --recoverer identity --augmentor identity --run_test --model_name_or_path $HOME/model_output/$TASK_NAME  --attack --new_attack --attacker beam-search --beam_width 5 --attack_name LongDeleteShortAll --attack_type ed1

