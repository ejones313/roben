# Script to run standard training + attack in CodaLab

# Activate at-env conda environment
source activate atenv
# Missing package?
pip install query
# conda list

# Set required environment variables
export TASK_NAME=MRPC
export CLUSTERER_PATH=$HOME/run-bash/clusterers/vocab100000_ed1.pkl
export GLUE_DIR=$HOME/data/glue_data

# Needed for CodaLab as typo-embeddings is mounted one level down
cd typo-embeddings || exit 1

# Clusters as defense
echo 'Defending...'
python run_glue.py --log_stdout_only --do_robust --task_name $TASK_NAME --do_lower_case --do_train --do_eval --data_dir $GLUE_DIR/$TASK_NAME --output_dir $HOME/model_output/$TASK_NAME --save_results --save_dir $HOME/defense --recoverer clust-rep --clusterer_path $CLUSTERER_PATH --augmentor identity --run_test