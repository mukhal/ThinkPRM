#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=30g
#SBATCH --time=02-10:00:00
#SBATCH --account=wangluxy1
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1

'''Usage:
# best of n on the MATH-500 dataset
sbatch recipes/launch_array.slurm recipes/Llama-3.2-1B-Instruct/best_of_n.yaml \
    --hub_dataset_id=<YOUR_ORG>/Llama-3.2-1B-Instruct-bon-completions
'''

source ~/.bashrc
set -x -e
conda activate sal

# Define the array of input files (assuming 10 input files)
STEP=50
ENDPOINT=$((SLURM_ARRAY_TASK_COUNT * STEP - STEP))
STARTS=($(seq 0 $STEP $ENDPOINT))  # Generate sequence from 0 to ENDPOINT with a step size of 100
# Use the SLURM_ARRAY_TASK_ID to pick the correct input file
INPUT_FILE=${INPUT_FILES[$SLURM_ARRAY_TASK_ID-1]}

DATASET_START=${STARTS[$SLURM_ARRAY_TASK_ID-1]}
DATASET_END=$((${STARTS[$SLURM_ARRAY_TASK_ID-1]}+$STEP))

python scripts/test_time_compute.py "$@" \
    --dataset_start=$DATASET_START \
    --dataset_end=$DATASET_END \
    --push_to_hub