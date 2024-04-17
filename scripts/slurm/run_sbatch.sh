#!/bin/bash

#SBATCH --job-name=test_job
#SBATCH --gres=gpu:A6000:1                    # Request 1 A6000 GPUs
#SBATCH --nodes=1                       # Request 1 node
#SBATCH --mem=30G                       # Adjust the memory per node as needed
#SBATCH --time=0-06:00                  # Time limit in D-HH:MM
#SBATCH --output=/home/dasheth/slurm_out/qa_project/%j.out


# # # Load any necessary modules
# module load cuda/11.2    # Specify the version of CUDA you need

MODEL=$1
TEMPLATE=$2

source activate /home/dasheth/miniconda3/envs/directed

cd /home/dasheth/qa/code-qa-dataset

python run_vllm_models.py --model_name $MODEL --prompt_template $TEMPLATE

