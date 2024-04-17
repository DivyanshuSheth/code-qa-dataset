#!/bin/bash
# sbatch /home/dasheth/qa/code-qa-dataset/scripts/slurm/run_sbatch.sh "deepseek"
# sbatch /home/dasheth/qa/code-qa-dataset/scripts/slurm/run_sbatch.sh "mistral"
# sbatch /home/dasheth/qa/code-qa-dataset/scripts/slurm/run_sbatch.sh "magicoder_6.7b"
# sbatch /home/dasheth/qa/code-qa-dataset/scripts/slurm/run_sbatch.sh "codellama_7b"
# sbatch /home/dasheth/qa/code-qa-dataset/scripts/slurm/run_sbatch.sh "codellama_13b"
sbatch /home/dasheth/qa/code-qa-dataset/scripts/slurm/run_sbatch.sh "wizardcoder_13b" "codeqa_questiononly"
sbatch /home/dasheth/qa/code-qa-dataset/scripts/slurm/run_sbatch.sh "wizardcoder_13b" "codeqa"