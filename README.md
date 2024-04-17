# Codexplain: A Benchmark Dataset for Code Question Answering
## Course Project for 11-797: Question Answering | Team Members: Divyanshu Sheth, Santiago Benoit

The aim of the project is to create a benchmark dataset for code question answering leveraging large language models, followed by a thorough evaluation of performances of current code generation models on it. This repository contains the source code developed in the project.

## Installation
Create a new Python environment and install required dependencies using ```pip -r install requirements.txt```. The tested version of Python for the code is Python 3.11.5.

## Run QA Generation

## Run Model Evaluation
Run ```scripts/slurm/call_sbatch_run.sh``` or ```scripts/run_models_1.sh``` with the appropriate arguments to run various models on the created QA dataset. The current list of supported models is the following (Hugging Face model tags):
- deepseek-ai/deepseek-coder-7b-instruct-v1.5
- mistralai/Mistral-7B-Instruct-v0.2
- WizardLM/WizardCoder-3B-V1.0
- WizardLM/WizardCoder-Python-13B-V1.0
- bigcode/starcoder
- bigcode/gpt_bigcode-santacoder
- codellama/CodeLlama-7b-Instruct-hf
- codellama/CodeLlama-13b-Instruct-hf
- ise-uiuc/Magicoder-S-DS-6.7B
