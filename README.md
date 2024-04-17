# Codexplain: A Benchmark Dataset for Code Question Answering
## Course Project for 11-797: Question Answering | Team Members: Divyanshu Sheth, Santiago Benoit

The aim of the project is to create a benchmark dataset for code question answering leveraging large language models, followed by a thorough evaluation of performances of current code generation models on it. This repository contains the source code developed in the project.

## Installation
Create a new Python environment and install required dependencies using ```pip -r install requirements.txt```. The tested version of Python for the code is Python 3.11.5.

## Run QA Generation Pipeline
Edit hyperparameters inside script, then:
```python run_generation.py```
Output file: generation_results.json

## Run Obfuscation Pipeline
Edit hyperparameters inside scripts, then:
```python run_primary_obfuscation.py && python run_secondary_obfuscation.py```
Output files: primary_obfuscation_results.json, secondary_obfuscation_results.json

## Complete Dataset
The complete dataset (normal and obfuscated versions) is available at this link: https://drive.google.com/file/d/1qPtnPjs_w0c4o24f5kPV5dUf3-yp2goh/view?usp=sharing. The file in ```data/``` is a subset of the complete data, with 14400 examples (100 code files, with various questions and obfuscations).

## Run Code Models on the Dataset
Run ```sbatch scripts/slurm/call_sbatch_run.sh``` or ```bash scripts/run_models_1.sh``` with the appropriate arguments to run various models on the created QA dataset. The current list of supported models is the following (Hugging Face model tags):
- deepseek-ai/deepseek-coder-7b-instruct-v1.5
- mistralai/Mistral-7B-Instruct-v0.2
- WizardLM/WizardCoder-3B-V1.0
- WizardLM/WizardCoder-Python-13B-V1.0
- bigcode/starcoder
- bigcode/gpt_bigcode-santacoder
- codellama/CodeLlama-7b-Instruct-hf
- codellama/CodeLlama-13b-Instruct-hf
- ise-uiuc/Magicoder-S-DS-6.7B

## Compute Evaluation Metrics
Run ```bash scripts/run_evaluate.sh``` with appropriate arguments to compute evaluation metrics (BERTScore) on the QA pairs obtained in the last step. Results will be saved in the ```results/``` directory.
