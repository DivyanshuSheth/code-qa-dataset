'''
Main inference script for code gen models using VLLM
'''

import os
import re
import json
import copy
import torch
from tqdm import tqdm
import logging
from decoders import build_decoder
from dataset import load_data
from prompts import code_qa_template, code_qa_template_questiononly, make_fewshot_prompt, default_fewshot_examples, summarize_template

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


input_key = 'secondary_obfuscated_code'
output_key = 'generated_answer'

os.environ['HF_HOME'] = '/data/datasets/hf_cache/'
os.environ['HF_DATASETS_CACHE'] = '/data/datasets/hf_cache/'
os.environ['TRANSFORMERS_CACHE'] = '/data/datasets/hf_cache/'


def load_model(args):
    decoder_args = {}
    if args.prompt_template == "codeqa":
        template = code_qa_template
    elif args.prompt_template == "codeqa_questiononly":
        template = code_qa_template_questiononly
    match args.model_name:
        case 'deepseek':
            model = build_decoder("deepseek-ai/deepseek-coder-7b-instruct-v1.5", 'vllm')
            tokenizer = model.get_tokenizer()
            if args.prompt_type == "qa":
                args.prompt_fn = lambda prompt, question: tokenizer.apply_chat_template(
                    [{"role": "user", "content": template.format(prompt=prompt, question=question)}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
        case 'mistral':
            model = build_decoder("mistralai/Mistral-7B-Instruct-v0.2", 'vllm')
            tokenizer = model.get_tokenizer()
            if args.prompt_type == "qa":
                args.prompt_fn = lambda prompt, question: tokenizer.apply_chat_template(
                    [{"role": "user", "content": template.format(prompt=prompt, question=question)}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
        case 'wizardcoder_3b':
            model = build_decoder("WizardLM/WizardCoder-3B-V1.0", 'vllm')
            tokenizer = model.get_tokenizer()
            if args.prompt_type == "qa":
                args.prompt_fn = lambda prompt, question: tokenizer.apply_chat_template(
                    [{"role": "user", "content": template.format(prompt=prompt, question=question)}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
        case 'wizardcoder_13b':
            model = build_decoder("WizardLM/WizardCoder-Python-13B-V1.0", 'vllm')
            tokenizer = model.get_tokenizer()
            if args.prompt_type == "qa":
                args.prompt_fn = lambda prompt, question: tokenizer.apply_chat_template(
                    [{"role": "user", "content": template.format(prompt=prompt, question=question)}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
        case 'starcoder_3b':
            model = build_decoder("bigcode/starcoder", 'vllm')
            tokenizer = model.get_tokenizer()
            if args.prompt_type == "qa":
                args.prompt_fn = lambda prompt, question: tokenizer.apply_chat_template(
                    [{"role": "user", "content": template.format(prompt=prompt, question=question)}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
        case 'santacoder':
            model = build_decoder("bigcode/gpt_bigcode-santacoder", 'vllm')
            tokenizer = model.get_tokenizer()
            if args.prompt_type == "qa":
                args.prompt_fn = lambda prompt, question: tokenizer.apply_chat_template(
                    [{"role": "user", "content": template.format(prompt=prompt, question=question)}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
        case 'codellama_7b':
            model = build_decoder("codellama/CodeLlama-7b-Instruct-hf", 'vllm')
            tokenizer = model.get_tokenizer()
            if args.prompt_type == "qa":
                args.prompt_fn = lambda prompt, question: tokenizer.apply_chat_template(
                    [{"role": "user", "content": template.format(prompt=prompt, question=question)}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
        case 'codellama_13b':
            model = build_decoder("codellama/CodeLlama-13b-Instruct-hf", 'vllm')
            tokenizer = model.get_tokenizer()
            if args.prompt_type == "qa":
                args.prompt_fn = lambda prompt, question: tokenizer.apply_chat_template(
                    [{"role": "user", "content": template.format(prompt=prompt, question=question)}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
        case 'magicoder_6.7b':
            model = build_decoder("ise-uiuc/Magicoder-S-DS-6.7B", 'vllm')
            tokenizer = model.get_tokenizer()
            if args.prompt_type == "qa":
                args.prompt_fn = lambda prompt, question: tokenizer.apply_chat_template(
                    [{"role": "user", "content": template.format(prompt=prompt, question=question)}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
        case 'codet5':
            from transformers import T5ForConditionalGeneration
            model = build_decoder("Salesforce/codet5p-770m-py", "hf", model_cls=T5ForConditionalGeneration)
            if os.path.isfile(args.load_weights):
                logger.info(f"Loading finetuned codet5 weights from {args.load_weights}")
                model.model.load_state_dict(torch.load(args.load_weights))
            else:
                logger.info("Using pretrained codet5 model.")
        case 'gpt-3.5-turbo':
            model = build_decoder("gpt-3.5-turbo", 'openai')
            if args.n_fewshot == 0:
                args.prompt_fn = lambda code: summarize_template.format(code=code)
            else:
                args.prompt_fn = lambda code: make_fewshot_prompt(code, default_fewshot_examples[:args.n_fewshot])
        case _:
            raise Exception(f"Unsupported model: {args.model_name}")
    return model


def inference_batch(args, model, batch_inputs, batch_outputs, generation_config) -> tuple[list[dict], bool]:
    '''
    Args:
        model: model for inference
        batch: list of input dicts
        outputs: list of output dicts, to be modified in-place
        generation_config: model sampling configs
    '''
    batch_prompts = []
    gen_indices = [] # indices of batch members that are to be generated

    for idx, (input_data, output_data) in enumerate(zip(batch_inputs, batch_outputs)):
        if output_key in output_data:
            continue
        prompt = input_data[input_key]
        question = input_data['question']
        gen_indices.append(idx)
        if hasattr(args, 'prompt_fn'):
            prompt = args.prompt_fn(prompt, question)
        batch_prompts.append(prompt)
    
    if len(batch_prompts) == 0:
        return batch_outputs, False

    is_first = not os.path.isfile(args.output_path)
    if is_first:
        # for the first example, print input as sanity check
        logger.info("Prompt:\n" + batch_prompts[0])

    if len(batch_prompts) == 1:
        batch_prompts = batch_prompts[0]
    
    completions, _ = model(batch_prompts, **generation_config)

    if is_first:
        logger.info("Completion:\n" + completions[0])

    for idx, gen_idx in enumerate(gen_indices):
        output_data = batch_outputs[gen_idx]
        curr_completions = completions[idx * args.n_samples:(idx+1) * args.n_samples]
        output_data[output_key] = curr_completions
    
    return batch_outputs, True


def main(args):
    data = load_data(
        args.dataset,
        split=args.split,
        data_start=args.data_start, 
        data_end=args.data_end,
        seed=args.seed,
    )

    model = load_model(args)

    logger.info(f"Writing outputs to file {args.output_path}")
    if os.path.isfile(args.output_path):
        with open(args.output_path, "r") as f:
            outputs = json.load(f)
    else:
        outputs = copy.deepcopy(data)

    generation_config = {
        'use_beam_search': args.use_beam_search,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'n': args.n_samples,
        'max_tokens': args.max_tokens,
    }

    for n_batch, start_idx in enumerate(tqdm(range(0, len(data), args.batch_size), disable=args.disable_tqdm)):
        end_idx = min(start_idx + args.batch_size, len(data))
        batch_inputs = data[start_idx:end_idx]
        batch_outputs = outputs[start_idx:end_idx]
        # print("######### Batch inputs [0] #########")
        # print(batch_inputs[0])
        # print("####################################")
        
        batch_outputs, is_modified = inference_batch(args, model, batch_inputs, batch_outputs, generation_config)

        if is_modified:
            outputs[start_idx:end_idx] = batch_outputs

            # save partial results
            if n_batch % args.save_freq == 0:
                logger.info(f"Saving partial output at batch #{n_batch+1}")
                with open(args.output_path, "w+") as out_fp:
                    json.dump(outputs, out_fp, indent=4)

    with open(args.output_path, "w+") as out_fp:
        json.dump(outputs, out_fp, indent=4)

    logger.info("All done!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, 
        choices=[
            'codet5', # Salesforce/codet5p-770m-py
            'deepseek', # deepseek-ai/deepseek-coder-7b-instruct-v1.5
            'mistral',
            'gpt-3.5-turbo',
            'starcoder_3b',
            "wizardcoder_3b",
            'wizardcoder_13b',
            'santacoder',
            'codellama_7b',
            'codellama_13b',
            'magicoder_6.7b',
        ]
    )
    parser.add_argument("--model_cache_dir", default="/data/datasets/hf_cache/", type=str)
    parser.add_argument("--load_weights", type=str, default=None)
    
    parser.add_argument("--dataset", default="codeqa", type=str)
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--data_start", type=int, default=0)
    parser.add_argument("--data_end", type=int, default=-1)

    # parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--use_beam_search", action='store_true')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=5)

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--save_freq", type=int, default=100)

    parser.add_argument("--disable_tqdm", action='store_true')
    parser.add_argument("--n_fewshot", type=int, default=0)

    parser.add_argument("--prompt_type", type=str, default='qa', choices=['summarize', 'codegen', 'qa'])
    parser.add_argument("--prompt_template", type=str, default=code_qa_template, choices=['codeqa', 'codeqa_questiononly'])

    args = parser.parse_args()
    if args.prompt_template == "codeqa":
        args.output_path = f"/home/dasheth/qa/code-qa-dataset/results/{args.model_name}.json"
    else:
        args.output_path = f"/home/dasheth/qa/code-qa-dataset/results/question_only/{args.model_name}.json"

    main(args)