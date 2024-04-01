import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datasets import load_dataset
from tqdm import tqdm


# Chenxiao Liu, & Xiaojun Wan. (2021). CodeQA: A Question Answering Dataset for Source Code Comprehension.
# https://arxiv.org/abs/2109.08365
categories = ['General', 'Functionality', 'Purpose', 'Property', 'Workflow', 'Edge Cases', 'Performance and Scalability', 'Control Flow', 'OOP Concepts']
pre_prompts = [
    None,
    None,
    None,
    None,
    None,
    'Explain the concept of edge cases in programming.'
    'Explain the concepts of performance and scalability in programming.'
    'Explain the concept of control flow in programming.'
    'List and explain key concepts in object-oriented programming (OOP).'
]
question_prompts = [
    'In one sentence, generate an answerable question based on this code. Only output the question.',
    'Functionality provides a definition of the range of functions that the subject and/or its interface can perform. In one sentence, generate an answerable question related to functionality based on this code. Only output the question.',
    'Purpose explains the reason why the subject is provided or the design rationale of the subject. In one sentence, generate an answerable question related to purpose based on this code. Only output the question.',
    'Property declares properties of the subject, such as pre-conditions and post-conditions of a method or some statements. In one sentence, generate an answerable question related to property based on this code. Only output the question.',
    'Workflow describes how the subject is done, which means implementation details like the design or the workflow of the subject. In one sentence, generate an answerable question related to workflow based on this code. Only output the question.',
    'In one sentence, generate an answerable question related to edge cases based on this code. Only output the question.',
    'In one sentence, generate an answerable question related to performance and scalability based on this code. Only output the question.',
    'In one sentence, generate an answerable question related to control flow based on this code. Only output the question.',
    'In one sentence, generate an answerable question related to object-oriented programming (OOP) concepts based on this code. Only output the question.'
]
answer_prompt = 'In one sentence, provide the correct answer to this question based on the code. Only output the answer.'
split = 'train'
# seed = 42
seed = 11797
n = 100


if __name__ == '__main__':
    assert len(categories) == len(question_prompts)
    torch.manual_seed(seed)
    dataset = load_dataset('code_search_net', 'python')
    datapoints = dataset[split].shuffle(seed=seed)[:n]['whole_func_string']
    for datapoint in datapoints:
        print(datapoint, end='\n\n', flush=True)
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-33b-instruct')
    model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-coder-33b-instruct', torch_dtype=torch.bfloat16).cuda()
    results = []
    for datapoint in tqdm(datapoints):
        for category, pre, prompt in zip(categories, pre_prompts, question_prompts):
            message = []
            if pre is not None:
                message.append({'role': 'user', 'content': pre})
                inputs = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors="pt").cuda()
                outputs = model.generate(inputs, max_new_tokens=512, do_sample=True, top_k=50, top_p=0.95, eos_token_id=tokenizer.eos_token_id)
                context = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
                message.append({'role': 'assistant', 'content': context})
            message.append({'role': 'user', 'content': '\n\n'.join([datapoint, prompt])})
            inputs = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors="pt").cuda()
            outputs = model.generate(inputs, max_new_tokens=512, do_sample=True, top_k=50, top_p=0.95, eos_token_id=tokenizer.eos_token_id)
            question = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            message.append({'role': 'assistant', 'content': question})
            message.append({'role': 'user', 'content': answer_prompt})
            inputs = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors="pt").cuda()
            outputs = model.generate(inputs, max_new_tokens=512, do_sample=True, top_k=50, top_p=0.95, eos_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            message.append({'role': 'assistant', 'content': answer})
            result = {
                'code': datapoint,
                'category': category,
                'question': question,
                'answer': answer
            }
            print(result, flush=True)
            results.append(result)
    with open('./generation_results.json', 'w') as f:
        json.dump(results, f)
    print('Done!', flush=True)
