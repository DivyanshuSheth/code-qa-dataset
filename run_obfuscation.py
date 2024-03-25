import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm


# obfuscations = ['None', 'Comments', 'Rename', 'Refactor', 'Dead Code']
obfuscations = ['None', 'Refactor', 'Dead Code']
prompts = [
    None,
    'Without renaming any existing variables/functions/classes, alter and/or refactor this code without changing its overall behavior. This can be done by reordering certain statements that do not depend on one another, or wrapping reused statements in functions, for example. Output the full block of code above with your modifications, without any additional outputs.',
    'Without modifying any existing lines of code, add blocks of code that are never executed or have no effect on this program\'s behavior. This can include unused variables, functions, or control structures. Output the full block of code above with your modifications, without any additional outputs.'
]


if __name__ == '__main__':
    assert len(obfuscations) == len(prompts)
    with open('./generation_results.json', 'r') as f:
        data = json.load(f)
    print(data, end='\n\n', flush=True)
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-33b-instruct')
    model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-coder-33b-instruct', torch_dtype=torch.bfloat16).cuda()
    results = []
    for datapoint in tqdm(data):
        code = datapoint['code']
        category = datapoint['category']
        question = datapoint['question']
        answer = datapoint['answer']
        for obfuscation, prompt in zip(obfuscations, prompts):
            obfuscated = code
            if obfuscation != 'None':
                message = []
                message.append({'role': 'user', 'content': '\n\n'.join([code, prompt])})
                inputs = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors="pt").cuda()
                outputs = model.generate(inputs, max_new_tokens=512, do_sample=True, top_k=50, top_p=0.95, eos_token_id=tokenizer.eos_token_id)
                obfuscated = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
                message.append({'role': 'assistant', 'content': obfuscated})
            result = {
                'original_code': code,
                'obfuscated_code': obfuscated,
                'obfuscation': obfuscation,
                'category': category,
                'question': question,
                'answer': answer
            }
            print(result, flush=True)
            results.append(result)
    with open('./obfuscation_results.json', 'w') as f:
        json.dump(results, f)
    print('Done!', flush=True)
