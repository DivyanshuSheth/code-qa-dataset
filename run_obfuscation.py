import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import ast
import re


obfuscations = ['None', 'Undocument', 'Rename', 'Refactor', 'Dead Code']
prompts = [
    None,
    None,
    None,
    'Without renaming any existing variables/functions/classes, alter and/or refactor this code without changing its overall behavior. For example, this can be done by reordering certain statements that do not depend on one another, or wrapping reused statements in functions, for example. Only output the full block of code above with your modifications, without any additional outputs.',
    'Without modifying any existing lines of code, add blocks of dead code that are never executed or have no effect on this program\'s behavior. For example, this can include unused variables, functions, or control structures. Try to hide them by giving them names that make them seem like legitimate live code. Only output the full block of code above with your modifications, without any additional outputs.'
]


class Transform(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.map = dict()

    def sub(self, name):
        if name in self.map.keys():
            eman = self.map[name]
        else:
            eman = '_' + str(len(self.map))
            self.map[name] = eman
        return eman

    def visit_arg(self, node):
        name = node.arg
        eman = self.sub(name)
        return ast.arg(**{**node.__dict__, 'arg': eman})

    def visit_Name(self, node):
        name = node.id
        eman = self.sub(name)
        return ast.Name(**{**node.__dict__, 'id': eman})

    def visit_ClassDef(self, node):
        name = node.name
        eman = self.sub(name)
        return ast.ClassDef(**{**node.__dict__, 'name': eman})

    def visit_FunctionDef(self, node):
        name = node.name
        eman = self.sub(name)
        return ast.FunctionDef(**{**node.__dict__, 'name': eman})

    def visit_AsyncFunctionDef(self, node):
        name = node.name
        eman = self.sub(name)
        return ast.AsyncFunctionDef(**{**node.__dict__, 'name': eman})


def substitute(s, d):
    s_split = re.split(r'\W+', s)
    for ss in s_split:
        if ss in d.keys():
            s = re.sub(rf'\b{ss}\b', d[ss], s)
    return s


if __name__ == '__main__':
    assert len(obfuscations) == len(prompts)
    with open('./generation_results.json', 'r') as f:
        data = json.load(f)
    print(data, end='\n\n', flush=True)
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-33b-instruct')
    model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-coder-33b-instruct', torch_dtype=torch.bfloat16).cuda()
    results = []
    cache = dict()
    for datapoint in tqdm(data):
        code = datapoint['code']
        category = datapoint['category']
        question = datapoint['question']
        answer = datapoint['answer']
        for obfuscation, prompt in zip(obfuscations, prompts):
            obfuscated = code
            q = question
            a = answer
            if obfuscation == 'Undocument':
                obfuscated = ast.unparse(ast.parse(code))
                obfuscated = re.sub(r'\"\"\".*\"\"\"', '\n', obfuscated)
                obfuscated = re.sub(r'\'\'\'.*\'\'\'', '\n', obfuscated)
            if obfuscation == 'Rename':
                # also uncomments
                transform = Transform()
                obfuscated = ast.unparse(transform.visit(ast.parse(code)))
                q = substitute(q, transform.map)
                a = substitute(a, transform.map)
            if prompt is not None:
                if (code, obfuscation) in cache.keys():
                    obfuscated = cache[(code, obfuscation)]
                else:
                    message = []
                    message.append({'role': 'user', 'content': '\n\n'.join([code, prompt])})
                    inputs = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors="pt").cuda()
                    outputs = model.generate(inputs, max_new_tokens=512, do_sample=True, top_k=50, top_p=0.95, eos_token_id=tokenizer.eos_token_id)
                    obfuscated = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
                    message.append({'role': 'assistant', 'content': obfuscated})
                    cache[(code, obfuscation)] = obfuscated
            result = {
                'original_code': code,
                'obfuscated_code': obfuscated,
                'obfuscation': obfuscation,
                'category': category,
                'question': q,
                'answer': a
            }
            print(result, flush=True)
            results.append(result)
    with open('./obfuscation_results.json', 'w') as f:
        json.dump(results, f)
    print('Done!', flush=True)
