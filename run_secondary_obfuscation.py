# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import ast
import ast_scope
import re
import random
import string


# obfuscations = ['None', 'Undocument', 'Rename', 'Refactor', 'Dead Code']
obfuscations = ['None', 'Undocument', 'Rename', 'Undocument and Rename']
# prompts = [
#     None,
#     None,
#     None,
#     'Without renaming any existing variables/functions/classes, alter and/or refactor this code without changing its overall behavior. For example, this can be done by reordering certain statements that do not depend on one another, or wrapping reused statements in functions, for example. Only output the full block of code above with your modifications, without any additional outputs.',
#     'Without modifying any existing lines of code, add blocks of dead code that are never executed or have no effect on this program\'s behavior. For example, this can include unused variables, functions, or control structures. Try to hide them by giving them names that make them seem like legitimate live code. Only output the full block of code above with your modifications, without any additional outputs.'
# ]


class Transform(ast.NodeTransformer):
    def __init__(self, globalz):
        super().__init__()
        self.globalz = globalz
        self.map = dict()

    def sub(self, name, defn=False):
        if not defn and name in self.globalz:
            return name
        else:
            if name in self.map.keys():
                eman = self.map[name]
            else:
                # eman = '_' + str(len(self.map))
                eman = '_' + ''.join(random.choices(string.ascii_letters + string.digits, k=16))
                self.map[name] = eman
            return eman

    def visit_arg(self, node):
        name = node.arg
        eman = self.sub(name)
        self.generic_visit(node)
        return ast.arg(**{**node.__dict__, 'arg': eman})

    def visit_Name(self, node):
        name = node.id
        eman = self.sub(name)
        self.generic_visit(node)
        return ast.Name(**{**node.__dict__, 'id': eman})

    def visit_ClassDef(self, node):
        name = node.name
        eman = self.sub(name, defn=True)
        self.generic_visit(node)
        return ast.ClassDef(**{**node.__dict__, 'name': eman})

    def visit_FunctionDef(self, node):
        name = node.name
        eman = self.sub(name, defn=True)
        self.generic_visit(node)
        return ast.FunctionDef(**{**node.__dict__, 'name': eman})

    def visit_AsyncFunctionDef(self, node):
        name = node.name
        eman = self.sub(name, defn=True)
        self.generic_visit(node)
        return ast.AsyncFunctionDef(**{**node.__dict__, 'name': eman})


def substitute(s, d):
    s_split = re.split(r'\W+', s)
    for ss in s_split:
        if ss in d.keys():
            s = re.sub(rf'\b{ss}\b', d[ss], s)
    return s


# def trim(s):
#     lines = s.split('\n')
#     lines = list(filter(lambda l: l.startswith('\t') or l.startswith(' ') or l.startswith('def ') or l.startswith('class '), lines))
#     return '\n'.join(lines)


seed = 11797


if __name__ == '__main__':
    # assert len(obfuscations) == len(prompts)
    random.seed(seed)
    # torch.manual_seed(seed)
    with open('./primary_obfuscation_results.json', 'r') as f:
        data = json.load(f)
    print(data, end='\n\n', flush=True)
    random.seed(seed)
    # tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-33b-instruct')
    # model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-coder-33b-instruct', torch_dtype=torch.bfloat16).cuda()
    results = []
    # cache = dict()
    for datapoint in tqdm(data):
        original = datapoint['original_code']
        code = datapoint['obfuscated_code']
        obf = datapoint['obfuscation']
        category = datapoint['category']
        question = datapoint['question']
        answer = datapoint['answer']
        for obfuscation in obfuscations:
            obfuscated = code
            q = question
            a = answer
            error = False
            if obfuscation == 'Undocument':
                # obfuscated = ast.unparse(ast.parse(code))
                # https://gist.github.com/phpdude/1ae6f19de213d66286c8183e9e3b9ec1
                try:
                    parsed = ast.parse(code)
                except:
                    print("Syntax error in obfuscated code!", flush=True)
                    error = True
                    try:
                        parsed = ast.parse(original)
                    except:
                        print("Syntax error in original code!", flush=True)
                        parsed = None
                if parsed is not None:
                    for node in ast.walk(parsed):
                        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                            continue
                        if not len(node.body):
                            continue
                        if not isinstance(node.body[0], ast.Expr):
                            continue
                        if not hasattr(node.body[0], 'value') or not isinstance(node.body[0].value, ast.Str):
                            continue
                        node.body = node.body[1:]
                    obfuscated = ast.unparse(parsed)
                    # obfuscated = re.sub(r'""".*"""', '\n', obfuscated)
                    # obfuscated = re.sub(r'\'\'\'.*\'\'\'', '\n', obfuscated)
            elif obfuscation == 'Rename':
                # also uncomments
                try:
                    parsed = ast.parse(code)
                except:
                    print("Syntax error in obfuscated code!", flush=True)
                    error = True
                    try:
                        parsed = ast.parse(original)
                    except:
                        print("Syntax error in original code!", flush=True)
                        parsed = None
                if parsed is not None:
                    scope = ast_scope.annotate(parsed)
                    globalz = list(sorted(scope.global_scope.symbols_in_frame))
                    transform = Transform(globalz)
                    obfuscated = ast.unparse(transform.visit(parsed))
                    q = substitute(q, transform.map)
                    a = substitute(a, transform.map)
            elif obfuscation == 'Undocument and Rename':
                try:
                    parsed = ast.parse(code)
                except:
                    print("Syntax error in obfuscated code!", flush=True)
                    error = True
                    try:
                        parsed = ast.parse(original)
                    except:
                        print("Syntax error in original code!", flush=True)
                        parsed = None
                if parsed is not None:
                    for node in ast.walk(parsed):
                        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                            continue
                        if not len(node.body):
                            continue
                        if not isinstance(node.body[0], ast.Expr):
                            continue
                        if not hasattr(node.body[0], 'value') or not isinstance(node.body[0].value, ast.Str):
                            continue
                        node.body = node.body[1:]
                    # obfuscated = ast.unparse(parsed)
                    # try:
                    #     parsed = ast.parse(obfuscated)
                    # except:
                    #     print("Syntax error in obfuscated code!", flush=True)
                    #     error = True
                    #     try:
                    #         parsed = ast.parse(code)
                    #     except:
                    #         print("Syntax error in obfuscated code!", flush=True)
                    #         error = True
                    #         parsed = ast.parse(original)
                    scope = ast_scope.annotate(parsed)
                    globalz = list(sorted(scope.global_scope.symbols_in_frame))
                    transform = Transform(globalz)
                    obfuscated = ast.unparse(transform.visit(parsed))
                    q = substitute(q, transform.map)
                    a = substitute(a, transform.map)
            # if prompt is not None:
            #     if (code, obfuscation) in cache.keys():
            #         obfuscated = cache[(code, obfuscation)]
            #     else:
            #         message = []
            #         message.append({'role': 'user', 'content': '\n\n'.join([code, prompt])})
            #         inputs = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors="pt").cuda()
            #         outputs = model.generate(inputs, max_new_tokens=512, do_sample=True, top_k=50, top_p=0.95, eos_token_id=tokenizer.eos_token_id)
            #         obfuscated = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            #         message.append({'role': 'assistant', 'content': obfuscated})
            #         obfuscated = trim(obfuscated)
            #         cache[(code, obfuscation)] = obfuscated
            result = {
                'original_code': original,
                'primary_obfuscated_code': code,
                'secondary_obfuscated_code': obfuscated,
                'primary_obfuscation': obf,
                'secondary_obfuscation': obfuscation,
                'category': category,
                'question': q,
                'answer': a,
                'error': error
            }
            print(result, flush=True)
            results.append(result)
    with open('./secondary_obfuscation_results.json', 'w') as f:
        json.dump(results, f)
    print('Done!', flush=True)
