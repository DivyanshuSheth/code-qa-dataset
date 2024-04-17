import json
import logging
import random
import re
from tqdm import tqdm
from nltk.tokenize.punkt import PunktSentenceTokenizer
import datasets
import string
from typing import List


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

REGEX_SPECIAL = "\\, ., +, *, ?, ^, $, (, ), [, ], {, }, |".split(", ")
triple_quotes = '("""' + "|" + '"' + "|" + "'" + "|" + "''')"
def build_comment_regex(comment):
    for sp in REGEX_SPECIAL:
        comment = comment.replace(sp, f"\\{sp}")
    return '\s*\w*r?u?' + triple_quotes + r'\s*' + comment + r'\s*' + triple_quotes + r"\s*\n"


def parse_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                json_data = json.loads(line.strip())
                data.append(json_data)
            except json.JSONDecodeError as e:
                logger.info(f"Error decoding line: {e}")
    return data

def remove_space_before_punc(text):
    return re.sub(r'\s([?.!"](?:\s|$))', r'\1', text)


def create_entry_point(
    intent: str, 
    first_nwords: int = 4, 
    stop_words: List[str] = ["in", "of", "a", "to", "and", "for", "with", "that"], 
    delimiters: List[str] = ["`", "'"], 
    verbose: bool = False, 
) -> str: 
    """Heuristically assign (meaningful) function name from the rewritten-intent."""
    words = [w.lower() for w in intent.split()[: first_nwords]]
    if verbose: print(f"[words 0] {words}")
    for idx, word in enumerate(words):
        try: 
            word_num = float(word)
            break 
        except: 
            continue 
    else: idx += 1
    words = words[: idx]
    if verbose: print(f"[words 1] {words}")
  
    for idx, word in enumerate(words): 
        if any([word==sw for sw in stop_words]) and idx > 1: 
            break
    else: idx += 1
    words = words[: idx]
    if verbose: print(f"[words 2] {words}")  
    for idx, word in enumerate(words): 
        if any([word.startswith(de) for de in delimiters]): 
            break
    else: idx += 1
    words = words[: idx]
    if verbose: print(f"[words 3] {words}")
    words = [''.join([c for c in word if (not c in string.punctuation)]) for word in words]
    words = [word for word in words if word.strip()]
    if verbose: print(f"[words 4] {words}")
    if len(words) < 2: 
        words = ["f"] + words[: first_nwords]
    if words[0].startswith("¿"): words[0] = words[0][1: ]
    return '_'.join(words)


def replace_entry_point(function_head: str, alternative_name: str) -> str: 
    """Replace the default function name to semantically meaningful one.
    E.g., "f_12345" -> "count_items" 
    args: 
        function_head: "def f_3844801(myList):"
        description: e.g., "check if all elements in list are identical"
    rets: 
        sema_function_head: e.g. "def check_elements_identical(myList):"
    """
    arguments = function_head[function_head.index('('): ]
    return f"def {alternative_name}{arguments}"


bad = []
def load_data(dataset_name, split='test', data_start=0, data_end=-1, seed=14):
    '''
    Returns data - list of dicts, each of which has 'input' and 'output' keys
    For CodeSearchNet, data is pseudorandomly shuffled/subsampled from original data
    '''
    global bad
    match dataset_name:
        case 'csn':
            # use CSN data cleaned with https://github.com/BuiltOntheRock/FSE22_BuiltOntheRock
            if split == 'train':
                # TODO: only for train_codet5_v3
                data = parse_jsonl("data/csn_python_train.jsonl")
            else:
                data = parse_jsonl(f"data/cleaned_csn_python_{split}.jsonl")

            if seed > 0: # default seed is 14
                random.seed(seed) # take a "random" shuffle of data
                random.shuffle(data)

            if data_end == -1: # default end is 500
                data_end = len(data)

            data = data[data_start:data_end]

            bad_comment_starts = ['r"""', "r'''", 'u"""']
            bad_tokenized_comment_starts = ['r ', 'u ']
            for i, ex in enumerate(tqdm(data, desc='Processing data')):
                for bad_comment_start in bad_comment_starts:
                    if ex['docstring'].startswith(bad_comment_start):
                        ex['docstring'] = ex['docstring'][len(bad_comment_start):].lstrip()
                        break
                
                ex['code'] = re.sub(build_comment_regex(ex['docstring']), '\n', ex['code'], flags=re.DOTALL)
                if ex['docstring'] in ex['code']:
                    bad.append(i)
                
                ex['docstring'] = " ".join(ex['docstring_tokens'])
                # correct formatting errors
                ex['docstring'] = remove_space_before_punc(ex['docstring'])
                ex['docstring'] = ex['docstring'].replace(" - ",  "-")
                for bad_start in bad_tokenized_comment_starts:
                    if ex['docstring'].startswith(bad_start):
                        ex['docstring'] = ex['docstring'][len(bad_start):].lstrip()
                        break

            data = [
                {'input': ex.pop('code'), 'output': ex.pop('docstring'), **ex}
                for ex in data
            ]
        case 'codecontests':
            raise NotImplementedError()
        
        case 'odex':
            data = datasets.load_dataset("neulab/odex", split='test')
            data = [{
                'input': (
                    replace_entry_point(
                        ex['prompt'],
                        create_entry_point(ex['intent'])
                    ) + ex['canonical_solution'] + ex['suffix']
                ).replace("\t", " " * 4).strip(),
                'output': ex['intent'],
                **ex
            } for ex in data]

        case 'codeqa':
            with open("/home/dasheth/qa/code-qa-dataset/data/secondary_obfuscation_results.json", "r") as f:
                data = json.load(f)

        case _:
            raise Exception(f"Unknown dataset: {dataset_name}")
    
    return data


def get_first_sentence(text):
    tokenizer = PunktSentenceTokenizer()
    tokenizer.train(text)
    sentences = tokenizer.tokenize(text)
    if len(sentences) == 0:
        return text
    return sentences[0]


if __name__ == '__main__':
    from argparse import Namespace
    args = Namespace(dataset='csn', split='test', data_start=0, data_end=10000, seed=14)
    data = load_data(
        args.dataset,
        args.split,
        args.data_start,
        args.data_end,
        args.seed,
    )

    print(data[0]['input'])

    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from train.train_utils import preprocess_batch

    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5p-770m-py')
    collate_fn = lambda batch: preprocess_batch(batch, tokenizer)

    loader = DataLoader(data, shuffle=False, batch_size=32,
                        num_workers=0, collate_fn=collate_fn)
    
    maxlen = 0
    for x, y in loader:
        maxlen = max(maxlen, y.shape[1])
    