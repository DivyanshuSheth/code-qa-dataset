code_qa_template_questiononly = '''
Answer the question below. Only output the answer to the question, nothing else. Limit your answer to one sentence.

Question: {question}


Answer in one sentence:'''

code_qa_template = '''
Given the code below, answer the question that follows it. Only output the answer to the question, nothing else. Limit your answer to one sentence.

Code:
```python
{prompt}
```

Question: {question}


Answer in one sentence:'''

codegen_template = '''\
Generate code to complete the following Python function based on its provided summary. Only output the code block.
```python
{prompt}
```

Summary: 
{summary}'''

summarize_template = '''\
Write a one sentence summary of the following Python code:
```python
{code}
```'''

fewshot_summarize_template = '''\
Write a one sentence summary of the Python code.

Here are some examples:
{fewshot_examples}

Your turn!

Input: 
```python
{code}
```
Output:'''

default_fewshot_examples = [
    {
        'input': 'def get_aa_info(code):\n    letter = \'X\'\n    # Try to get content from PDBE.\n    url_string = \'http://www.ebi.ac.uk/pdbe-srv/pdbechem/chemicalCompound/show/{0}\'.format(code)\n    r = requests.get(url_string)\n    # Raise error if content not obtained.\n    if not r.ok:\n        raise IOError("Could not get to url {0}".format(url_string))\n\n    # Parse r.text in an ugly way to get the required information.\n    description = r.text.split(\'<h3>Molecule name\')[1].split(\'</tr>\')[0]\n    description = description.strip().split(\'\\n\')[3].strip()[:255]\n    modified = r.text.split("<h3>Standard parent ")[1].split(\'</tr>\')[0]\n    modified = modified.replace(" ", "").replace(\'\\n\', \'\').split(\'<\')[-3].split(\'>\')[-1]\n    if modified == "NotAssigned":\n        modified = None\n    # Add the required information to a dictionary which can then be passed to add_amino_acid_to_json.\n    aa_dict = {\'code\': code, \'description\': description, \'modified\': modified, \'letter\': letter}\n    return aa_dict', 
        'output': 'Get dictionary of information relating to a new amino acid code not currently in the database.'
    },
    {
        'input': 'def _build_filters(_filters):\n    root = utils.NestedDict({})\n    for _filter in _filters:\n        operation = None\n        for operation, token in SPLIT_TOKENS:\n            # split "some.key=value" into ["some.key", "value"]\n            top_parts = _filter.split(token, 1)\n            if len(top_parts) == 2:\n                break\n        else:\n            raise exceptions.CLIAbort(\'Failed to find valid operation for: %s\'\n                                      % _filter)\n\n        key, value = top_parts\n        current = root\n        # split "some.key" into ["some", "key"]\n        parts = [part.strip() for part in key.split(\'.\')]\n\n        # Actually drill down and add the filter\n        for part in parts[:-1]:\n            current = current[part]\n\n        if operation == \'eq\':\n            current[parts[-1]] = utils.query_filter(value.strip())\n        elif operation == \'in\':\n            current[parts[-1]] = {\n                \'operation\': \'in\',\n                \'options\': [{\n                    \'name\': \'data\',\n                    \'value\': [p.strip() for p in value.split(\',\')],\n                }],\n            }\n\n    return root.to_dict()',
        'output': 'Builds filters using the filter options passed into the CLI.'
    },
    {
        'input': 'def get_quantities(self, quantities, filters=None, native_filters=None, return_iterator=False):\n        quantities = self._preprocess_requested_quantities(quantities)\n        filters = self._preprocess_filters(filters)\n        native_filters = self._preprocess_native_filters(native_filters)\n\n        it = self._get_quantities_iter(quantities, filters, native_filters)\n\n        if return_iterator:\n            return it\n\n        data_all = defaultdict(list)\n        for data in it:\n            for q in quantities:\n                data_all[q].append(data[q])\n        return {q: concatenate_1d(data_all[q]) for q in quantities}', 
        'output': 'Fetch quantities from this catalog.'
    },
    {
        'input': "def wind_shear(shear: str, unit_alt: str = 'ft', unit_wind: str = 'kt', spoken: bool = False) -> str:\n    if not shear or 'WS' not in shear or '/' not in shear:\n        return ''\n    shear = shear[2:].rstrip(unit_wind.upper()).split('/')  # type: ignore\n    wdir = core.spoken_number(shear[1][:3]) if spoken else shear[1][:3]\n    return f'Wind shear {int(shear[0])*100}{unit_alt} from {wdir} at {shear[1][3:]}{unit_wind}'", 
        'output': 'Translate wind shear into a readable string'
    },
    {
        'input': 'def creates_cycle(connections, test):\n    i, o = test\n    if i == o:\n        return True\n\n    visited = {o}\n    while True:\n        num_added = 0\n        for a, b in connections:\n            if a in visited and b not in visited:\n                if b == i:\n                    return True\n\n                visited.add(b)\n                num_added += 1\n\n        if num_added == 0:\n            return False', 
        'output': 'Returns true if the addition of the test connection would create a cycle assuming that no cycle already exists in the graph represented by connections.'
    }
]

fewshot_example_template = '''\
Input:
```python
{input}
```
Output: {output}'''

def make_fewshot_prompt(code, fewshot_examples):
    fewshot_text = []
    for ex in fewshot_examples:
        fewshot_text.append(fewshot_example_template.format(
            input=ex['input'], output=ex['output']
        ))
    fewshot_text = "\n\n".join(fewshot_text)
    return fewshot_summarize_template.format(
        fewshot_examples=fewshot_text,
        code=code
    )