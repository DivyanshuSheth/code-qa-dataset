import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
# from datasets import load_dataset


datapoints = [
r'''def export_hcurves_by_imt_csv(
        key, kind, rlzs_assoc, fname, sitecol, array, oq, checksum):
    """
    Export the curves of the given realization into CSV.

    :param key: output_type and export_type
    :param kind: a string with the kind of output (realization or statistics)
    :param rlzs_assoc: a :class:`openquake.commonlib.source.RlzsAssoc` instance
    :param fname: name of the exported file
    :param sitecol: site collection
    :param array: an array of shape (N, L) and dtype numpy.float32
    :param oq: job.ini parameters
    """
    nsites = len(sitecol)
    fnames = []
    for imt, imls in oq.imtls.items():
        slc = oq.imtls(imt)
        dest = add_imt(fname, imt)
        lst = [('lon', F32), ('lat', F32), ('depth', F32)]
        for iml in imls:
            lst.append(('poe-%s' % iml, F32))
        hcurves = numpy.zeros(nsites, lst)
        for sid, lon, lat, dep in zip(
                range(nsites), sitecol.lons, sitecol.lats, sitecol.depths):
            hcurves[sid] = (lon, lat, dep) + tuple(array[sid, slc])
        fnames.append(writers.write_csv(dest, hcurves, comment=_comment(
            rlzs_assoc, kind, oq.investigation_time) + (
                ', imt="%s", checksum=%d' % (imt, checksum)
            ), header=[name for (name, dt) in lst]))
    return fnames''',
r'''def get_surface_vertexes(cls, fault_trace,
                         upper_seismogenic_depth,
                         lower_seismogenic_depth, dip):
    """
    Get surface main vertexes.

    Parameters are the same as for :meth:`from_fault_data`, excluding
    mesh spacing.

    :returns:
        Instance of :class:`~openquake.hazardlib.geo.polygon.Polygon`
        describing the surface projection of the simple fault with
        specified parameters.
    """
    # Similar to :meth:`from_fault_data`, we just don't resample edges
    dip_tan = math.tan(math.radians(dip))
    hdist_top = upper_seismogenic_depth / dip_tan
    hdist_bottom = lower_seismogenic_depth / dip_tan

    strike = fault_trace[0].azimuth(fault_trace[-1])
    azimuth = (strike + 90.0) % 360

    # Collect coordinates of vertices on the top and bottom edge
    lons = []
    lats = []
    for point in fault_trace.points:
        top_edge_point = point.point_at(hdist_top, 0, azimuth)
        bottom_edge_point = point.point_at(hdist_bottom, 0, azimuth)
        lons.append(top_edge_point.longitude)
        lats.append(top_edge_point.latitude)
        lons.append(bottom_edge_point.longitude)
        lats.append(bottom_edge_point.latitude)

    lons = numpy.array(lons, float)
    lats = numpy.array(lats, float)
    return lons, lats''',
r'''def add_tags(self, dic, prefix):
    """
    :param dic: a dictionary tagname -> tagvalue
    :returns: a list of tag indices, one per tagname
    """        
    # fill missing tagvalues with "?", raise an error for unknown tagnames
    idxs = []
    for tagname in self.tagnames:
        if tagname in ('exposure', 'country'):
            idxs.append(self.add(tagname, prefix))
            continue
        try:
            tagvalue = dic.pop(tagname)
        except KeyError:
            tagvalue = '?'
        else:
            if tagvalue in '?*':
                raise ValueError(
                    'Invalid tagvalue="%s"' % tagvalue)
        idxs.append(self.add(tagname, tagvalue))
    if dic:
        raise ValueError(
            'Unknown tagname %s or <tagNames> not '
            'specified in the exposure' % ', '.join(dic))
    return idxs'''
]
# prompts = [
#     'Generate 3 questions and answers (up to 1 sentence each) about this code.',
#     'Generate 3 "why" questions and answers (up to 1 sentence each) about this code.',
#     'Generate 3 questions and answers (up to 1 sentence each) about edge cases in this code.',
#     'Generate 3 questions and answers (up to 1 sentence each) about functionality in this code.',
#     'Generate 3 extractive questions (up to 1 sentence each) with answer spans from this code.'
# ]
categories = ['general', 'why', 'edge cases', 'functionality', 'extractive']
question_prompts = [
    'In one sentence, generate an answerable question based on this code. Only output the question.',
    'In one sentence, generate an answerable "why" question based on this code. Only output the question.',
    'In one sentence, generate an answerable question about edge cases based on this code. Only output the question.',
    'In one sentence, generate an answerable question about functionality based on this code. Only output the question.',
    'In one sentence, generate an extractive question that can be answered with a one-line span extracted directly from this code. Only output the question.'
]
next_prompts = [
    'In one sentence, generate another answerable question based on the code. Only output the question.',
    'In one sentence, generate another answerable "why" question based on the code. Only output the question.',
    'In one sentence, generate another answerable question about edge cases based on the code. Only output the question.',
    'In one sentence, generate another answerable question about functionality based on the code. Only output the question.',
    'In one sentence, generate another extractive question that can be answered with a one-line span extracted directly from the code. Only output the question.'
]
extractive_prompt = 'What is extractive question answering?'
answer_prompt = 'In one sentence, generate the correct answer to this question based on the code. Only output the answer.'
span_prompt = 'Provide a one-line span extracted directly from the code which contains the correct answer to this question. Only output the answer span.'
n = 3


if __name__ == '__main__':
    # dataset = load_dataset("code_search_net", "python")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-33b-instruct")
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-33b-instruct", torch_dtype=torch.bfloat16).cuda()
    results = []
    for datapoint in datapoints:
        for category, prompt, nxt in zip(categories, question_prompts, next_prompts):
            message = []
            if category == 'extractive':
                message.append({'role': 'user', 'content': extractive_prompt})
                inputs = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors="pt").cuda()
                outputs = model.generate(inputs, max_new_tokens=512, do_sample=True, top_k=50, top_p=0.95, eos_token_id=tokenizer.eos_token_id)
                context = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
                message.append({'role': 'assistant', 'content': context})
            for i in range(n):
                if i == 0:
                    message.append({'role': 'user', 'content': '\n\n'.join([datapoint, prompt])})
                else:
                    message.append({'role': 'user', 'content': nxt})
                inputs = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors="pt").cuda()
                outputs = model.generate(inputs, max_new_tokens=512, do_sample=True, top_k=50, top_p=0.95, eos_token_id=tokenizer.eos_token_id)
                question = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
                message.append({'role': 'assistant', 'content': question})
                if category == 'extractive':
                    message.append({'role': 'user', 'content': span_prompt})
                else:
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
    with open('./deepseek_coder_33b.json', 'w') as f:
        json.dump(results, f)
    print('Done!', flush=True)
