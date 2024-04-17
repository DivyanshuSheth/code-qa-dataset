import os
import json
from evaluate import load
import argparse

# predictions = ["hello world", "general kenobi"]
# references = ["hello world", "general kenobi"]
# results = bertscore.compute(predictions=predictions, references=references, model_type="microsoft/deberta-xlarge-mnli")
# print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True, choices=["results", "question_only"])
    args = parser.parse_args()

    bertscore = load("bertscore")
    if args.output_dir == "results":
        with open(os.path.join("/home/dasheth/qa/code-qa-dataset/results", f"{args.model_name}.json"), "r") as f:
            results_here = json.load(f)
    elif args.output_dir == "question_only":
        with open(os.path.join("/home/dasheth/qa/code-qa-dataset/results/question_only", f"{args.model_name}.json"), "r") as f:
            results_here = json.load(f)
    if args.output_dir == "results":
        output_path = os.path.join("/home/dasheth/qa/code-qa-dataset/results/scores", f"{args.model_name}_scores.json")
        output_path_complete_scores = os.path.join("/home/dasheth/qa/code-qa-dataset/results/scores_complete", f"{args.model_name}_scores.json")
    elif args.output_dir == "question_only":
        output_path = os.path.join("/home/dasheth/qa/code-qa-dataset/results/scores_question_only", f"{args.model_name}_scores.json")
        output_path_complete_scores = os.path.join("/home/dasheth/qa/code-qa-dataset/results/scores_complete_question_only", f"{args.model_name}_scores.json")
    all_predictions = {'overall': []}
    all_references = {'overall': []}
    error_count = 0
    for datapoint in results_here:
        reference_here = datapoint['answer'].strip().replace('\"', "")
        primary_obfuscation_here = datapoint['primary_obfuscation']
        secondary_obfuscation_here = datapoint['secondary_obfuscation']
        if str(primary_obfuscation_here) not in all_predictions.keys():
            all_predictions[primary_obfuscation_here] = {}
            all_references[primary_obfuscation_here] = {}
        if str(secondary_obfuscation_here) not in all_predictions[primary_obfuscation_here].keys():
            all_predictions[primary_obfuscation_here][secondary_obfuscation_here] = []
            all_references[primary_obfuscation_here][secondary_obfuscation_here] = []
        if len(datapoint['generated_answer']) != 0:
            all_predictions[primary_obfuscation_here][str(secondary_obfuscation_here)].append(datapoint["generated_answer"][0].lstrip().rstrip())
            all_references[primary_obfuscation_here][str(secondary_obfuscation_here)].append(reference_here)
            all_predictions['overall'].append(datapoint["generated_answer"][0].lstrip().rstrip())
            all_references['overall'].append(reference_here)
        else:
            error_count += 1
    print("Model: ", args.model_name)
    print("Total count: ", len(results_here))
    print("Error count: ", error_count)

    results_dict = {'overall': None}
    results_dict_complete = {'overall': None}
    for primary, secondaries in all_predictions.items():
        if primary == 'overall':
            bertscore_overall = bertscore.compute(predictions=all_predictions['overall'], 
                                                  references=all_references['overall'], 
                                                  model_type="microsoft/deberta-xlarge-mnli",
                                                  num_layers=40,
                                                  rescale_with_baseline=True,
                                                  lang="en",
                                                  batch_size=256)
            results_dict['overall'] = sum(bertscore_overall['f1']) / len(bertscore_overall['f1'])
            results_dict_complete['overall'] = bertscore_overall
        else:
            for secondary, secondary_list in secondaries.items():
                bertscore_here = bertscore.compute(predictions=secondary_list, 
                                                   references=all_references[primary][secondary], 
                                                   model_type="microsoft/deberta-xlarge-mnli",
                                                   num_layers=40,
                                                   rescale_with_baseline=True,
                                                   lang="en",
                                                   batch_size=256)
                if primary not in results_dict.keys():
                    results_dict[primary] = {}
                if secondary not in results_dict[primary].keys():
                    results_dict[primary][secondary] = None
                results_dict[primary][secondary] = sum(bertscore_here['f1']) / len(bertscore_here['f1'])
                results_dict_complete[primary][secondary] = bertscore_here
    # results = bertscore.compute(predictions=all_predictions, references=all_references, model_type="microsoft/deberta-xlarge-mnli")
    # results_f1 = sum(results['f1']) / len(results['f1'])
    with open(output_path, "w") as f1:
        json.dump(results_dict, f1, indent=4)
    with open(output_path_complete_scores, "w") as f1:
        json.dump(results_dict_complete, f1, indent=4)