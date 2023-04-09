import json
import os

import numpy as np


def basic_eval(predicted, ground_truth):
    tp = 0
    fp = 0
    fn = 0
    pred = (' '.join(predicted)).lower()
    for g in ground_truth:
        if g in pred:
            tp += 1
        else:
            fn += 1

    if tp == 0:
        return 0, 0, 0

    fp = len(predicted) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


path = '../../results/ner/'
precisions = {}
recalls = {}
f1s = {}
for ner in os.listdir(path):
    predicted = np.load(path + ner, allow_pickle=True)

    ground_name = ner.split('_')
    method = ground_name.pop(0)
    if method not in precisions:
        precisions[method] = []
        recalls[method] = []
        f1s[method] = []

    ground_name = '_'.join(ground_name)
    with open('../../data/aesop/annotations/' + ground_name.replace('npy', 'json'), 'r') as file:
        ground_truth = json.load(file)

    precision, recall, f1 = basic_eval(list(predicted.tolist()), ground_truth['characters'])
    precisions[method].append(precision)
    recalls[method].append(recall)
    f1s[method].append(f1)

methods_dict = {}
for method in precisions:
    methods_dict[method + '_precision_score'] = sum(precisions[method]) / len(precisions[method])
    methods_dict[method + '_recall_score'] = sum(recalls[method]) / len(recalls[method])
    methods_dict[method + '_f1_score'] = sum(f1s[method]) / len(f1s[method])

final = {'precision': precisions, 'recall': recalls, 'f1': f1s, 'summed': methods_dict}
with open('../../results/ner/ner_results.json', "w") as fp:
    json.dump(final, fp)
