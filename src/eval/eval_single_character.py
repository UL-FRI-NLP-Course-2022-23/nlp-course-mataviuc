import json
import os

def basic_eval(predicted, ground_truth,method):
    if method=='basic':
        for character1 in predicted:
            if len(predicted[character1])==0:
                predicted[character1]=0
            else:
                predicted[character1] = sum(predicted[character1]) / len(
                    predicted[character1])

                threshold=-0.19
                if predicted[character1] > threshold:
                    predicted[character1] = 1
                elif predicted[character1] < threshold:
                    predicted[character1] = -1
                else:
                    predicted[character1] = 0
    elif method=='stanza':
        for character1 in predicted:
            if len(predicted[character1]) == 0:
                predicted[character1] = 0
            else:
                predicted[character1] = sum(predicted[character1]) / len(
                    predicted[character1])

                if predicted[character1] > 0:
                    predicted[character1] = 1
                elif predicted[character1] ==0:
                    predicted[character1]=-1
    tp = 0
    fp = 0
    fn = 0
    for character1 in ground_truth:
        if predicted[character1] == ground_truth[character1]['overall']:
            tp += 1
        elif ground_truth[character1]['overall'] == 0:
            fp += 1
        else:
            fn += 1

    if tp == 0:
        return 0, 0, 0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


path = '../../results/sentiment/single/'
precisions = {}
recalls = {}
f1s = {}
for sent in os.listdir(path):

    with open('../../results/sentiment/single/' + sent, 'r') as file:
        predicted = json.load(file)

    ground_name = sent.split('_')
    method = ground_name.pop(0)
    if method not in precisions:
        precisions[method] = []
        recalls[method] = []
        f1s[method] = []

    ground_name = '_'.join(ground_name)
    with open('../../data/aesop/annotations/' + ground_name, 'r') as file:
        ground_truth = json.load(file)

    precision, recall, f1 = basic_eval(predicted, ground_truth['sentiments'],method)
    precisions[method].append(precision)
    recalls[method].append(recall)
    f1s[method].append(f1)

methods_dict = {}
for method in precisions:
    methods_dict[method + '_precision_score'] = sum(precisions[method]) / len(precisions[method])
    methods_dict[method + '_recall_score'] = sum(recalls[method]) / len(recalls[method])
    methods_dict[method + '_f1_score'] = sum(f1s[method]) / len(f1s[method])

final = {'precision': precisions, 'recall': recalls, 'f1': f1s, 'summed': methods_dict}
with open('../../results/sentiment/single_sentiment_results.json', "w") as fp:
    json.dump(final, fp,indent=4)