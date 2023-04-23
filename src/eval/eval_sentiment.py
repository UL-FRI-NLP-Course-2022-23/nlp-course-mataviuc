import json
import os


def basic_eval(predicted, ground_truth,method):
    if method=='vader':
        for character1 in predicted:
            for character2 in predicted[character1]:
                predicted[character1][character2] = sum(predicted[character1][character2]) / len(
                    predicted[character1][character2])

                if predicted[character1][character2] > 0:
                    predicted[character1][character2] = 1
                elif predicted[character1][character2] < -0.08:
                    predicted[character1][character2] = -1
                else:
                    predicted[character1][character2] = 0
    elif method=='stanza':
        for character1 in predicted:
            for character2 in predicted[character1]:
                predicted[character1][character2] = sum(predicted[character1][character2]) / len(
                    predicted[character1][character2])

                if predicted[character1][character2] > 0:
                    predicted[character1][character2] = 1
                elif predicted[character1][character2] ==0:
                    predicted[character1][character2] = -1
    tp = 0
    fp = 0
    fn = 0
    for character1 in ground_truth:
        for character2 in ground_truth:
            if character1 == character2:
                continue

            if (character1 not in predicted) or (character2 not in predicted[character1]):
                fn+=1
            elif predicted[character1][character2] == ground_truth[character1][character2]:
                tp += 1
            elif ground_truth[character1][character2] == 0:
                fp += 1
            else:
                fn += 1

    if tp == 0:
        return 0, 0, 0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


path = '../../results/sentiment/relationships'
precisions = {}
recalls = {}
f1s = {}
for sent in os.listdir(path):

    with open('../../results/sentiment/relationships/' + sent, 'r') as file:
        predicted = json.load(file)

    ground_name = sent.split('_')
    method = ground_name.pop(0)
    if method not in precisions:
        precisions[method] = []
        recalls[method] = []
        f1s[method] = []

    ground_name = '_'.join(ground_name)
    with open('../../data/aesop/annotations/' + ground_name.replace('npy', 'json'), 'r') as file:
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
with open('../../results/sentiment/relationship_sentiment_results.json', "w") as fp:
    json.dump(final, fp,indent=4)
