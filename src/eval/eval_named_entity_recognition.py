import os
import numpy as np
import json

def basic_eval(predicted,ground_truth):
    tp=0
    fp=0
    fn=0
    pred = (' '.join(predicted)).lower()
    for g in ground_truth:
        if g in pred:
            tp+=1
        else:
            fn+=1

    if tp==0:
        return 0,0,0

    fp = len(predicted)-tp
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1=(2*precision*recall)/(precision+recall)

    return precision,recall,f1


path='../../results/ner/'
precisions=[]
recalls=[]
f1s=[]
for ner in os.listdir(path):
    predicted=np.load(path+ner,allow_pickle=True)

    ground_name=ner.split('_')
    ground_name.pop(0)
    ground_name='_'.join(ground_name)
    with open('../../data/aesop/annotations/'+ground_name.replace('npy','json'),'r') as file:
        ground_truth=json.load(file)

    precision,recall,f1=basic_eval(list(predicted.tolist()),ground_truth['characters'])
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    print()

