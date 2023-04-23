import numpy as np
import os
import json

path = '../data/aesop/annotations/'
characters={}
for ner in os.listdir(path):
    ground_name = ner.split('_')

    ground_name = '_'.join(ground_name)
    with open(path + ner, 'r') as file:
        ground_truth = json.load(file)

        sentiment=ground_truth['sentiments']

        for character in sentiment:
            sentiment_list=[]
            for other_character in sentiment[character]:
                sentiment_list.append(int(sentiment[character][other_character]))

            unique,counts=np.unique(np.array(sentiment_list),return_counts=True)
            if len(unique)>1:
                mask = unique != 0
                i_max = np.argmax(unique[mask])
                sent = unique[mask][i_max]
            else:
                sent = unique[0]
            ground_truth['sentiments'][character]['overall'] = int(sent)


    with open(path + ner, 'w') as f:
        json.dump(ground_truth, f,indent=4)





