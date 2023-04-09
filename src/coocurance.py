import math
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import spacy
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



def draw_graph(story,characters,story_name,plot=True):
    sentences = sent_tokenize(story)
    entity_matrix = np.zeros((len(characters), len(characters)))

    for i,character1 in enumerate(characters):
       for j,character2 in enumerate(characters):
           if character1 == character2:
               continue
           for sentence in sentences:
                if (character1.lower()  in sentence.lower() ) and (character2.lower()  in sentence.lower()):
                    entity_matrix[i][j]+=1
                    entity_matrix[j][i] += 1

    df = pd.DataFrame(data=entity_matrix)
    df=df.set_index(np.array(characters))
    df.columns = np.array(characters)

    if plot:
        row_sums = df.sum(axis=1) > 0
        selected = df[row_sums].loc[:, row_sums]
        plt.clf()
        l = {i: n for i, n in enumerate(selected.columns)}
        G = nx.from_numpy_matrix(np.matrix(selected))
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw(G,labels=l, with_labels=True, width=weights, font_size=18, edge_color='lightblue')

        plt.title(story_name.replace('.txt',''))
        plt.savefig("../results/figs/"+story_name.replace('txt','png'))
        plt.show()

    return df

dir_path='../data/aesop/coreferenced/'
for story_name in os.listdir(dir_path):
    with open(dir_path+story_name, 'r') as file:
        story = file.read().replace('\n', ' ')

    with open('../data/aesop/annotations/'+story_name.replace('txt','json'),'r') as file:
        ground_truth=json.load(file)
    entity_matrix=draw_graph(story,ground_truth['characters'],story_name)
    print()