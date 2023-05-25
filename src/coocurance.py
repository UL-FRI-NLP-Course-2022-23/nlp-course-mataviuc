import json
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize

colors_dict_edges={
    1: '#A1D573',
    -1: '#E46425',
    0: '#F5B521'
}

colors_dict_nodes={
    1: '#78A054',
    -1: '#B34E1D',
    0: '#BC8C1A'
}

def draw_graph(story, characters, story_name, sentiment,plot=True):
    sentences = sent_tokenize(story)
    entity_matrix = np.zeros((len(characters), len(characters)))

    for i, character1 in enumerate(characters):
        for j, character2 in enumerate(characters):
            if character1 == character2:
                continue
            for sentence in sentences:
                if (character1.lower() in sentence.lower()) and (character2.lower() in sentence.lower()):
                    entity_matrix[i][j] += 1
                    entity_matrix[j][i] += 1

    df = pd.DataFrame(data=entity_matrix)
    df = df.set_index(np.array(characters))
    df.columns = np.array(characters)

    if plot:
        row_sums = df.sum(axis=1) > 0
        selected = df[row_sums].loc[:, row_sums]

        plt.clf()
        ax = plt.subplot(111)
        ax.margins(0.3)

        l = {i: n for i, n in enumerate(selected.columns)}
        G = nx.from_numpy_matrix(np.matrix(selected))

        for u,v,d in G.edges(data=True):
            c1=characters[u]
            c2=characters[v]
            d['color'] = colors_dict_edges[sentiment[c1][c2]]

        # for character1 in G.nodes():
        #     for character2 in G.nodes():
        #         G[character1][character2]['color']=sentiment[character1][character2]

        colors_node = []
        for n in G:
            c=characters[n]
            colors_node.append(colors_dict_nodes[sentiment[c]['overall']])

        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        colors_edge = [G[u][v]['color'] for u, v in edges]

        nx.draw(G, ax=ax,labels=l, with_labels=True, node_color=colors_node, width=weights, font_size=18, edge_color=colors_edge)

        ax.set_title(story_name.replace('.txt', '').replace('_',' '))
        plt.savefig("../results/figs/" + story_name.replace('txt', 'png'))
        plt.show()

    return df


dir_path = '../data/aesop/coreferenced/'
for story_name in ['The_Cat_Maiden.txt','The_Hare_With_Many_Friends.txt']:#os.listdir(dir_path):
    with open(dir_path + story_name, 'r') as file:
        story = file.read().replace('\n', ' ')

    with open('../data/aesop/annotations/' + story_name.replace('txt', 'json'), 'r') as file:
        ground_truth = json.load(file)
    entity_matrix = draw_graph(story, ground_truth['characters'], story_name, ground_truth['sentiments'])
    print()
