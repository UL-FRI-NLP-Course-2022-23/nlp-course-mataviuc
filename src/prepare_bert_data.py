import json
import os
from nltk.tokenize import sent_tokenize, word_tokenize

import pandas as pd
df = pd.DataFrame(columns=['Sentence #', 'Word', 'POS', 'Tag'])
dir_path_story = '../data/aesop/original/'
dir_path_annots = '../data/aesop/annotations/'
method = 'bert'
for story_name in os.listdir(dir_path_story):
    with open(dir_path_story + story_name, 'r') as file:
        story = file.read().replace('\n', ' ')
    with open(dir_path_annots+ story_name.replace('txt','json')) as file:
        ground_truth = json.load(file)
    import pandas as pd
    import nltk


    sentences = sent_tokenize(story)
    for ii, sentence in enumerate(sentences):
        words = word_tokenize(sentence)
        for jj, word in enumerate(words):
            tag = 'O'
            for char in ground_truth['characters']:
                for sub_name in char.split(" "):

                    if word.lower() == sub_name and nltk.pos_tag([sub_name])[0][1] not in ['JJ', 'JJS', 'JJR']:
                        tag = 'PER'
            if jj == 0:
                df = pd.concat([df, pd.DataFrame(
                    [{'Sentence #': 'Sentence: ' + str(ii + 1), 'Word': word, 'POS': '0', 'Tag': tag}])],
                               ignore_index=True)
            else:
                df = pd.concat([df, pd.DataFrame([{'Sentence #': '', 'Word': word, 'POS': '0', 'Tag': tag}])],
                               ignore_index=True)
df.to_csv('../data/bert.csv', index=False)