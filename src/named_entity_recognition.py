import os
from collections import Counter
from pathlib import Path

import stanza
import numpy as np
import spacy

from coreference import replace_coreferences
from allennlp.predictors import Predictor

#from utils import read_story

#ner_spacy = spacy.load('en_core_web_sm')

def named_entity_recognition(story,story_name,method='stanza',coreference=True):
    if coreference:
        story=replace_coreferences(story,story_name)

    named_entities=[]
    if method=='allennlp':
        predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz")
        results = predictor.predict(sentence=story)

        named_entities = []
        i = 0
        while i < len(results['words']):
            if results['tags'][i] in ['U-PER', 'U-MISC']:
                named_entities.append(results['words'][i])
            elif results['tags'][i] in ['B-PER', 'B-MISC']:
                str = results['words'][i]
                i += 1
                while results['tags'][i] in ['I-PER', 'I-MISC'] and i < len(results['words']):
                    str += ' '
                    str += results['words'][i]
                    i += 1
                named_entities.append(str)
            i += 1
    # elif method=='stanza':
    #     ner_stanza = stanza.Pipeline('en', processors='tokenize,ner')
    #     doc = ner_stanza(story)
    #     named_entities = [entity.text for entity in doc.ents if entity.type == 'PERSON'] #TODO SET

    return set(named_entities)

dir_path='../data/aesop/test_story/'
method='allennlp'
for story_name in os.listdir(dir_path):
    with open(dir_path+story_name, 'r') as file:
        story = file.read().replace('\n', ' ')
        named_entities=named_entity_recognition(story,story_name,method=method,coreference=True)
        save_path='../results/ner/'+method+'_'+story_name.replace('txt','npy')
        np.save(save_path,np.array(named_entities))

