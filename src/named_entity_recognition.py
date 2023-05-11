import os
import pandas as pd
import numpy as np
import stanza
import torch
from allennlp.predictors import Predictor

from transformers import BertTokenizer
from coreference import replace_coreferences

df_data = pd.read_csv("../data/bert.csv", encoding="latin1").fillna(method="ffill")
tag_list = df_data.Tag.unique()
tag_list = np.append(tag_list, "PAD")
label2code = {label: i for i, label in enumerate(tag_list)}
code2label = {v: k for k, v in label2code.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('ner_bert_pt_finetuned12.pt').to(device)
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)



def named_entity_recognition(story, story_name, method='stanza', coreference=True):
    if coreference:
        story = replace_coreferences(story, story_name)

    named_entities = []
    if method == 'bert':
        tokenized_sentence = tokenizer.encode(story)
        input_ids = torch.tensor([tokenized_sentence]).to(device)
        with torch.no_grad():
            output = model(input_ids)
        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
        tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
        new_tokens, new_labels = [], []
        for token, label_idx in zip(tokens, label_indices[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(code2label[label_idx])
                new_tokens.append(token)
        named_entities = []
        for token, label in zip(new_tokens, new_labels):

            i = 0
            while i < len(new_tokens):
                if new_labels[i] in ['U-PER' ]:
                    named_entities.append([i])
                elif new_labels[i] in ['B-PER']:
                    str = new_tokens[i]
                    i += 1
                    while new_labels[i] in ['B-PER'] and i <len(new_tokens):
                        str += ' '
                        str += new_tokens[i]
                        i += 1
                    named_entities.append(str)
                i += 1

    elif method == 'allennlp':
        predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz")
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
    elif method=='stanza':
        ner_stanza = stanza.Pipeline('en', processors='tokenize,ner')
        doc = ner_stanza(story)
        named_entities = [entity.text for entity in doc.ents if entity.type == 'PERSON']

    return set(named_entities)


dir_path = '../data/aesop/original/'
method = 'bert'
for story_name in os.listdir(dir_path):
    with open(dir_path + story_name, 'r') as file:
        story = file.read().replace('\n', ' ')
        named_entities = named_entity_recognition(story, story_name, method=method, coreference=False)
        print(named_entities)
        save_path = '../results/ner/' + method + '_' + story_name.replace('txt', 'npy')
        np.save(save_path, np.array(named_entities))
        print()