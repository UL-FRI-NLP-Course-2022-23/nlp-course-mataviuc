import os
from transformers import BertTokenizer, BertConfig
import pandas as pd
import torch
import numpy as np

model = torch.load('ner_bert_pt_finetuned.pt', map_location=torch.device('cpu'))

df_data = pd.read_csv("../data/bert.csv", encoding="latin1").fillna(method="ffill")
tag_list = df_data.Tag.unique()
tag_list = np.append(tag_list, "PAD")
label2code = {label: i for i, label in enumerate(tag_list)}
code2label = {v: k for k, v in label2code.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('ner_bert_pt_finetuned.pt').to(device)
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
def named_entity_recognition(story, story_name, method='stanza', coreference=True):


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
                if new_labels[i] in ['PER' ]:
                    named_entities.append([i])
                elif new_labels[i] in ['PER']:
                    str = new_tokens[i]
                    i += 1
                    while new_labels[i] in ['PER'] and i <len(new_tokens):
                        str += ' '
                        str += new_tokens[i]
                        i += 1
                    named_entities.append(str)
                i += 1

    return set(named_entities)


dir_path = '../data/new_data/'
method = 'bert'
for story_name in os.listdir(dir_path):
    with open(dir_path + story_name, 'r') as file:
        story = file.read().replace('\n', ' ')
        named_entities = named_entity_recognition(story, story_name, method=method, coreference=False)
        print(named_entities)
        # save_path = '../results/ner/' + method + '_' + story_name.replace('txt', 'npy')
        # np.save(save_path, np.array(named_entities))
        print()