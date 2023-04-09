from allennlp.predictors.predictor import Predictor
import re
import numpy as np

MODEL_URL = 'https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz'




def replace_coreferences(story,story_name):
    predictor = Predictor.from_path(MODEL_URL)
    prediction = predictor.predict(document=story)
    resolved_story = ' '.join(prediction['document'])
    pattern = r'\b(he|she|they|him|her|them|his|hers|theirs|\bI\b|\bit\b)\b'
    pattern2=r"\b(a|an|the|and|'s|´s)\b"
    indexes = []
    for j,e in enumerate(prediction['clusters']):
        names = []
        for i in e:
            name = ' '.join(prediction['document'][i[0]:i[1] + 1])

            if not re.search(pattern, name, re.IGNORECASE):
                name = re.sub(pattern2,'',name,re.IGNORECASE)
                name=name.strip()
                names.append(name)
        counts = np.unique(np.array(names), return_counts=True)
        if counts[0].shape[0] >= 1:
            person=counts[0][np.argmax([counts[1]])]

            indexes.append(person)
        else:
            indexes.append("")

        if indexes[j]=='':
            continue
        for i in e:
            resolved_story= re.sub(r"\b"+' '.join(prediction['document'][i[0]:i[1] + 1])+r"\b",indexes[j],resolved_story,re.IGNORECASE)

    regex = r"\s+([,.;!?:])"
    resolved_story = re.sub(regex, r"\1", resolved_story,re.IGNORECASE)
    with open('../data/aesop/coreferenced/'+story_name, 'w') as f:
        f.write(resolved_story)

    return resolved_story
