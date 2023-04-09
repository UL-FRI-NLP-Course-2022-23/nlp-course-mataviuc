from allennlp.predictors.predictor import Predictor
import re
import numpy as np

MODEL_URL = 'https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz'




def replace_coreferences(story,story_name):
    predictor = Predictor.from_path(MODEL_URL)
    prediction = predictor.predict(document=story)

    # pattern = r'\b(he|she|they|him|her|them|his|hers|theirs|\bI\b|\bit\b)\b'
    # #the|\ban\b|\ba\b|\band\b
    # indexes = []
    # for j,e in enumerate(prediction['clusters']):
    #     names = []
    #     for i in e:
    #         name = ' '.join(prediction['document'][i[0]:i[1] + 1])
    #         if not re.search(pattern, name, re.IGNORECASE):
    #             names.append(name)
    #     counts = np.unique(np.array(names), return_counts=True)
    #     if counts[0].shape[0] >= 1:
    #         indexes.append(counts[0][np.argmax([counts[1]])])
    #     else:
    #         indexes.append("")
    #
    #     #np_prediction=np.array(prediction['document'])
    #     test_story=' '.join(prediction['document'])
    #     for i in e:
    #         test_story=test_story.replace(' '.join(prediction['document'][i[0]:i[1] + 1]),indexes[j])
    #         #prediction['document'][i[0]:i[1] + 1]=indexes[j]
    #         print()

    resolved_story=predictor.coref_resolved(story)



    with open('../data/aesop/coreferenced/'+story_name, 'w') as f:
        f.write(resolved_story)

    return resolved_story
