import json
from nltk.tokenize import sent_tokenize
import spacy
import textacy

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

def analyse_characters(story, characters,method='basic'):
    sent = SentimentIntensityAnalyzer()

    entity_sentiment={}
    sentences=sent_tokenize(story)
    if method=='basic':
        for character in characters:
            entity_sentiment[character]=[]
            for sentence in sentences:
                if character.lower() in sentence.lower():
                    entity_sentiment[character].append(sent.polarity_scores(sentence)['compound'])
    elif method=='SVO': # only for coreferenced
        nlp = spacy.load('en_core_web_sm')
        for character in characters:
            entity_sentiment[character]=[]
            for sentence in sentences:
                text = nlp(sentence)
                text_ext = list(textacy.extract.subject_verb_object_triples(text))
                for triplet in text_ext:
                    if character.lower() in [str(i).lower() for i in triplet.subject]:
                        entity_sentiment[character].append(sent.polarity_scores(sentence)['compound'])

    return entity_sentiment

def analyse_relationships(story,characters,method='vader'):
    sent = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(story)

    entity_sentiment = {}
    if method=='vader':
        for character1 in characters:
            if character1 not in entity_sentiment:
                entity_sentiment[character1]={}

            for character2 in characters:
                if character1==character2:
                    continue

                if character2 not in entity_sentiment:
                    entity_sentiment[character2] = {}

                for sentence in sentences:
                    if (character1.lower()  in sentence.lower() ) and (character2.lower()  in sentence.lower()):
                        if character1 not in entity_sentiment[character2]:
                            entity_sentiment[character2][character1]=[]
                        if character2 not in entity_sentiment[character1]:
                            entity_sentiment[character1][character2]=[]

                        score=sent.polarity_scores(sentence)['compound']
                        entity_sentiment[character1][character2].append(score)
                        entity_sentiment[character2][character1].append(score)

    return entity_sentiment



dir_path='../data/aesop/test_story/'
for story_name in os.listdir(dir_path):
    with open(dir_path+story_name, 'r') as file:
        story = file.read().replace('\n', ' ')

    with open('../data/aesop/annotations/'+story_name.replace('txt','json'),'r') as file:
        ground_truth=json.load(file)
    entity_sentiment=analyse_characters(story,ground_truth['characters'],method='SVO')
    print()