import json
import os

import spacy
import stanza
import textacy
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment', tokenize_no_ssplit=True)


def analyse_characters(story, characters, method='basic'):
    sent = SentimentIntensityAnalyzer()

    entity_sentiment = {}
    sentences = sent_tokenize(story)
    if method == 'basic':
        for character in characters:
            entity_sentiment[character] = []
            for sentence in sentences:
                if character.lower() in sentence.lower():
                    entity_sentiment[character].append(sent.polarity_scores(sentence)['compound'])
    elif method=='stanza':
        for character in characters:
            entity_sentiment[character] = []
            for sentence in sentences:
                if character.lower() in sentence.lower():
                    doc = nlp(sentence)
                    sentence_sentiment=doc.sentences[0].sentiment
                    entity_sentiment[character].append(sentence_sentiment)
    # elif method == 'SVO':  # only for coreferenced
    #     nlp = spacy.load('en_core_web_sm')
    #     for character in characters:
    #         entity_sentiment[character] = []
    #         for sentence in sentences:
    #             text = nlp(sentence)
    #             text_ext = list(textacy.extract.subject_verb_object_triples(text))
    #             for triplet in text_ext:
    #                 if character.lower() in [str(i).lower() for i in triplet.subject]:
    #                     entity_sentiment[character].append(sent.polarity_scores(sentence)['compound'])

    return entity_sentiment


def analyse_relationships(story, characters, method='vader'):
    sent = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(story)

    entity_sentiment = {}
    if method == 'vader':
        for character1 in characters:
            if character1 not in entity_sentiment:
                entity_sentiment[character1] = {}

            for character2 in characters:
                if character1 == character2:
                    continue

                if character2 not in entity_sentiment:
                    entity_sentiment[character2] = {}

                for sentence in sentences:
                    if (character1.lower() in sentence.lower()) and (character2.lower() in sentence.lower()):
                        if character1 not in entity_sentiment[character2]:
                            entity_sentiment[character2][character1] = []
                        if character2 not in entity_sentiment[character1]:
                            entity_sentiment[character1][character2] = []

                        score = sent.polarity_scores(sentence)['compound']
                        entity_sentiment[character1][character2].append(score)
                        entity_sentiment[character2][character1].append(score)
    if method == 'stanza':
        for character1 in characters:
            if character1 not in entity_sentiment:
                entity_sentiment[character1] = {}

            for character2 in characters:
                if character1 == character2:
                    continue

                if character2 not in entity_sentiment:
                    entity_sentiment[character2] = {}

                for sentence in sentences:
                    if (character1.lower() in sentence.lower()) and (character2.lower() in sentence.lower()):
                        if character1 not in entity_sentiment[character2]:
                            entity_sentiment[character2][character1] = []
                        if character2 not in entity_sentiment[character1]:
                            entity_sentiment[character1][character2] = []

                        doc = nlp(sentence)
                        score = doc.sentences[0].sentiment
                        entity_sentiment[character1][character2].append(score)
                        entity_sentiment[character2][character1].append(score)

    return entity_sentiment


dir_path = '../data/aesop/original/'
method = 'stanza'
single_analysis=False
for story_name in os.listdir(dir_path):
    with open(dir_path + story_name, 'r') as file:
        story = file.read().replace('\n', ' ')

    with open('../data/aesop/annotations/' + story_name.replace('txt', 'json'), 'r') as file:
        ground_truth = json.load(file)
    if single_analysis:
        entity_sentiment = analyse_characters(story, ground_truth['characters'], method=method)
        with open('../results/sentiment/single/' + method + '_' + story_name.replace('txt', 'json'), "w+") as fp:
            json.dump(entity_sentiment, fp,indent=4)
    else:
        entity_sentiment = analyse_relationships(story, ground_truth['characters'], method=method)
        with open('../results/sentiment/relationships/' + method + '_' + story_name.replace('txt', 'json'), "w+") as fp:
            json.dump(entity_sentiment, fp,indent=4)