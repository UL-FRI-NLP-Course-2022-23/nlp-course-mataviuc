import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize


def sent_count(story):
    return len(sent_tokenize(story))


def word_count(story):
    return len(word_tokenize(story))


def avg_words_in_sentence(stories):
    words_in_senteces = []
    for story in stories:
        sentences = sent_tokenize(story)
        for sentence in sentences:
            words_in_senteces.append(len(word_tokenize(sentence)))
    return sum(words_in_senteces) / len(words_in_senteces)


def vocabulary_size(stories):
    global_vocabulary = np.array([])
    avg_vocabulary_size = 0
    for story in stories:
        words = word_tokenize(story)
        unique, _ = np.unique(words, return_counts=True)
        avg_vocabulary_size += len(unique)
        global_vocabulary = np.append(global_vocabulary, unique)
    vocabulary, _ = np.unique(global_vocabulary, return_counts=True)
    return len(vocabulary), avg_vocabulary_size / len(stories)
