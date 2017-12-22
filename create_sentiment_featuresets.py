#! /usr/bin/env python
# -*- coding:utf-8
# Create a feature set for identifying positive or negative sentiments using neural networks
# Special thanks: Harisson@pythonprogramming.net

'''
    File name: create_sentiments_featuresets.py
    Author: chimney37
    Date created: 11/12/2017
    Python Version: 3.62
'''
import nltk  # flake8: NOQA
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer

'''
This is an example for an approach that does not assume preloaded data and
setup. This is the usual challange with machine learning - how do we prepare
the data and it what format? This file focuses on creating the featureset from
data that is not straightforward to be fed straight to a neural network.

We will use 2 files, 1 Pos (ident_nn_pos.txt) and Neg (ident_nn_neg.txt).
Challange 1: data is in language/word format, rather than numerical. 2: text is
not in same length of words or characters. Big deal, because we need all
featuresets to be exactly the same length going into training.

Solution: compile a list of all unique words in training set. Assuming ~3500
words, these will be lexicon. We create 1) a training vector of zerors that is
1x3500 in size; 2) A list of all unique words that is also 1x3500. For every
word that is in our sample sentence, check if it is in our unique word vector.
If so, index value of the word in unique word index is set to 1 in the training
vector. This is a simple bag-of-words model.

Example: if unique word list is [x,y,z,r,t]. Let's say we have a training
sentence that is "x y q p a b c". We initialize a training vector [0 0 0 0 0].
Iterating through all words in the sample sentence, if one word is in unique
word list, we make that index's value in the training vector to be 1. Since "x"
and "y" are in the unique word list, our training feature vector will be [1 1 0
0 0]. For the output we have either a positive or negative sentiment, so we
just use one-hot encoding and have the label vector be [POS, NEG], where
positive data is [1 0] and negative data is [0 1].

To aid us in pre-processing for problem of language processing, we will make
use of NTLK (Natural language toolkit). Our main interest is for the word
tokenizer, as well as as the Lemmatizer. Word tokenizer separates the word for
us. The lemmatizer takes similar words and converts them into the same single
word (a form of normalization). This is similar to stemming, but a lemma is an
actual word while a stem is not. To see if something is a word, we can check
WordNet. This will help us keep our lexicon smaller,
without losing too much value.
'''

lemmatizer = WordNetLemmatizer()

# for doing up to this amount of rows of data
hm_lines = 100000

# We will be used to sort most common lemmas, and pickle to save the process so
# we don't repeat every time


def create_lexicon(pos, neg):
    lexicon = []

    def create(src, lexicon):
        with open(src, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l)
                lexicon += list(all_words)

    create(pos, lexicon)
    create(neg, lexicon)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
        # print(w_counts[w])
        # if word occurs less than 1000 but more than 50, include it in
        # lexicon. Should be a function of % of entire dataset
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    print("example of lemmas in lexicon:")
    print(l2)
    print("len of lexicon:" + str(len(l2)))
    return l2


# create feature vector
def sample_handling(sample, lexicon, classification):
    featureset = []

    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1

            features = list(features)
            featureset.append([features, classification])
        print("Created featureset")
        print("E.g.:", featureset[0])
    return featureset


# creates feature sets(inputs) and labels(output) for training and testing
def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    featureset = []
    featureset += sample_handling(pos, lexicon, [1, 0])
    featureset += sample_handling(neg, lexicon, [0, 1])
    random.shuffle(featureset)
    featureset = np.array(featureset)

    testing_size = int(test_size*len(featureset))
    print("testing size:", testing_size)

    train_x = list(featureset[:, 0][:-testing_size])
    train_y = list(featureset[:, 1][:-testing_size])
    test_x = list(featureset[:, 0][-testing_size:])
    test_y = list(featureset[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('ident_nn_pos.txt', 'ident_nn_neg.txt')
    # pickle this data
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)
