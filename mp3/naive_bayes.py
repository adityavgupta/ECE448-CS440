# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 1 of MP3. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as numpy
import math
from collections import Counter


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter you provided with --laplace (1.0 by default)

    pos_prior - positive prior probability (between 0 and 1)
    """



    # TODO: Write your code here
    pos_words_count_map = getWordCountMap(train_set, train_labels, 0)
    neg_words_count_map = getWordCountMap(train_set, train_labels, 1)

    pos_words_probs_map, pos_unknown = makeProbs(pos_words_count_map, smoothing_parameter)
    neg_words_probs_map, neg_unknown = makeProbs(neg_words_count_map, smoothing_parameter)

    #log_pos_probs = np.log(pos_probs)
    #log_neg_probs = np.log(neg_probs)
    log_pos_unkown = np.log(pos_unknown)
    log_neg_unkown = np.log(neg_unknown)

    dev_labels[]

    for i in range(len(dev)):
        pos_p = 0
        neg_p = 0

        for word in dev_set[i]:
            if word in pos_words_count_map:
                pos_p += pos_words_probs_map["word"]
            else:
                pos_p += pos_unknown

            if word in neg_words_count_map:
                neg_p += neg_words_probs_map["word"]
            else:
                neg_p += neg_unknown

        if (pos_p > neg_p):
            dev_labels.append(1)
        else:
            dev_labels.append(0)

    # return predicted labels of development set (make sure it's a list, not a numpy array or similar)
    return dev_labels

def getWordCountMap(train_set, train_labels, isNeg):
    word_count = {}

    for i in range(len(train_labels)):
        if (train_labels[i] != isNeg):
            continue
        cur = train_set[i]

        for word in cur:
            if word in word_count:
                word_count["word"] += 1
            word_count["word"] = 1

    return word_count

def makeProbs (wordCount, smoothing_parameter):
    probmap = {}
    tot_words = 0
    tot_types = len(wordCount)

    for word in wordCount:
        tot_words += wordCount["word"]

    u_prob = smoothing_parameter/(tot_words + smoothing_parameter*(tot_types+1))

    for word in wordCount:
        prob = (wordCount["word"] + smoothing_parameter)/(tot_words + smoothing_parameter*(tot_types+1))
        probmap["word"] = prob

        return probmap, u_prob 
