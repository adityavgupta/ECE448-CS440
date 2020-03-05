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
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


import numpy as np
import math
from collections import Counter

def getWordCountMap(train_set, train_labels, isNeg):
    word_count = {}

    for i in range(len(train_labels)):
        if (train_labels[i] != isNeg):
            continue
        cur = train_set[i]

        for word in cur:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

    return word_count

def getBigramMap(train_set, train_labels, isNeg):
    bigram_map = {}
    for i in range(len(train_labels)):
        if train_labels[i] != isNeg:
            continue
        cur = train_set[i]
        for j in range(len(cur)-1):
            if j%2 == 0:
                bg = tuple((cur[j], cur[j+1]))
                if bg in bigram_map:
                    bigram_map[bg] += 1
                else:
                    bigram_map[bg] = 1
    return bigram_map

def makeProbs (bg_map, unigram_smoothing_parameter):
    probmap = {}
    tot_bg = 0
    tot_types = len(bg_map)

    for bg in bg_map:
        tot_bg += bg_map[bg]

    u_prob = unigram_smoothing_parameter/(tot_bg + unigram_smoothing_parameter*(tot_types))

    for bg in bg_map:
        prob = (bg_map[bg] + unigram_smoothing_parameter)/(tot_bg + unigram_smoothing_parameter*(tot_types))
        probmap[bg] = prob

    return probmap, u_prob 


def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda,unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """
 


    # TODO: Write your code here

    # Unigram part
    pos_words_count_map = getWordCountMap(train_set, train_labels, 1)
    neg_words_count_map = getWordCountMap(train_set, train_labels, 0)

    pos_words_probs_map, pos_unknown = makeProbs(pos_words_count_map, unigram_smoothing_parameter)
    neg_words_probs_map, neg_unknown = makeProbs(neg_words_count_map, unigram_smoothing_parameter)

    log_pos_unkown = np.log(pos_unknown)
    log_neg_unkown = np.log(neg_unknown)

    dev_labels = []
    dev_neg = []
    dev_pos = []

    for list in dev_set:
        pos_p = 0
        neg_p = 0

        for word in list:
            if word in pos_words_probs_map:
                pos_p += np.log(pos_words_probs_map[word])
            else:
                pos_p += log_pos_unkown

            if word in neg_words_probs_map:
                neg_p += np.log(neg_words_probs_map[word])
            else:
                neg_p += log_neg_unkown
        pos_p += np.log(pos_prior)
        neg_p += np.log(1-pos_prior)

        dev_pos.append(pos_p)
        dev_neg.append(neg_p)

    # Bigram part
    bi_pos_count_map = getBigramMap(train_set, train_labels, 1)
    bi_neg_count_map = getBigramMap(train_set, train_labels, 0)

    bi_pos_probs_map, bi_pos_uk = makeProbs(bi_pos_count_map, bigram_smoothing_parameter)
    bi_neg_probs_map, bi_neg_uk = makeProbs(bi_neg_count_map, bigram_smoothing_parameter)

    log_bi_pos_uk = np.log(bi_pos_uk)
    log_bi_neg_uk = np.log(bi_neg_uk)

    bi_dev_pos = []
    bi_dev_neg = []

    for list in dev_set:
        bi_pos_p = 0
        bi_neg_p = 0

        for j in range(len(list)-1):
            if j%2 == 0:
                bg = tuple((list[j], list[j+1]))

                if bg in bi_pos_probs_map:
                    bi_pos_p += np.log(bi_pos_probs_map[bg])
                else:
                    bi_pos_p += log_bi_pos_uk

                if bg in bi_neg_probs_map:
                    bi_neg_p += np.log(bi_neg_probs_map[bg])
                else:
                    bi_neg_p += log_bi_neg_uk
        bi_pos_p += np.log(pos_prior)
        bi_neg_p += np.log(1-pos_prior)

        bi_dev_pos.append(bi_pos_p)
        bi_dev_neg.append(bi_neg_p)

    for i in range(len(dev_set)):
        p_n = (1-bigram_lambda)*dev_neg[i] + (bigram_lambda)*bi_dev_neg[i]
        p_p = (1-bigram_lambda)*dev_pos[i] + (bigram_lambda)*bi_dev_pos[i]

        if p_p > p_n:
            dev_labels.append(1)
        else:
            dev_labels.append(0)

    return dev_labels