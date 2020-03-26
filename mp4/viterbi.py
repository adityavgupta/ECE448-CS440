"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import numpy as np
from collections import Counter

def baseline(train, test):
    '''
    TODO: implement the baseline algorithm. This function has time out limitation of 1 minute.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
            test data (list of sentences, no tags on the words)
            E.g  [[word1,word2,...][word1,word2,...]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''
    predicts = []
    word_tag = {}
    tag_ct = Counter()
    for sentence in train:
        for w_tag in sentence:
            w, t = w_tag
            if w not in word_tag:
                word_tag[w] = Counter()
 
            word_tag[w][t] += 1
            tag_ct[t] += 1

    t_max = max(tag_ct.keys(), key=(lambda key: tag_ct[key]))

    for sentence in test:
        tag_pred = []
        for word in sentence:
            if word in word_tag:
                max_t = max((word_tag[word]).keys(), key=lambda key:word_tag[word][key])
                tag_pred.append((word, max_t))
            else:
                tag_pred.append((word, t_max))
        predicts.append(tag_pred)

    return predicts


def viterbi_p1(train, test):
    '''
    TODO: implement the simple Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''

    predicts = []
    word_tag = {}       # a dict of the form {word:{tag:count of this tag}}
    tag_ct = Counter()  # a dict of the form {tag:its count}
    tag_idx = {}        # a dict of the form {tag: its index in the tag_ct dict}
    idx = 0

    # build the wird_tag and tag_ct dicts
    for sentence in train:
        for w_tag in sentence:
            w, t = w_tag
            if w not in word_tag:
                word_tag[w] = Counter()

            word_tag[w][t] += 1
            tag_ct += 1

    # make the tag_idx map 
    for t in tag_ct.keys():
        tag_idx[tag] = idx
        idx += 1

    k = 0.00001
    # emission prob
    for k in word_tag.keys():
        for t in word_tag[key].keys():
            word_tag[k][t] = (k+word_tag[k][t])/(tag_ct[t] + k*len(tag_ct))

    init_tag_probs = np.zeros(idx)
    trans_probs_table = np.zeros(shape=(idx, idx))

    for sentence in train:
        first = True
        for i in range(len(sentence)-1):
            w, t = sentence[i]
            curr_t_i = tag_idx[t]

            if first:
                init_tag_probs[curr_t_i] += 1
                first = False

            next_t = sentence[i+1][1]
            trans_probs_table[curr_t_i][tag_idx[next_t]] += 1

    # initial prob
    for itp in init_tag_probs:
        itp = (itp+k)/(len(train)+k*len(tag_ct))

    # laplace smoothing
    for t_ct in tag_ct:
        t, ct = t_ct
        prev_idx = tag_idx[t]
        for i in range(len(trans_probs_table)):
            trans_probs_table[prev_idx][i] = (trans_probs_table[prev_idx][i] + k)/(ct + k*len(tag_ct)) 


    return predicts

def viterbi_p2(train, test):
    '''
    TODO: implement the optimized Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''


    predicts = []
    raise Exception("You must implement me")
    return predicts
