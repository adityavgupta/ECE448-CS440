"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import numpy as np
from collections import Counter
from math import log

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

def make_trellis(sentence, tags, init_prob, ip_uk, trans_prob, trans_prob_uk, emission_prob, emission_prob_uk):
    tags = list(tags)
    ts = [{t:(init_prob.get(t, ip_uk) + emission_prob.get(sentence[0], emission_prob_uk[t])) for t in tags}]

    back_ptr = {}
    i = 0
    for word in sentence[1:]:
        temp = {}
        i += 1

        for tag_curr in tags:
            tag_prev = tags[0]
            temp[tag_curr] = ts[-1][tag_prev]+trans_prob[tag_prev].get(tag_prev, trans_prob_uk[tag_prev])+emission_prob[tag_curr].get(word, emission_prob_uk[tag_curr])
            back_ptr[(tag_curr, i)] = (tag_prev, i-1)

            for tag_prev in tags:
                tptc = ts[-1].get(tag_prev, ip_uk) + trans_prob[tag_prev].get(tag_curr, trans_prob_uk[tag_prev])+emission_prob[tag_curr].get(word, emission_prob_uk[tag_curr])
                if temp[tag_curr] < tptc:
                    temp[tag_curr] = tptc
                    back_ptr[(tag_curr, i)] = (tag_prev, i-1)
        ts.append(temp)
    return ts, i, back_ptr


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
    k = 0.00001

    init_tag_ct = Counter()

    prev_to_tag = {}

    prev_tag_ct = Counter()

    tag_ct = Counter()
    tag_word = {}

    tags = set()
    words = set()

    for sentence in train:
        for i in range(len(sentence)):
            w, t = sentence[i]
            # first word
            if i == 0:
                init_tag_ct[t] += 1
            else:
                prev_word, prev_tag = sentence[i-1]

                # prev->cur+1
                if prev_tag not in prev_to_tag:
                    prev_to_tag[prev_tag] = Counter()
                prev_to_tag[prev_tag][t] += 1

                # prev+1
                prev_tag_ct[prev_tag] += 1
            # (w, t) + 1
            if t not in tag_word:
                tag_word[t] = Counter()
            tag_word[t][w] += 1

            # t+1
            tag_ct[t] += 1

            words.add(w)
            tags.add(t)

    # initial prob
    init_prob = {}
    for (tag,count) in init_tag_ct.items():
        init_prob[tag] = (count+k)/(len(train) + k*len(tags))
    # unkown initial prob
    ip_uk = log(k/(len(train) + k*len(tags)))

    # transition prob
    trans_prob = {}
    for (pt, tag_count) in prev_to_tag.items():
        temp = {}
        for (tc, ct) in tag_count.items():
            temp[tc] = log((ct + k)/(prev_tag_ct[pt] + k*len(tags)))
        trans_prob[pt] = temp
    # unknown transition prob
    trans_prob_uk = {}
    for pt in list(tags):
        trans_prob_uk[pt] = log(k/(prev_tag_ct.get(pt, 0)+ k*len(tags)))

    # emission prob
    emission_prob = {}
    for (tag,tt) in tag_word.items():
        temp = {}
        for (word,count) in tt.items():
            temp[word] = log((k + count)/(tag_ct[t] + k*(1+len(words))))
            emission_prob[tag] = temp
    # unknown emission prob
    emission_prob_uk = {}
    for tag in list(tags):
        emission_prob_uk[tag] = log(k/(tag_ct.get(tag, 0) + k*(1+len(words))))

    # build the trellis

    #estimated_test = [[] for i in range(len(test))]
    #for s in test:
    #    mat = matrix(s, tags)
    #    back_ptr = make_b_ptr(s, tags)
    #    estimated_test[i] = viterbi_helper(s, mat, back_ptr, init_prob, ip_uk, trans_prob, trans_prob_uk, emission_prob, emission_prob_uk)
    #predicts = estimated_test

    for sentence in test:
        trellis, i, back_ptr = make_trellis(sentence, tags, init_prob, ip_uk, trans_prob, trans_prob_uk, emission_prob, emission_prob_uk)
        ts_max = max(trellis[-1].items(), key=(lambda x: x[1]))
        p = []
        c = (ts_max[0], i)
        while i >= 0:
            p.append(ts_max[0])
            c = back_ptr.get(c, (-1, -1))

        p = list(reversed(predicts))
        predicts.append(list(zip(sentence, p)))
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
