"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import numpy as np
from collections import Counter
from math import log
from math import inf

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

'''def make_trellis(sentence, tags, init_prob, ip_uk, trans_prob, trans_prob_uk, emission_prob, emission_prob_uk):
    tags = list(tags)
    ts = [{t:(init_prob.get(t, ip_uk) + emission_prob[t].get(sentence[0], emission_prob_uk[t])) for t in tags}]

    back_ptr = {}
    i = 0
    for word in sentence[1:]:
        temp = {}
        i += 1

        for tag_curr in tags:
            tag_prev = tags[0]
            temp[tag_curr] = ts[-1][tag_prev]+trans_prob[tag_prev].get(tag_prev, trans_prob_uk[tag_prev])+emission_prob[tag_curr].get(word, emission_prob_uk[tag_curr])
            back_ptr[(tag_curr, i)] = (tag_prev, i-1)

            for tag__prev in tags:
                tptc = ts[-1].get(tag__prev, ip_uk) + trans_prob[tag__prev].get(tag_curr, trans_prob_uk[tag__prev])+emission_prob[tag_curr].get(word, emission_prob_uk[tag_curr])
                if temp[tag_curr] < tptc:
                    temp[tag_curr] = tptc
                    back_ptr[(tag_curr, i)] = (tag_prev, i-1)
        ts.append(temp)
    ts_max = max(ts[-1].items(), key=(lambda x: x[1]))
    p = []
    c = (ts_max[0], i)t
    while c[1] >= 0:
        p.append(c[0])
        c = back_ptr.get(c, (-1, -1))
    p = list(reversed(p))
    return p'''

def make_matrix(sentence, tags):
    #matrix = [{tag:0 for tag in tags} for i in range(len(sentence))]
    matrix = []
    for i in range(len(sentence)):
        matrix.append({tag:0 for tag in tags})
    return matrix

def make_b_ptr(sentence, tags):
    back_ptr=[];
    for i in range(len(sentence)):
        back_ptr.append({tag:None for tag in list(tags)})
    return back_ptr

def helper(sentence,matrix, back_ptr, initial_prob, ip_uk, transition_prob, tp_uk, emission_prob, ep_uk, isDict):
    for key, value in matrix[0].items():
        pi = 0
        b = 0
        if key in initial_prob:
            pi = initial_prob[key]
        else:
            pi=ip_uk
        if (sentence[0], key) in emission_prob:
            b = emission_prob[(sentence[0], key)]
        else:
            if isDict == 0:
                b = ep_uk
            else:
                b = ep_uk[key]
        matrix[0][key] = pi+b
    for i in range(1, len(matrix)):
        for k in matrix[i].keys():
            max_prob = -inf
            max_key = ""
            b=0
            if (sentence[i], k) in emission_prob:
                b = emission_prob[(sentence[i], k)]
            else:
                if isDict == 0:
                    b= ep_uk
                else:
                    b = ep_uk[key]
            for k_prime in matrix[i-1].keys():
                a = 0
                if (k_prime, k) in transition_prob:
                    a = transition_prob[(k_prime, k)]
                else:
                    a = tp_uk

                if (a+b+matrix[i-1][k_prime]) > max_prob:
                    max_prob = a+b+matrix[i-1][k_prime]
                    max_key = k_prime
            matrix[i][k] = max_prob
            back_ptr[i][k] = max_key
    index = len(matrix)-1
    key_ = max(matrix[index], key=matrix[index].get)
    return_s = []
    while key_ != None and index>=0:
        return_s = [(sentence[index], key_)]+return_s
        key_ = back_ptr[index][key_]
        index -= 1
    return return_s


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
    k = 0.0001

    # make the sets of unique words and tags
    words = set()
    tags = set()
    for s in train:
        for pair in s:
            w,t = pair
            words.add(w)
            tags.add(t)

    # initial prob
    init_tag_ct = Counter()
    for sentence in train:
        init_tag_ct[sentence[0][1]] += 1
    initial_prob = dict(init_tag_ct)

    for (t, c) in initial_prob.items():
        initial_prob[t] = log((c+k)/(len(train) + k*len(tags)))
    ip_uk = log(k/(len(train) + k*len(tags)))

    # transition prob
    transition = []
    for s in train:
        for i in range(1, len(s)):
            curr_t = s[i][1]
            prev_t = s[i-1][1]
            transition.append((prev_t, curr_t))
    transition_prob = dict(Counter(transition))
    #tp_uk = dict()
    for tag_1 in list(tags):
        denominator = 0
        for tag in list(tags):
            if (tag_1, tag) in transition_prob:
                denominator += transition_prob[(tag_1, tag)]
        for tag in list(tags):
            if (tag_1, tag) in transition_prob:
                transition_prob[(tag_1, tag)] = log((transition_prob[(tag_1,tag)]+k)/(denominator+k*len(tags)))
            else:
                transition_prob[(tag_1, tag)] = log(k/(len(train)+k*len(tags))) #migh need to change the len used in denominator
    tp_uk = log(k/(len(train)+k*len(tags)))

    # emission prob
    emission_prob = Counter()
    tag_ct = Counter()
    for s in train:
        for wt_pair in s:
            emission_prob[wt_pair] += 1
            tag_ct[wt_pair[1]] += 1
    emission_prob = dict(emission_prob)

    for tag in list(tags):
        for word in list(words):
            if (word, tag) in emission_prob:
                emission_prob[(word, tag)]=log((emission_prob[(word,tag)]+k)/(tag_ct[tag]+k*(len(words)+1)))
            else:
                emission_prob[(word, tag)] = log(k/(len(train)+k*(len(words)+1)))
    ep_uk = log(k/(len(train)+k*(len(words)+1)))

    estimated_test = [[] for i in range(len(test))]
    i = 0
    for s in test:
        matrix = make_matrix(s, tags)
        back_ptr = make_b_ptr(s, tags)
        estimated_test[i] = helper(s,matrix, back_ptr, initial_prob, ip_uk, transition_prob, tp_uk, emission_prob, ep_uk, 0)
        i += 1

    predicts = []
    predicts = estimated_test
    return predicts

# part 2 hapax
def hapax(train, tags, alpha):
    
    wt_ctr = Counter()
    for s in train:
        for pair in s:
            wt_ctr[pair[0]] += 1
    wt_ctr = dict(wt_ctr)
    happax = [k for k,v in wt_ctr.items() if v==1]
    happax = set(happax)

    hap_tags = Counter()
    for s in train:
        for pair in s:
            if pair[0] in happax:
                hap_tags[pair[1]] += 1
    hap_tags = dict(hap_tags)
    sum_ = sum(hap_tags.values())
   
    #hap_tags = {k: (v+alpha)/(sum_ + alpha*len(tags)) for k,v in }
    for k,v in hap_tags.items():
        hap_tags[k] = (v+alpha)/(sum_ + alpha*len(tags))
    #hap_tags = {tag:alpha/(sum_+len(tags)*alpha) for tag in tags if tag not in hap_tags}
    for tag in tags:
        if tag not in hap_tags:
            hap_tags[tag] = alpha/(sum_+len(tags)*alpha)
    
    return hap_tags


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
    k = 10**(-5)

    # make the sets of unique words and tags
    tot_pairs = 0
    words = set()
    tags = set()
    for s in train:
        for pair in s:
            tot_pairs += 1
            w,t = pair
            words.add(w)
            tags.add(t)

    # initial prob
    init_tag_ct = Counter()
    for sentence in train:
        init_tag_ct[sentence[0][1]] += 1
    initial_prob = dict(init_tag_ct)

    for (t, c) in initial_prob.items():
        initial_prob[t] = log((c+k)/(len(train) + k*len(tags)))
    ip_uk = log(k/(len(train) + k*len(tags)))

    # transition prob
    transition = []
    for s in train:
        for i in range(1, len(s)):
            curr_t = s[i][1]
            prev_t = s[i-1][1]
            transition.append((prev_t, curr_t))
    transition_prob = dict(Counter(transition))
    #tp_uk = dict()
    for tag_1 in list(tags):
        denominator = 0
        for tag in list(tags):
            if (tag_1, tag) in transition_prob:
                denominator += transition_prob[(tag_1, tag)]
        for tag in list(tags):
            if (tag_1, tag) in transition_prob:
                transition_prob[(tag_1, tag)] = log((transition_prob[(tag_1,tag)]+k)/(denominator+k*(len(tags)+1)))
            else:
                transition_prob[(tag_1, tag)] = log(k/(tot_pairs+k*(len(tags)+1))) #might need to change the len used in denominator
    tp_uk = log(k/(tot_pairs+k*(len(tags)+1)))

    # hapax calculations
    hapax_dict = hapax(train, tags, k)

    # emission prob
    emission_prob = Counter()
    tag_ct = Counter()
    for s in train:
        for wt_pair in s:
            emission_prob[wt_pair] += 1
            tag_ct[wt_pair[1]] += 1
    emission_prob = dict(emission_prob)

    for tag in list(tags):
        denominator = 0
        for word in list(words):
            if (word, tag) in emission_prob:
                denominator += emission_prob[(word,tag)]
        for word in list(words):
            if (word, tag) in emission_prob:
                emission_prob[(word, tag)]=log((emission_prob[(word,tag)]+k*hapax_dict[tag])/(denominator+k*(len(words)+1)*hapax_dict[tag]))
            else:
                emission_prob[(word, tag)] = log((k*hapax_dict[tag])/(tag_ct[tag]+k*hapax_dict[tag]*(len(words)+1)))
    ep_uk = {tag:log((k*hapax_dict[tag])/(tag_ct[tag]+k*hapax_dict[tag]*(len(words)+1))) for tag in list(tags)}

    estimated_test = [[] for i in range(len(test))]
    i= 0
    for s in test:
        matrix = make_matrix(s, tags)
        back_ptr = make_b_ptr(s, tags)
        estimated_test[i] = helper(s, matrix, back_ptr, initial_prob, ip_uk, transition_prob, tp_uk, emission_prob, ep_uk, 1)
        i += 1
    predicts = []
    predicts = estimated_test
    return predicts
