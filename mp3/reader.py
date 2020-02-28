# reader.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
"""
This file is responsible for providing functions for reading the files
"""
from os import listdir
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm


porter_stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
bad_words = {'aed','oed','eed'} # these words fail in nltk stemmer algorithm
def loadDir(name,stemming,lower_case):
    # Loads the files in the folder and returns a list of lists of words from
    # the text in each file
    X0 = []
    count = 0
    for f in tqdm(listdir(name)):
        fullname = name+f
        text = []
        with open(fullname, 'rb') as f:
            for line in f:
                if lower_case:
                    line = line.decode(errors='ignore').lower()
                    text += tokenizer.tokenize(line)
                else:
                    text += tokenizer.tokenize(line.decode(errors='ignore'))
        if stemming:
            for i in range(len(text)):
                if text[i] in bad_words:
                    continue
                text[i] = porter_stemmer.stem(text[i])
        X0.append(text)
        count = count + 1
    return X0

def load_dataset(train_dir, dev_dir, stemming, lower_case):
    X0 = loadDir(train_dir + '/pos/',stemming, lower_case)
    X1 = loadDir(train_dir + '/neg/',stemming, lower_case)
    X = X0 + X1
    Y = len(X0) * [1] + len(X1) * [0]
    Y = np.array(Y)

    X_test0 = loadDir(dev_dir + '/pos/',stemming, lower_case)
    X_test1 = loadDir(dev_dir + '/neg/',stemming, lower_case)
    X_test = X_test0 + X_test1
    Y_test = len(X_test0) * [1] + len(X_test1) * [0]
    Y_test = np.array(Y_test)

    return X,Y,X_test,Y_test
