# mp3.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

import sys
import argparse
import configparser
import copy
import numpy as np

import reader
import naive_bayes as nb

"""
This file contains the main application that is run for Part 1 of this MP.
"""

def compute_accuracies(predicted_labels, dev_set, dev_labels):
    yhats = predicted_labels
    accuracy = np.mean(yhats == dev_labels)
    tp = np.sum([yhats[i] == dev_labels[i] and yhats[i] == 1 for i in range(len(yhats))])
    precision = tp / np.sum([yhats[i] == 1 for i in range(len(yhats))])
    recall = tp / (np.sum([yhats[i] != dev_labels[i] and yhats[i] == 0 for i in range(len(yhats))]) + tp)
    f1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, f1, precision, recall


def main(args):
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(args.training_dir,args.development_dir,args.stemming,args.lower_case)
    predicted_labels = nb.naiveBayes(train_set, train_labels, dev_set, args.laplace, args.pos_prior)

    accuracy, f1, precision, recall = compute_accuracies(predicted_labels, dev_set, dev_labels)
    print("Accuracy:",accuracy)
    print("F1-Score:",f1)
    print("Precision:",precision)
    print("Recall:",recall)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP3 Naive Bayes')

    parser.add_argument('--training', dest='training_dir', type=str, default = '../data/movies_review/train',
                        help='the directory of the training data')
    parser.add_argument('--development', dest='development_dir', type=str, default = '../data/movies_review/dev',
                        help='the directory of the development data')
    parser.add_argument('--stemming',dest="stemming", type=bool, default=False,
                        help='Use porter stemmer')
    parser.add_argument('--lower_case',dest="lower_case", type=bool, default=False,
                        help='Convert all word to lower case')
    parser.add_argument('--laplace',dest="laplace", type=float, default = 1.0,
                        help='Laplace smoothing parameter - default 1.0')
    parser.add_argument('--pos_prior',dest="pos_prior", type=float, default = 0.8,
                        help='Positive prior, i.e. Num_positive_comments / Num_comments')
    args = parser.parse_args()
    main(args)
