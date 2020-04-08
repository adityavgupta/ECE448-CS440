# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

import numpy as np
import time
"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set
"""

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters
    weights = np.zeros(len(train_set[0])+1)
    for epoch in range(max_iter):
        for features, label in zip(train_set, train_labels):
            prediction = 1 if (np.dot(features, weights[1:])+weights[0]) > 0 else 0
            weights[1:] += learning_rate*(label - prediction)*features
            weights[0] += learning_rate*(label - prediction)*1

    w = weights[1:]
    b = weights[0]
    return w, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    st = time.time()
    trained_weight, trained_bias = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    dev_label = []
    for img in dev_set:
        pred_res = 1 if (np.dot(img, trained_weight)+trained_bias) > 0 else 0
        dev_label.append(pred_res)
    print("time taken = %s seconds" % (time.time()-st))
    print(trained_weight.size)
    return dev_label

def sigmoid(x):
    # TODO: Write your code here
    # return output of sigmoid function given input x
    s = 1/(1+np.exp(-x))
    return s

def trainLR(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters

    # for my gradient calculations, I used this website: https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
    weights = np.zeros(len(train_set[0]))
    b = 0
    for epoch in range(max_iter):
        prediction = sigmoid(np.dot(train_set, weights)+b)
        gradient = np.dot(np.transpose(train_set),(prediction-train_labels))/train_labels.size
        weights -= learning_rate*gradient
        b -= learning_rate*np.sum(prediction-train_labels)/train_labels.size

    W = weights
    return W, b

def classifyLR(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train LR model and return predicted labels of development set
    tw, tb = trainLR(train_set, train_labels, learning_rate, max_iter)
    dev_label = []
    #i = 0
    for img in dev_set:
        pred_res = 1 if sigmoid(np.dot(img, tw)+tb) >= 0.5 else 0
        #i += 1
        dev_label.append(pred_res)
    return dev_label

#################
#  EXTRA CREDIT #
#################

from collections import Counter
import heapq

# mode calcualtion
def mode(_list):
    data = Counter(_list)
    return data[True] > data[False]

# calculate the euclidean distance between two vectors
def euclideanDist(v1, v2):
    dist = np.linalg.norm(v1-v2)
    return dist

# locate the most similar neigbors
def getNeighbors(train_set,train_labels, test_row, num_neigbors):
    distances_indices = []
    for index, train_row in enumerate(train_set):
        dist = euclideanDist(train_row, test_row)
        heapq.heappush(distances_indices, (dist, index))
    # k-nearest neighbors
    k_neighbors = distances_indices[:num_neigbors]
    k_labels = [train_labels[i] for distance, i in k_neighbors]
    return k_neighbors, k_labels

# predict the classification
def predict_classification(train_set, train_labels, test_row, num_neigbors):
    neighbors, labels = getNeighbors(train_set,train_labels, test_row, num_neigbors)
    prediction = mode(labels)
    return prediction

def classifyEC(train_set, train_labels, dev_set, k):
    # Write your code here if you would like to attempt the extra credit
    predictions = [predict_classification(train_set, train_labels, img, k) for img in dev_set]
    return predictions
