# mp6.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
import sys
import argparse
import configparser
import copy
import numpy as np

import reader
import neuralnet_p1 as p1
import neuralnet_p2 as p2
import neuralnet_p3 as p3
import torch

"""
This file contains the main application that is run for this MP.
"""

def compute_accuracies(predicted_labels,dev_set,dev_labels, N_class):
    yhats = predicted_labels
    if len(yhats) != len(dev_labels):
        print("Lengths of predicted labels don't match length of actual labels", len(yhats),len(dev_labels))
        return 0.,0.,0.,0.
    accuracy = np.mean(yhats == dev_labels)
    tp = []
    for n in range(N_class):
        tp.append(np.sum([yhats[i] == dev_labels[i] and yhats[i] == n for i in range(len(dev_labels))]))
    precision = []
    for n in range(N_class):
        precision.append(tp[n] / np.sum([yhats[i]==n for i in range(len(dev_labels))]))
    recall = []
    for n in range(N_class):
        recall.append(tp[n] / np.sum(dev_labels == n))
    f1 = []
    for n in range(N_class):
        f1.append(2 * (precision[n] * recall[n]) / (precision[n] + recall[n]))

    return accuracy,f1,precision,recall

def main(args):
    ### Part 1 ###
    train_set, train_labels, dev_set,dev_labels = reader.load_dataset(args.dataset_file, part=1)
    train_set    = torch.tensor(train_set,dtype=torch.float32)
    train_labels = torch.tensor(train_labels,dtype=torch.int64)
    dev_set      = torch.tensor(dev_set,dtype=torch.float32)
    losses,predicted_labels,net = p1.fit(train_set,train_labels, dev_set,args.max_iter)
    accuracy,f1,precision,recall = compute_accuracies(predicted_labels,dev_set,dev_labels, N_class=3)

    print(' ##### Part 1 results #####')
    print("Accuracy:",accuracy)
    print("F1-Score:",f1)
    print("Precision:",precision)
    print("Recall:",recall)
    print("num_parameters:", sum([ np.prod(w.shape) for w  in net.get_parameters()]))
    torch.save(net, "net_p1.model")

    ### Part 2 ###
    train_set, train_labels, dev_set,dev_labels = reader.load_dataset(args.dataset_file, part=2)
    train_set    = torch.tensor(train_set,dtype=torch.float32)
    train_labels = torch.tensor(train_labels,dtype=torch.int64)
    dev_set      = torch.tensor(dev_set,dtype=torch.float32)
    _,predicted_labels,net = p2.fit(train_set,train_labels, dev_set,2*args.max_iter)
    accuracy,f1,precision,recall = compute_accuracies(predicted_labels,dev_set,dev_labels, N_class=5)
    print(' ##### Part 2 results #####')
    print("Accuracy:",accuracy)
    print("F1-Score:",f1)
    print("Precision:",precision)
    print("Recall:",recall)
    print("num_parameters:", sum([ np.prod(w.shape) for w  in net.get_parameters()]))
    torch.save(net, "net_p2.model")

    ### Part 3 ###
    # input provided will be normalized to 0.0 - 1.0
    train_set, _, dev_set,_ = reader.load_dataset(args.dataset_file, part=2)   # use same data as part 2
    train_set    = torch.tensor(train_set,dtype=torch.float32)/255.0
    dev_set      = torch.tensor(dev_set,dtype=torch.float32)/255.0
    _,x_recon,net = p3.fit(train_set, dev_set,args.max_iter*2)
    diff = (x_recon - dev_set.numpy())**2
    MSE = diff.mean()
    print(' ##### Part 3 results #####')
    print("MSE:",MSE)
    print("num_parameters:", sum([ np.prod(w.shape) for w  in net.get_parameters()]))
    torch.save(net, "net_p3.model")

    ## Show some original images in 1st row & reconstructed images in 2nd row
    ## For debug only
    # import matplotlib.pyplot as plt
    # idx = np.random.choice(x_recon.shape[0], 10)
    # fig,axes=plt.subplots(2,10)
    # for i in range(10):
    #     axes[0,i].imshow(dev_set[idx[i]].reshape(28,28), cmap='gray')
    #     axes[1,i].imshow(x_recon[idx[i]].reshape(28,28), cmap='gray')
    # plt.show()
    # ###

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP6 Neural Net')

    parser.add_argument('--dataset', dest='dataset_file', type=str, default = '../data/',
                        help='the directory of the training data')
    parser.add_argument('--max_iter',dest="max_iter", type=int, default = 50,
                        help='Maximum iterations - default 10')

    args = parser.parse_args()
    main(args)
