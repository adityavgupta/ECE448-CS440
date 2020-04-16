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
# from os import listdir
import numpy as np
import os

# Load the training and development dataset
def load_dataset(dir_dataset, part):
    # part = which part of MP6, either 1 or 2
    X_train = np.load(os.path.join(dir_dataset, 'X_train_part{}.npy'.format(part)))
    y_train = np.load(os.path.join(dir_dataset, 'y_train_part{}.npy'.format(part)))
    X_dev   = np.load(os.path.join(dir_dataset, 'X_dev_part{}.npy'.format(part)))
    y_dev   = np.load(os.path.join(dir_dataset, 'y_dev_part{}.npy'.format(part)))

    return X_train,y_train,X_dev,y_dev
