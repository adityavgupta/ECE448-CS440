# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019

"""
This is the main entry point for MP6. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate, loss_fn, in_size,out_size):
        """
        Initialize the layers of your neural network
        @param lrate: The learning rate for the model.
        @param loss_fn: The loss function
        @param in_size: Dimension of input
        @param out_size: Dimension of output
        """
        super(NeuralNet, self).__init__()
        """
        1) DO NOT change the name of self.encoder & self.decoder
        2) Both of them need to be subclass of torch.nn.Module and callable, like
           output = self.encoder(input)
        3) Use 2d conv for extra credit part.
           self.encoder should be able to take tensor of shape [batch_size, 1, 28, 28] as input.
           self.decoder output tensor should have shape [batch_size, 1, 28, 28].
        """
        self.encoder = torch.nn.Sequential(torch.nn.Conv2d(1,6, kernel_size=5), nn.ReLU(True), torch.nn.Conv2d(6,1,kernel_size=5), torch.nn.ReLU(True))
        self.decoder = torch.nn.Sequential(torch.nn.ConvTranspose2d(1,6, kernel_size=5), torch.nn.ReLU(True), torch.nn.ConvTranspose2d(6,1,kernel_size=5), nn.ReLU(True))
        self.loss_fn = loss_fn
        self.lrate = lrate

    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """
        # return self.net.parameters()
        return self.parameters()

    def forward(self, x):
        """ A forward pass of your autoencoder
        @param x: an (N, in_size) torch tensor
        @return xhat: an (N, out_size) torch tensor of output from the network.
                      Note that self.decoder output needs to be reshaped from
                      [N, 1, 28, 28] to [N, out_size] beforn return.
        """
        x = x.view(x.shape[0],1,28,28)
        return self.decoder(self.encoder(x))

    def step(self, x):
        # x [100, 784]
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        optimizer = torch.optim.Adam(self.get_parameters(), lr=self.lrate, weight_decay = 0.01)
        #_input = y
        _target = self.forward(x)
        #print(_target.shape)
        loss = self.loss_fn
        _loss = loss(_target.view(-1,1,28,28), _target.view(-1,1,28,28)) #loss function is MSELoss()

        # Zero gradients, perform a backward pass, and update the weights
        optimizer.zero_grad()
        _loss.backward()
        optimizer.step()

        return _loss.item()

def fit(train_set,dev_set,n_iter,batch_size=100):
    """ Fit a neural net.  Use the full batch size.
    @param train_set: an (N, out_size) torch tensor
    @param dev_set: an (M, out_size) torch tensor
    @param n_iter: int, the number of batches to go through during training (not epoches)
                   when n_iter is small, only part of train_set will be used, which is OK,
                   meant to reduce runtime on autograder.
    @param batch_size: The size of each batch to train on.
    # return all of these:
    @return losses: list of total loss (as type float) at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return xhats: an (M, out_size) NumPy array of reconstructed data.
    @return net: A NeuralNet object
    # NOTE: This must work for arbitrary M and N
    """
    lrate = 0.1
    distance = nn.MSELoss() #loss_fn
    in_size = 784
    out_size = 5
    losses = list()
    ts_mean = train_set.mean()
    ts_std = train_set.std()

    train_set1 = (train_set-ts_mean)/ts_std
    net = NeuralNet(lrate, distance, in_size, out_size)

    for i in range(n_iter):
        # if i goes beyond N/n_iter in this case 7500/100 = 75, wrap around the batches
        if(i >=75):
            batch = train_set1[(i-75)*batch_size:(i-75+1)*batch_size]
            #label_batch = train_labels[(i-77)*batch_size:(i-77+1)*batch_size]
        else:
            batch = train_set1[i*batch_size:(i+1)*batch_size]
            #label_batch = train_labels[i*batch_size:(i+1)*batch_size]
        #print(i, batch)
        losses.append(net.step(batch))

    xhats = np.zeros(dev_set.shape[0])
    dev_set = (dev_set-ts_mean)/ts_std
    res = net(dev_set)
    res = res.view(len(dev_set), 784).detach().numpy()
    i = 0
    for r in res:
        xhats[i] = np.argmax(res[i])
        i += 1
    #print(xhats.shape)
    return losses,res, net


