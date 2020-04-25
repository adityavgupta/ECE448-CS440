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
        self.encoder = torch.nn.Sequential(torch.nn.Conv2d(1,15, kernel_size=6), nn.ReLU())#, torch.nn.Conv2d(6,1,kernel_size=10), nn.ReLU())
        self.decoder = torch.nn.Sequential(torch.nn.ConvTranspose2d(15,1, kernel_size=6), torch.nn.ReLU())#, torch.nn.ConvTranspose2d(6,1,kernel_size=10), nn.ReLU())
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
        x_prime = x.view(x.shape[0],1,28,28)
        return self.decoder(self.encoder(x_prime)).view(x.shape[0], 28*28)

    def step(self, x):
        # x [100, 784]
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        optimizer = torch.optim.Adam(self.get_parameters(), lr=self.lrate)
        _target = self.forward(x)
        _loss = self.loss_fn(_target, _target) #loss function is MSELoss()

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
    lrate = 1e-4
    distance = nn.MSELoss() #loss_fn
    in_size, out_size = 784, 5

    batch_num = train_set.size()[0]//batch_size
    
    net = NeuralNet(lrate, distance, in_size, out_size)
    losses = list()
    
    for i in range(n_iter):
        start = (i%batch_num)*batch_size
        batch = train_set[start:start+batch_size]
        losses.append(net.step(batch))

    xhats = net(dev_set)
    xhats = xhats.view(len(dev_set), 784).detach().numpy()

    return losses, xhats, net

