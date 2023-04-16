"""Class that represents the network to be evolved."""
import random
import logging
import torch
from torch import nn
import pickle
from math import isnan
import numpy as np


class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices):
        """Initialize our network.

        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                        'num_layers': [1, 2, 3, 4],
                        'num_hidden': [4, 8, 16],
                        'preepochs': [1, 2, 3, 4],
                        'load_model_name': [1, 2, 3, 4]

        """
        self.accuracy = 0.
        self.iter = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters
        

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])


    def create_set(self, network):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.network = network        


    def train(self, dataset):
        """Train the network and record the accuracy.

        Args:
            dataset (dict): 
                'V0': (array) 
                'L': (array) 

        """
        try:
            with open('real_energies.pickle', 'rb') as handle:
                real_ene = pickle.load(handle)
        except:
            real_ene = []
        max_epochs = 5000

        if self.accuracy == 0.:
            self.accuracy, self.iter = self.train_and_score(
                epochs=max_epochs, V0=dataset['V0'], L=dataset['L'], real_energies=real_ene)
            if self.accuracy is None or isnan(self.accuracy):
                self.accuracy = 100000

    def train_and_score(self, optimizer=None, epochs=5000, learning_rate=1e-2, V0=[20], L=[0.5], delta=1e-5, print_log=True, real_energies=[]):
        '''
        real_energies : array of shape (2,2) ---> real_energies[V0][L]
        '''
        i = 0
        acc = 0
        n_iter = 0
        for v in V0:
            for l in L:
                self.net = None
                self.net = self.create_set(self.network)
                if len(real_energies) == 0:
                    real_ene = get_energy(v, l, plot=False)
                    # real_ene = real_ene[0]
                else:
                    print("skipping calculating real energy")
                    real_ene = real_energies[v][l]

                if self.net.pretrain:
                    print("pretraining: ", self.net.pretrain_values)
                    self.net.train(optimizer=optimizer, epochs=epochs, learning_rate=learning_rate,
                                   V0=self.net.pretrain_values[0], L=self.net.pretrain_values[1], delta=delta)

                ene, num_iter = self.net.train(
                    optimizer=optimizer, epochs=epochs, learning_rate=learning_rate, V0=v, L=l, delta=delta)

                err = np.abs(real_ene - ene)
                acc += err
                if num_iter >= epochs - 1:
                    acc += 100
                n_iter += num_iter
                i += 1

                if print_log:
                    print()
                    print("for V0=", v, " and L=", l)
                    print("predicted energy : ", ene)
                    print("real energy: ", real_ene)
                    print("error: ", err)
                    print("number of iterations: ", num_iter)
                    print()

        self.accuracy = acc / i
        self.avg_num_iter = n_iter / i
        if print_log:
            print("average error: ", self.accuracy)
            print("average number of iterations:", self.avg_num_iter)
            print()
        return self.accuracy, self.avg_num_iter

    def print_network(self):
        """Print out a network."""
        logging.info("Network average error: %.10f" % (self.accuracy))
        logging.info("Network average iterations: %.10f" % (self.iter))
        logging.info(self.network)
        logging.info(self.net.layers)
        logging.info("----------------------------------------------")
