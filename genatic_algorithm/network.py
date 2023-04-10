"""Class that represents the network to be evolved."""
import random
import logging
from model import HarmonicModel
import torch
from torch import nn
import pickle
from math import isnan
import numpy as np
from energy import get_energy


class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices):
        """Initialize our network.

        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                        'nb_neurons': [4, 8, 16],
                        'nb_layers': [1, 2, 3, 4],
                        'activation': ['relu', 'tanh', 'sigmoid', ''],
                        # 'pretraining_V0' : [-20, -10, -5, 0, 5, 10, 20],
                        # 'pretraining_L' : [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        """
        self.accuracy = 0.
        self.iter = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters
        self.net = HarmonicModel()

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])
        ########### TODO: FUTURE DEVELOPMENT: make it so we select an activation per-layer #####################

        # Get our network parameters.
        nb_layers = self.network['nb_layers']
        nb_neurons = self.network['nb_neurons']
        activation = self.network['activation']
        # pre_v0 = self.network['pretraining_V0']
        # pre_L = self.network['pretraining_L']
        mesh = [-10, 10]
        mesh_density = 200

        self.net = HarmonicModel(mesh, mesh_density, pretraining=None)
        # self.net = HarmonicModel(mesh, mesh_density, pretraining=[pre_v0, pre_L])

        w1 = torch.rand(nb_neurons, 1, requires_grad=True) * (-1.)
        b = torch.rand(nb_neurons, requires_grad=True) * \
            2. - 1.    # Set of bias parameters

        self.net.add(nn.Linear(1, nb_neurons, bias=True), w1, b, activation)

        for _ in range(nb_layers):
            self.net.add(nn.Linear(nb_neurons, nb_neurons,
                         bias=True), activation=activation)

        self.net.add(nn.Linear(nb_neurons, 1, bias=True))

    def create_set(self, network):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.network = network

        # Get our network parameters.
        nb_layers = self.network['nb_layers']
        nb_neurons = self.network['nb_neurons']
        activation = self.network['activation']
        # pre_v0 = self.network['pretraining_V0']
        # pre_L = self.network['pretraining_L']
        mesh = [-10, 10]
        mesh_density = 200

        self.net = HarmonicModel(mesh, mesh_density, pretraining=None)
        # self.net = HarmonicModel(mesh, mesh_density, pretraining=[pre_v0, pre_L])

        w1 = torch.rand(nb_neurons, 1, requires_grad=True) * (-1.)
        b = torch.rand(nb_neurons, requires_grad=True) * \
            2. - 1.    # Set of bias parameters

        self.net.add(nn.Linear(1, nb_neurons, bias=True), w1, b, activation)

        for _ in range(nb_layers):
            self.net.add(nn.Linear(nb_neurons, nb_neurons,
                         bias=False), activation=activation)

        self.net.add(nn.Linear(nb_neurons, 1, bias=False))

        return self.net

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
