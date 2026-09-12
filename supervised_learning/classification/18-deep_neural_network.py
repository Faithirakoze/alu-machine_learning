#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """
        Class constructor

        nx is the number of input features
        layers is a list representing the number of nodes in
        each layer of the network
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                prev = nx
            else:
                prev = layers[i - 1]

            self.__weights["W" + str(i + 1)] = (
                np.random.randn(layers[i], prev) * np.sqrt(2 / prev)
            )
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for the number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for the cache dictionary"""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights dictionary"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        X is a numpy.ndarray with shape (nx, m) that contains
        the input data
            nx is the number of input features to the neuron
            m is the number of examples
        """
        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            W = self.__weights["W" + str(i)]
            b = self.__weights["b" + str(i)]
            A_prev = self.__cache["A" + str(i - 1)]

            z = np.matmul(W, A_prev) + b
            self.__cache["A" + str(i)] = 1 / (1 + np.exp(-z))

        return self.__cache["A" + str(self.__L)], self.__cache
