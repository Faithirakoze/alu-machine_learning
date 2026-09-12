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

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Y is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        A is a numpy.ndarray with shape (1, m) containing the
        activated output of the neuron for each example
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions

        X is a numpy.ndarray with shape (nx, m) that contains
        the input data
            nx is the number of input features to the neuron
            m is the number of examples
        Y is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Y is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        cache is a dictionary containing all the intermediary
        values of the network
        alpha is the learning rate
        """
        m = Y.shape[1]
        L = self.__L
        weights = self.__weights.copy()

        dz = cache["A" + str(L)] - Y

        for i in range(L, 0, -1):
            A_prev = cache["A" + str(i - 1)]
            W = weights["W" + str(i)]

            dW = (1 / m) * np.matmul(dz, A_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            if i > 1:
                dz = np.matmul(W.T, dz) * (A_prev * (1 - A_prev))

            self.__weights["W" + str(i)] = W - alpha * dW
            self.__weights["b" + str(i)] = weights["b" + str(i)] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the deep neural network

        X is a numpy.ndarray with shape (nx, m) that contains
        the input data
            nx is the number of input features to the neuron
            m is the number of examples
        Y is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        iterations is the number of iterations to train over
        alpha is the learning rate
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

        return self.evaluate(X, Y)
