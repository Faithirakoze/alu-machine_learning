#!/usr/bin/env python3
"""Defines a deep neural network performing multiclass classification"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """Defines a deep neural network performing multiclass classification"""

    def __init__(self, nx, layers, activation='sig'):
        """
        Class constructor

        nx is the number of input features
        layers is a list representing the number of nodes in
        each layer of the network
        activation represents the type of activation function
        used in the hidden layers
            sig represents a sigmoid activation
            tanh represents a tanh activation
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if activation != 'sig' and activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

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

    @property
    def activation(self):
        """Getter for the activation function type"""
        return self.__activation

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

            if i == self.__L:
                # softmax activation for the output layer
                t = np.exp(z)
                self.__cache["A" + str(i)] = t / np.sum(t, axis=0,
                                                        keepdims=True)
            else:
                # hidden layer activation depends on self.__activation
                if self.__activation == 'sig':
                    self.__cache["A" + str(i)] = 1 / (1 + np.exp(-z))
                else:
                    self.__cache["A" + str(i)] = np.tanh(z)

        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using categorical
        cross-entropy

        Y is a one-hot numpy.ndarray with shape (classes, m) that
        contains the correct labels for the input data
        A is a numpy.ndarray with shape (classes, m) containing the
        activated output of the neuron for each example
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions

        X is a numpy.ndarray with shape (nx, m) that contains
        the input data
            nx is the number of input features to the neuron
            m is the number of examples
        Y is a one-hot numpy.ndarray with shape (classes, m) that
        contains the correct labels for the input data
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        prediction = np.zeros_like(A)
        prediction[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Y is a numpy.ndarray with shape (classes, m) that contains
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
                if self.__activation == 'sig':
                    dz = np.matmul(W.T, dz) * (A_prev * (1 - A_prev))
                else:
                    dz = np.matmul(W.T, dz) * (1 - A_prev ** 2)

            self.__weights["W" + str(i)] = W - alpha * dW
            self.__weights["b" + str(i)] = weights["b" + str(i)] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network

        X is a numpy.ndarray with shape (nx, m) that contains
        the input data
            nx is the number of input features to the neuron
            m is the number of examples
        Y is a numpy.ndarray with shape (classes, m) that contains
        the correct labels for the input data
        iterations is the number of iterations to train over
        alpha is the learning rate
        verbose is a boolean that defines whether or not to print
        information about the training
        graph is a boolean that defines whether or not to graph
        information about the training once training has completed
        step controls how often data is captured/printed/plotted
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iters = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)

            if i % step == 0 or i == iterations:
                costs.append(cost)
                iters.append(i)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(iters, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format

        filename is the file to which the object should be saved
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object

        filename is the file from which the object should be loaded

        Returns: the loaded object, or None if filename doesn't exist
        """
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
