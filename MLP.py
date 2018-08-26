import numpy as np

"""
A simple python program to implement a Multilayer Perceptron with 2 hidden layers with 20 neurons each.
"""


class MLP(object):

    def sigmoid(self, x):
        """
        Sigmoid function that is the activation function for our neural network.
        :param x: x value inputted into function
        :return: activation value
        """
        return (1 + np.exp(-x)) ** (-1)

    def __init__(self, learn, x, y):
        """
        :param learn: learning rate
        :type learn: float
        :param x: input training data
        :type x: ndarray (n0-features, m0-samples)
        :param y: correct output data
        :type y: ndarray (m0 samples)
        """
        self.learn = learn
        self.x = x
        self.y = y
        self.w0 = np.random.random((20, self.x.shape[1]))
        self.w1 = np.random.random((20, 20))
        self.w2 = np.random.random(20)
        self.sigmoid = np.vectorize(self.sigmoid)


    def train(self):
        """
        Train the weights with data set x for outputs y
        :return: weights (w0, w1, w2) and errors for each iteration
        """
        self.error = []
        errors = 0
        for xi, target in zip(self.x, self.y):
            self.feedforward(xi)
            cost = target.T - self.output3

            for i in range(self.w2.shape[0]):
                self.w2[i] += -self.learn * cost.sum() * self.sigmoid(self.z2) * \
                              (1 - self.sigmoid(self.z2)) * self.output2[i]

            for i in range(self.w1.shape[0]):
                for j in range(self.w1.shape[1]):
                    self.w1[i, j] += -self.learn * cost.sum() * self.sigmoid(self.z2) * (1 - self.sigmoid(self.z2)) * self.w2[i] * \
                               self.sigmoid(self.z1[i]) * (1 - self.sigmoid(self.z1[i])) * self.output1[i]

            for i in range(self.w0.shape[0]):
                for j in range(self.w0.shape[1]):
                    self.w0[i, j] += -self.learn * cost.sum() * self.sigmoid(self.z2) * (1 - self.sigmoid(self.z2)) * self.w2[i] * \
                                    self.sigmoid(self.z1[i]) * (1 - self.sigmoid(self.z1[i])) * self.w1[i, j] * \
                                    self.sigmoid(self.z0[i]) * (1 - self.sigmoid(self.z0[i])) * xi[j]

            errors = cost
        self.error.append(errors)

        return self

    def feedforward(self, x):
        """
        Predict the output given the inputs
        :param x: input vector of features
        :type x: ndarray
        :return: All activation values and x values.
        """
        self.z0 = np.dot(self.w0, x)
        self.output1 = self.sigmoid(self.z0)
        self.z1 = np.matmul(self.w1, self.output1)
        self.output2 = self.sigmoid(self.z1)
        self.z2 = np.matmul(self.w2, self.output2)
        self.output3 = self.sigmoid(self.z2)

        return self.z0, self.output1, self.z1, self.output2, self.z2, self.output3

    def print_weights(self):
        print('Layer 1: ')
        print(self.w0)
        print('Layer 2: ')
        print(self.w1)
        print('Layer 3: ')
        print(self.w2)
