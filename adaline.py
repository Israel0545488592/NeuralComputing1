from turtle import forward
import numpy as np

# Mathematical Essentials

def sigmoid (x):   # the threshold function

    if x > 50:
        return 1
    elif x < -50:
        return -1
    else:
        return 1 / (1 + np.exp(-x))

def norm (vector : np.ndarray):
    return np.sqrt(vector.dot(vector))


class adaline:

    def __init__(self, input_size : int, learning_rate : float):

        self.weights = np.random.uniform(-1, 1, input_size) + 0.001

        self.learning_factor = learning_rate


    def forward (self, example : np.ndarray):   # forwarding the data acording to the weights
        return example.dot(self.weights)

    def predict (self, example : np.ndarray):   # applying activation function
        return sigmoid(self.forward(example))


    def learn (self, example : np.ndarray, positive : bool):

        pred = self.predict(example)

        if (pred < 0.5 and positive) or (pred >= 0.5 and not positive):    # mistake

            # Taking the partial derivatives of the MSE times the learning rate

            if positive:
                gradient = (-1 - pred) * example
            else:
                gradient = (1 - pred) * example

            self.weights -= self.learning_factor * gradient

            return gradient



    def train (self, data : np.ndarray, labels : np.ndarray):

        if len(data.shape) < 2:
            raise ValueError('expecting attributes AND class')

        histroy = []

        for i in range(len(data)):

            error = self.learn(data[i], labels[i])

            if error is not None:   # mistake
                histroy.append(norm(error))

        return histroy