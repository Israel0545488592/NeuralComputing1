import numpy as np

''' TODO: precede with jupiter'''

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

    def __init__(self, input_size : int, learning_factor : float):

        self.weights = np.random.uniform(-1, 1, input_size) + 0.001

        self.learning_factor = learning_factor


    def predict (self, example : np.ndarray):
        return example.dot(self.weights)


    def learn (self, example : np.ndarray, positive : bool):

        pred = self.predict(example)

        if (sigmoid(pred) < 0.5 and positive) or (sigmoid(pred) >= 0.5 and not positive):    # mistake

            gradient = np.array([0] * len(self.weights))

            gradient = gradient.astype('float64')

            for i in range(len(gradient)):

                # Taking the partial derivatives of the MSE times the learning rate

                if positive:
                    gradient[i] = self.learning_factor * (1 - pred) * example[i]
                else:
                    gradient[i] = self.learning_factor * (-1 - pred) * example[i]

            self.weights -= gradient

            return gradient



    def train (self, data : np.ndarray):

        if len(data.shape) < 2:
            raise ValueError('expecting attributes AND class')

        histroy = []

        for i in range(len(data)):

            if data[i][-1] == 1:
                positive = True
            elif data[i][-1] == -1:
                positive = False
            else:
                raise ValueError('Target should ve bipolar')

        error = self.learn(data[i][:-1])

        if error is not None:   # mistake
            histroy.append(norm(error))

        return histroy

