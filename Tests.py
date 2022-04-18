import unittest as unt
import numpy as np
from adaline import sigmoid, norm, adaline

class Test(unt.TestCase):

    def test_sigmoid(self):

        self.assertEqual(sigmoid(0), 0.5)
        self.assertEqual(sigmoid(100), 1)
        self.assertEqual(sigmoid(-100), 0)

    def test_norm(self):

        vector = np.arange(5)
        self.assertEqual(np.sqrt(30), norm(vector))
        vector = np.ones(25)
        self.assertEqual(5, norm(vector))
        vector *= -1
        self.assertEqual(5, norm(vector))
        vector *= 0
        self.assertEqual(0, norm(vector))


class Test(unt.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.neuron = adaline(2, 0.3)

    def test_predict(cls):

        cls.assertEqual(len(cls.neuron.weights), 2)
        cls.assertTrue(cls.neuron.weights[0] < 1)
        cls.assertTrue(cls.neuron.weights[0] > -1)
        cls.assertTrue(cls.neuron.weights[1] != 0)
        cls.assertTrue(cls.neuron.weights[1] < 1)
        cls.assertTrue(cls.neuron.weights[1] > -1)
        cls.assertTrue(cls.neuron.weights[1] != 0)


if __name__ == '__main__':
    unt.main()