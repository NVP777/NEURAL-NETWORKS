import numpy as np


class ReLU:
    def __init__(self):
        self.X = None

    def forward(self, X):
        #saving input for backward propogation
        self.X = X

        return np.maximum(0, X)

    def backward(self, dA):
        #relu derivative is 1 where X > 0 and 0 where X <= 0
        #so we multiply the incoming gradient dA with this derivative
        return dA * (self.X > 0)