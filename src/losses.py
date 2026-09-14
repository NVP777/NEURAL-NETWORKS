import numpy as np


class MSELoss:
    def __init__(self):
        self.prediction = None
        self.target = None

    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target

        return np.mean((prediction - target) ** 2)

    def backward(self):
        n = self.prediction.size

        return 2 * (self.prediction - self.target) / n