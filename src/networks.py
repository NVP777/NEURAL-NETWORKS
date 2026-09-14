from src.layers import Linear
from src.activations import ReLU


class NeuralNetwork:
    def __init__(self):
        self.layer1 = Linear(2, 8)
        self.relu = ReLU()
        self.layer2 = Linear(8, 1)

    def forward(self, X):
        z1 = self.layer1.forward(X)
        a1 = self.relu.forward(z1)
        output = self.layer2.forward(a1)

        return output

    def backward(self, dLoss):
        da1 = self.layer2.backward(dLoss)
        dz1 = self.relu.backward(da1)
        self.layer1.backward(dz1)

    def update(self, learning_rate):
        self.layer1.update(learning_rate)
        self.layer2.update(learning_rate)