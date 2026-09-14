import numpy as np


class Linear:
    def __init__(self, in_features, out_features):
        
        self.W = np.random.randn(in_features, out_features) * 0.1

      
        self.b = np.zeros((1, out_features))

        
        self.dW = None
        self.db = None

        #we need to save the input X because backward propagation will need it
        self.X = None

    def forward(self, X):
        self.X = X

        #Z = XW + b
        return X @ self.W + self.b

    def backward(self, dZ):
        """
        dZ = gradient of the loss with respect to this layer's output
        """

        #gradient with respect to weights
        self.dW = self.X.T @ dZ

        #gradient with respect to bias
        self.db = np.sum(dZ, axis=0, keepdims=True)

        #gradient with respect to input, to pass to previous layer
        dX = dZ @ self.W.T

        return dX

    def update(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db