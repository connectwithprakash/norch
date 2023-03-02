import numpy as np

from norch.layer import Module


class Sigmoid(Module):
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        pos_mask = (self.input >= 0)
        neg_mask = (self.input < 0)
        z = np.zeros_like(self.input)
        z[pos_mask] = np.exp(-self.input[pos_mask])
        z[neg_mask] = np.exp(self.input[neg_mask])
        top = np.ones_like(self.input)
        top[neg_mask] = z[neg_mask]
        self.output = top / (1 + z)
        return self.output

    def backward(self, gradwrtoutput):
        grad_input = gradwrtoutput * self.output * (1 - self.output)
        return grad_input

    def __repr__(self):
        return f"{self.__class__.__name__}()"
