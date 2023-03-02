import numpy as np

from norch.layer import Module


class ReLU(Module):
    def __init__(self, slope=0.1):
        self.slope = slope
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        out = np.where(self.mask, x, self.slope * x)
        return out

    def backward(self, dout):
        dx = dout.copy()
        dx[~self.mask] = self.slope * dx[~self.mask]
        return dx

    def __repr__(self):
        return f"{self.__class__.__name__}()"
