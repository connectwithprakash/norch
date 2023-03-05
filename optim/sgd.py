import numpy as np

from .base_optimizer import BaseOptimizer


class SGD(BaseOptimizer):
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for param_pair in self.params:
            for param in param_pair:
                param.grad = np.zeros_like(param)

    def step(self):
        for param_pair in self.params:
            for param in param_pair:
                param -= self.lr * param.grad

    def __repr__(self):
        return f"{self.__class__.__name__}(lr={self.lr})"


if __name__ == '__main__':
    a = SGD(1, 0.01)
