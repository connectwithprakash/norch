import numpy as np
from .base_optimizer import BaseOptimizer


class SGD(BaseOptimizer):
    """
    Implements stochastic gradient descent (SGD) optimizer with momentum.

    Parameters:
    -----------
    params : list of Tensors
        The parameters to optimize.
    lr : float, optional (default=0.01)
        Learning rate.
    """

    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        """
        Resets the gradients of all the parameters to zero.
        """
        for param_pair in self.params:
            for param in param_pair:
                param.grad = np.zeros_like(param)

    def step(self):
        """
        Performs a single optimization step.
        """
        for param_pair in self.params:
            for param in param_pair:
                param -= self.lr * param.grad

    def __repr__(self):
        """
        Returns a string representation of the optimizer with the current
        learning rate.
        """
        return f"{self.__class__.__name__}(lr={self.lr})"
