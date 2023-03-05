import numpy as np

from .base_loss import BaseLoss


class MSE(BaseLoss):
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        self.input = None
        self.target = None

    def forward(self, input, target):
        if input.shape != target.shape:
            raise ValueError(
                f"input and target should have the same shape, but got {input.shape} and {target.shape}")

        self.input = input
        self.target = target
        if self.reduction == 'mean':
            return np.mean((input - target) ** 2)
        elif self.reduction == 'sum':
            return np.sum((input - target) ** 2)

    def backward(self):
        if self.reduction == 'mean':
            return 2 * (self.input - self.target) / self.input.shape[0]
        elif self.reduction == 'sum':
            return 2 * (self.input - self.target)

    def param(self):
        return []

    def grad(self):
        return []

    def __repr__(self):
        return f"{self.__class__.__name__}(reduction=\"{self.reduction}\")"
