import numpy as np
from .base_loss import BaseLoss

torch = np


class CrossEntropyLoss(BaseLoss):
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        self.input = None
        self.target = None
        super().__init__()

    def forward(self, input, target):
        if input.shape != target.shape:
            raise ValueError(
                f"input and target should have the same shape, but got {input.shape} and {target.shape}")

        self.input = input
        self.target = target
        if self.reduction == 'mean':
            return np.mean(-target * np.log(input))
        elif self.reduction == 'sum':
            return np.sum(-target * np.log(input))

    def backward(self):
        if self.reduction == 'mean':
            return -self.target / self.input / self.input.shape[0]
        elif self.reduction == 'sum':
            return -self.target / self.input

    def __repr__(self):
        return f"{self.__class__.__name__}(reduction=\"{self.reduction}\")"

