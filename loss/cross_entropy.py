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


class CrossEntropyWithLogitsLoss(BaseLoss):
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        self.shift_logits = None
        self.probs = None
        self.labels = None

    def forward(self, logits, labels):
        self.labels = labels
        max_logits = np.max(logits, axis=1, keepdims=True)
        shifted_logits = logits - max_logits
        self.shift_logits = shifted_logits
        log_sum_exp = np.log(
            np.sum(np.exp(shifted_logits), axis=1, keepdims=True))
        log_probs = shifted_logits - log_sum_exp
        self.probs = np.exp(log_probs)
        nll_loss = -np.sum(labels * log_probs, axis=1)
        if self.reduction == 'mean':
            return np.mean(nll_loss)
        elif self.reduction == 'sum':
            return np.sum(nll_loss)
        else:
            raise ValueError(
                "Invalid reduction type. Expected 'mean' or 'sum', but got {}".format(self.reduction))

    def backward(self):
        dlog_softmax = self.probs - self.labels
        if self.reduction == 'mean':
            dlog_softmax /= len(self.labels)
        elif self.reduction == 'sum':
            pass
        else:
            raise ValueError(
                "Invalid reduction type. Expected 'mean' or 'sum', but got {}".format(self.reduction))
        return dlog_softmax

    def __repr__(self):
        return f"{self.__class__.__name__}(reduction=\"{self.reduction}\")"
