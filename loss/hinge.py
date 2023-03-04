import numpy as np

from .base_loss import BaseLoss


class HingeLoss(BaseLoss):
    def __init__(self, margin=1.0, reduction='mean'):
        self.margin = margin
        self.reduction = reduction
        self.scores = None
        self.labels = None

    def forward(self, scores, labels):
        self.scores = scores
        self.labels = labels
        y_true = 2 * labels - 1  # Convert one-hot encoding to +1/-1
        hinge_loss = np.maximum(0, self.margin - y_true * scores)
        class_loss = np.sum(hinge_loss, axis=1)
        if self.reduction == 'mean':
            return np.mean(class_loss)
        elif self.reduction == 'sum':
            return np.sum(class_loss)
        else:
            raise ValueError(
                "Invalid reduction type. Expected 'mean' or 'sum', but got {}".format(self.reduction))

    def backward(self):
        y_true = 2 * self.labels - 1  # Convert one-hot encoding to +1/-1
        d_scores = -y_true * (self.margin - y_true * self.scores > 0)
        if self.reduction == 'mean':
            d_scores /= len(self.labels)
        elif self.reduction == 'sum':
            pass
        else:
            raise ValueError(
                "Invalid reduction type. Expected 'mean' or 'sum', but got {}".format(self.reduction))
        return d_scores

    def __repr__(self):
        return f"{self.__class__.__name__}(margin={self.margin}, reduction='{self.reduction}')"
