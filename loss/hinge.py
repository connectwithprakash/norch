import numpy as np

from .base_loss import BaseLoss


class HingeLoss(BaseLoss):
    """
    Hinge loss is a margin-based loss function used for binary classification. 
    The hinge loss is 0 if the correct class score and the highest score for other classes differ by at least the margin parameter.

    Parameters:
        margin (float): The margin value. Default is 1.0.
        reduction (str): Specifies the reduction to apply to the output. Default is 'mean'.

    Attributes:
        margin (float): The margin value.
        reduction (str): Specifies the reduction to apply to the output.
        scores (numpy.ndarray): The predicted scores.
        labels (numpy.ndarray): The true class labels.
    """

    def __init__(self, margin=1.0, reduction='mean'):
        self.margin = margin
        self.reduction = reduction
        self.scores = None
        self.labels = None

    def forward(self, scores, labels):
        """
        Compute the hinge loss between the predicted scores and true class labels.

        Args:
            scores (numpy.ndarray): Predicted scores.
            labels (numpy.ndarray): True class labels.

        Returns:
            The computed hinge loss.
        """
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
        """
        Compute the gradient of the loss with respect to the input scores.

        Returns:
            The computed gradient.
        """
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
