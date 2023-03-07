import numpy as np
from .base_loss import BaseLoss

import numpy as np
from .base_loss import BaseLoss


class CrossEntropyLoss(BaseLoss):
    """
    Computes the Cross Entropy loss between the input and target tensors.

    Parameters:
    -----------
    reduction : str, optional (default='mean')
        Specifies the reduction to apply to the output. Possible values are 'mean' and 'sum'.

    Attributes:
    -----------
    reduction : str
        The reduction type to apply to the output.
    input : np.ndarray
        The input tensor.
    target : np.ndarray
        The target tensor.
    """

    def __init__(self, reduction='mean'):
        """
        Initializes the CrossEntropyLoss instance.

        Parameters:
        -----------
        reduction : str, optional (default='mean')
            Specifies the reduction to apply to the output. Possible values are 'mean' and 'sum'.
        """
        self.reduction = reduction
        self.input = None
        self.target = None
        super().__init__()

    def forward(self, input, target):
        """
        Computes the forward pass of the Cross Entropy loss.

        Parameters:
        -----------
        input : np.ndarray
            The input tensor.
        target : np.ndarray
            The target tensor.

        Returns:
        --------
        loss : float
            The loss value.
        """
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
        """
        Computes the backward pass of the Cross Entropy loss.

        Returns:
        --------
        grad_input : np.ndarray
            The gradient of the input tensor.
        """
        if self.reduction == 'mean':
            return -self.target / self.input / self.input.shape[0]
        elif self.reduction == 'sum':
            return -self.target / self.input

    def __repr__(self):
        """
        Returns a string representation of the CrossEntropyLoss instance.

        Returns:
        --------
        repr : str
            A string representation of the CrossEntropyLoss instance.
        """
        return f"{self.__class__.__name__}(reduction=\"{self.reduction}\")"


class CrossEntropyWithLogitsLoss(BaseLoss):
    """
    Computes the cross-entropy loss between predicted logits and target labels.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            Can be 'mean' or 'sum'. Default is 'mean'.
    """

    def __init__(self, reduction='mean'):
        self.reduction = reduction
        self.shift_logits = None
        self.probs = None
        self.labels = None

    def forward(self, logits, labels):
        """
        Computes the forward pass of the cross-entropy loss.

        Args:
            logits (numpy.ndarray): Predicted logits of shape (batch_size, num_classes).
            labels (numpy.ndarray): Target labels of shape (batch_size, num_classes).

        Returns:
            numpy.ndarray: The cross-entropy loss.
        """
        self.labels = labels
        max_logits = np.max(logits, axis=1, keepdims=True)
        # shift the logits to avoid numerical instability
        shifted_logits = logits - max_logits
        self.shift_logits = shifted_logits
        log_sum_exp = np.log(
            np.sum(np.exp(shifted_logits), axis=1, keepdims=True))  # compute log of the sum of exponentials of the shifted logits
        # compute the log of the softmax probabilities
        log_probs = shifted_logits - log_sum_exp
        self.probs = np.exp(log_probs)
        # compute negative log-likelihood loss
        nll_loss = -np.sum(labels * log_probs, axis=1)
        if self.reduction == 'mean':
            return np.mean(nll_loss)  # return the mean loss across the batch
        elif self.reduction == 'sum':
            return np.sum(nll_loss)  # return the sum of loss across the batch
        else:
            raise ValueError(
                "Invalid reduction type. Expected 'mean' or 'sum', but got {}".format(self.reduction))

    def backward(self):
        """
        Computes the backward pass of the cross-entropy loss.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the logits.
        """
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
        """
        Returns a string representation of the loss module.

        Returns:
            str: A string representation of the loss module.
        """
        return f"{self.__class__.__name__}(reduction=\"{self.reduction}\")"
