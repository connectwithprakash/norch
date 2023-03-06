import numpy as np

from .base_loss import BaseLoss


class MSELoss(BaseLoss):
    """
    Mean Squared Error Loss.

    This is a loss function that calculates the mean squared error between two inputs.
    This loss is commonly used in regression problems.

    Parameters
    ----------
    reduction : str, optional
        Reduction type to apply on loss. 'mean' (default) or 'sum'.

    Attributes
    ----------
    reduction : str
        Reduction type to apply on loss.
    input : numpy.ndarray
        Input tensor of the loss function.
    target : numpy.ndarray
        Target tensor of the loss function.
    """

    def __init__(self, reduction='mean'):
        """
        Mean Squared Error loss constructor.

        Parameters:
        -----------
        reduction : str, optional
            Specifies the reduction to apply to the output. 
            Should be one of ['mean', 'sum']. Default is 'mean'.
        """
        self.reduction = reduction
        self.input = None
        self.target = None

    def forward(self, input, target):
        """
        Compute the Mean Squared Error loss between input and target.

        Parameters:
        -----------
        input : numpy.ndarray
            Input array.
        target : numpy.ndarray
            Target array.

        Returns:
        --------
        numpy.float64
            Mean Squared Error loss value.
        """
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
        """
        Compute the gradient of the Mean Squared Error loss.

        Returns:
        --------
        numpy.ndarray
            Gradient of the Mean Squared Error loss.
        """
        if self.reduction == 'mean':
            return 2 * (self.input - self.target) / self.input.shape[0]
        elif self.reduction == 'sum':
            return 2 * (self.input - self.target)

    def param(self):
        """
        Returns an empty list, as the Mean Squared Error loss has no parameters.

        Returns:
        --------
        list
            An empty list.
        """
        return []

    def grad(self):
        """
        Returns an empty list, as the Mean Squared Error loss has no gradients.

        Returns:
        --------
        list
            An empty list.
        """
        return []

    def __repr__(self):
        """
        Returns a string representation of the Mean Squared Error loss.

        Returns:
        --------
        str
            String representation of the Mean Squared Error loss.
        """
        return f"{self.__class__.__name__}(reduction=\"{self.reduction}\")"
