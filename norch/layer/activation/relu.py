import numpy as np
from norch.layer import Module


class ReLU(Module):
    """
    A class for the Rectified Linear Unit (ReLU) activation function.

    Parameters
    ----------
    slope : float, optional
        The slope of the function for x < 0. Default is 0.1.

    Attributes
    ----------
    mask : numpy.ndarray
        A boolean mask that is used during the forward pass to identify which
        values of the input are positive and which are negative.
    """

    def __init__(self, slope=0.1):
        """
        Initialize a ReLU object.

        Parameters
        ----------
        slope : float, optional
            The slope of the function for x < 0. Default is 0.1.
        """
        self.slope = slope
        self.mask = None

    def forward(self, x):
        """
        Compute the forward pass of the ReLU activation function.

        Parameters
        ----------
        x : numpy.ndarray
            The input to the activation function.

        Returns
        -------
        out : numpy.ndarray
            The output of the activation function.
        """
        # Create a boolean mask to identify which values of the input are positive.
        self.mask = x > 0
        # Apply the ReLU function to the input.
        out = np.where(self.mask, x, self.slope * x)
        return out

    def backward(self, dout):
        """
        Compute the backward pass of the ReLU activation function.

        Parameters
        ----------
        dout : numpy.ndarray
            The gradient of the output with respect to the loss.

        Returns
        -------
        dx : numpy.ndarray
            The gradient of the input with respect to the loss.
        """
        # Compute the gradient of the input with respect to the loss.
        dx = dout.copy()
        dx[~self.mask] = self.slope * dx[~self.mask]
        return dx
