import numpy as np
from norch.layer import Module


class Sigmoid(Module):
    """Sigmoid activation function module."""

    def __init__(self):
        """Initialize the Sigmoid module."""
        self.input = None

    def forward(self, input):
        """
        Compute the forward pass of the Sigmoid activation function.

        Args:
            input (ndarray): Input array.

        Returns:
            ndarray: Output array after applying Sigmoid activation function.
        """
        self.input = input
        pos_mask = (self.input >= 0)
        neg_mask = (self.input < 0)
        z = np.zeros_like(self.input)
        z[pos_mask] = np.exp(-self.input[pos_mask])
        z[neg_mask] = np.exp(self.input[neg_mask])
        top = np.ones_like(self.input)
        top[neg_mask] = z[neg_mask]
        self.output = top / (1 + z)
        return self.output

    def backward(self, gradwrtoutput):
        """
        Compute the backward pass of the Sigmoid activation function.

        Args:
            gradwrtoutput (ndarray): Gradient of the loss function with respect to the output.

        Returns:
            ndarray: Gradient of the loss function with respect to the input.
        """
        grad_input = gradwrtoutput * self.output * (1 - self.output)
        return grad_input
