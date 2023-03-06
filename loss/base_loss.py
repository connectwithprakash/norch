from abc import ABC


class BaseLoss(ABC):
    """
    Abstract base class for implementing custom loss functions.

    This class defines the interface for custom loss functions, which should
    implement the `forward` and `backward` methods. The `forward` method
    computes the loss between the `input` and `target` tensors, while the
    `backward` method computes the gradients of the loss with respect to the
    `input` tensor. The `__call__` method is provided for convenience and is
    equivalent to calling the `forward` method.

    Note:
        This class should not be used directly, but rather be subclassed by a
        specific loss function implementation.

    Attributes:
        None

    Methods:
        forward(input, target):
            Compute the loss value for a given input and target.

        backward():
            Compute the gradient of the loss value with respect to the input.
    """

    def forward(self, input, target):
        """
        Compute the loss value for a given input and target.

        Args:
            input (numpy.ndarray): The input data.
            target (numpy.ndarray): The target data.

        Returns:
            float: The loss value.
        """
        raise NotImplementedError

    def backward(self):
        """
        Compute the gradient of the loss value with respect to the input.

        Args:
            None

        Returns:
            numpy.ndarray: The gradient of the loss value with respect to the input.
        """
        raise NotImplementedError

    def __call__(self, *input):
        """
        Call the forward method with the given input.

        Args:
            input: The input arguments.

        Returns:
            float: The loss value.
        """
        return self.forward(*input)
