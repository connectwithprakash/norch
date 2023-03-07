import numpy as np


class Tensor(np.ndarray):
    """
    A class for creating and managing tensors for a deep learning model.

    Parameters
    ----------
    shape : tuple of ints
        The shape of the tensor.
    init_method : str, optional
        The initialization method for the tensor. Default is 'normal'.

    Returns
    -------
    tensor : Tensor object
        A Tensor object with the specified shape and initialized with the
        specified method.

    Attributes
    ----------
    gradient : numpy.ndarray
        A numpy array of zeros with the same shape as the tensor. This is
        used to store the gradient of the tensor during backpropagation.
    """

    def __new__(cls, shape, init_method='normal'):
        """
        Create a new Tensor object.

        Parameters
        ----------
        shape : tuple of ints
            The shape of the tensor.
        init_method : str, optional
            The initialization method for the tensor. Default is 'normal'.

        Returns
        -------
        tensor : Tensor object
            A Tensor object with the specified shape and initialized with the
            specified method.
        """
        # Initialize the tensor with the specified shape and initialization method,
        # and create an attribute for storing the gradient.
        tensor = cls.init_weights(
            shape=shape, init_method=init_method).view(cls)
        tensor.gradient = np.zeros(shape)
        return tensor

    @property
    def grad(self):
        """
        Get the gradient of the tensor.

        Returns
        -------
        gradient : numpy.ndarray
            A numpy array containing the gradient of the tensor.
        """
        return self.gradient

    @grad.setter
    def grad(self, value):
        """
        Set the gradient of the tensor.

        Parameters
        ----------
        value : numpy.ndarray
            A numpy array containing the gradient of the tensor.
        """
        self.gradient = value

    @staticmethod
    def init_weights(shape, init_method):
        """
        Initialize the tensor with the specified shape and initialization method.

        Parameters
        ----------
        shape : tuple of ints
            The shape of the tensor.
        init_method : str
            The initialization method for the tensor. Must be 'zero' or 'normal'.

        Returns
        -------
        weights : numpy.ndarray
            A numpy array containing the initialized weights.
        """
        # Initialize the tensor with the specified method.
        if init_method == 'zero':
            return np.zeros(shape)
        elif init_method == 'normal':
            return np.random.normal(size=shape)
        else:
            # Raise an error if the specified method is not supported.
            raise NotImplementedError("Unsupported initialization method.")

    def __array_finalize__(self, obj):
        """
        Finalize the creation of the Tensor object.

        Parameters
        ----------
        obj : numpy.ndarray
            The numpy array to be finalized.
        """
        # Set the gradient attribute of the Tensor object.
        if obj is None:
            return
        self.gradient = getattr(obj, 'gradient', None)
