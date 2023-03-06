from abc import ABC


class Module(ABC):
    """
    An abstract base class for neural network modules.

    This class defines the interface that must be implemented by all neural network
    modules.

    Attributes
    ----------
    param : list
        A list of the module's learnable parameters.
    grad : list
        A list of the gradients of the module's learnable parameters.
    """

    @property
    def param(self):
        """
        Get the module's learnable parameters.

        Returns
        -------
        list
            A list of the module's learnable parameters.
        """
        return []

    @property
    def grad(self):
        """
        Get the gradients of the module's learnable parameters.

        Returns
        -------
        list
            A list of the gradients of the module's learnable parameters.
        """
        return []

    def forward(self, input):
        """
        Compute the forward pass of the module.

        Parameters
        ----------
        input : Any
            The input to the module.

        Returns
        -------
        Any
            The output of the module.
        """
        raise NotImplementedError

    def __call__(self, *input):
        """
        Call the forward method of the module.

        Parameters
        ----------
        input : tuple
            The input to the module.

        Returns
        -------
        Any
            The output of the module.
        """
        return self.forward(*input)

    def backward(self, gradwrtoutput):
        """
        Compute the backward pass of the module.

        Parameters
        ----------
        gradwrtoutput : Any
            The gradient of the loss with respect to the output of the module.

        Returns
        -------
        Any
            The gradient of the loss with respect to the input of the module.
        """
        raise NotImplementedError

    def load_param(self, *params):
        """
        Load the module's parameters.

        Parameters
        ----------
        params : tuple
            The parameters to load.
        """
        pass

    def __repr__(self):
        """
        Return a string representation of the module.

        Returns
        -------
        repr : str
            A string representation of the module.
        """
        return f"{self.__class__.__name__}()"
