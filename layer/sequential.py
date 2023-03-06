from norch.layer import Module


class Sequential(Module):
    """
    A sequential container for holding a list of layers and applying them in order during forward pass.

    Args:
        *args (Module): Any number of module objects.

    Attributes:
        layers (list): List of module objects.

    """

    def __init__(self, *args):
        """
        Initializes the Sequential object with the given list of layers.

        Args:
            *args (Module): Any number of module objects.
        """
        self.layers = args

    def forward(self, input):
        """
        Passes the input through the layers in order and returns the output.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for module in self.layers:
            input = module(input)
        return input

    def backward(self, gradwrtoutput):
        """
        Passes the gradient with respect to the output through the layers in reverse order and returns the gradient with respect to the input.

        Args:
            gradwrtoutput (torch.Tensor): Gradient with respect to the output.

        Returns:
            torch.Tensor: Gradient with respect to the input.
        """
        for module in reversed(self.layers):
            gradwrtoutput = module.backward(gradwrtoutput)
        return gradwrtoutput

    @property
    def param(self):
        """
        Returns a list of the learnable parameters for each module in the layers.

        Returns:
            list: List of learnable parameters.
        """
        params = []
        for module in self.layers:
            params.append(module.param)
        return params

    @property
    def grad(self):
        """
        Returns a list of the gradients with respect to the learnable parameters for each module in the layers.

        Returns:
            list: List of gradients with respect to the learnable parameters.
        """
        grads = []
        for module in self.layers:
            grads += module.grad
        return grads

    def load_param(self, *params):
        """
        Loads the specified parameters into each module in the layers.

        Args:
            *params (tuple): Tuples of parameter tensors.

        Returns:
            None.
        """
        for module in self.layers:
            module.load_param(params)

    def __repr__(self):
        """
        Returns a string representation of the Sequential object.

        Returns:
            str: String representation of the Sequential object.
        """
        layers_str = "\n".join(
            [f"  ({i}): {layer}" for i, layer in enumerate(self.layers)])
        return f"{self.__class__.__name__}(\n{layers_str}\n)"
