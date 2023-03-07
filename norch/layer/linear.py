import numpy as np

from norch.dtype import Tensor
from norch.layer import Module


class Linear(Module):
    """
    A linear layer module that performs a linear transformation of input data.
    """

    def __init__(self, in_features, out_features, bias=True, init_method='normal'):
        """
        Constructor for Linear layer.

        Args:
        - in_features: int, the number of input features
        - out_features: int, the number of output features
        - bias: bool, whether to include a bias term in the transformation
        - init_method: str, the weight initialization method to use
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.init_method = init_method
        self.W = None
        self.b = None
        self.input = None
        self.init_weights()

    def init_weights(self):
        """
        Initializes the weight tensor and bias tensor (if bias=True) for the linear layer
        """
        self.W = Tensor((self.in_features, self.out_features),
                        init_method=self.init_method)
        self.b = Tensor(self.out_features)

    def load_param(self, *params):
        """
        Loads weight and bias tensors into the linear layer.

        Args:
        - params: tuple, contains the weight and bias tensors
        """
        self.W = params[0]
        if self.bias:
            self.b = params[1]

    def forward(self, input):
        """
        Forward pass through the linear layer.

        Args:
        - input: numpy array, the input data tensor

        Returns:
        - output: numpy array, the output data tensor after linear transformation
        """
        self.input = input
        output = np.matmul(input, self.W)
        if self.bias:
            output += self.b
        return output

    def backward(self, gradwrtoutput):
        """
        Backward pass through the linear layer.

        Args:
        - gradwrtoutput: numpy array, the gradient of the loss with respect to the output of the layer

        Returns:
        - grad_input: numpy array, the gradient of the loss with respect to the input of the layer
        """
        self.W.grad = np.matmul(self.input.T, gradwrtoutput)
        if self.bias:
            self.b.grad = np.sum(gradwrtoutput, axis=0)
        return np.matmul(gradwrtoutput, self.W.T)

    @property
    def param(self):
        """
        Getter method for the weight and bias tensors.

        Returns:
        - params: list, the weight and bias tensors
        """
        return [self.W, self.b]

    @property
    def grad(self):
        """
        Getter method for the gradients of the weight and bias tensors.

        Returns:
        - grads: list, the gradients of the weight and bias tensors
        """
        return [self.W.grad, self.b.grad]

    def __repr__(self):
        """
        Returns a string representation of the linear layer.

        Returns:
        - string: str, a string representation of the linear layer
        """
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, init_method=\"{self.init_method}\")"
