class BaseOptimizer(object):
    """Base class for all optimizers.

    Optimizers are responsible for updating the parameters of the model 
    during training to minimize the loss function. 

    Each optimizer should implement the `step` and `zero_grad` methods.
    """

    def step(self):
        """Performs a single optimization step."""
        raise NotImplementedError

    def zero_grad(self):
        """Zeroes out the gradients of all parameters."""
        raise NotImplementedError
