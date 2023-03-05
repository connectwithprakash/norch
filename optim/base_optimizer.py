class BaseOptimizer(object):
    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError
