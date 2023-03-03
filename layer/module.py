from abc import ABC


class Module(ABC):
    def forward(self, input):
        raise NotImplementedError

    def __call__(self, *input):
        return self.forward(*input)

    def backward(self, gradwrtoutput):
        raise NotImplementedError

    @property
    def param(self):
        return []

    @property
    def grad(self):
        return []

    def load_param(self, *params):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"
