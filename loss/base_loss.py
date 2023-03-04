from abc import ABC


class BaseLoss(ABC):
    def forward(self, **kwargs):
        raise NotImplementedError

    def backward(self, **kwargs):
        raise NotImplementedError

    def __call__(self, *input):
        return self.forward(*input)
