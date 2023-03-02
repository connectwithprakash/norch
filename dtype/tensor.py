import numpy as np


class Tensor(np.ndarray):
    def __new__(cls, shape, init_method='normal'):
        parameter = cls.init_weights(
            shape=shape, init_method=init_method).view(cls)
        parameter.gradient = np.zeros(shape)
        return parameter

    @property
    def grad(self):
        return self.gradient

    @grad.setter
    def grad(self, value):
        self.gradient = value

    @staticmethod
    def init_weights(shape, init_method):
        if init_method == 'zero':
            return np.zeros(shape)
        elif init_method == 'normal':
            return np.random.normal(size=shape)
        else:
            raise NotImplementedError

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.gradient = getattr(obj, 'gradient', None)
