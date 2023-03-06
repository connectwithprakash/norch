from .mse import MSELoss
from .hinge import HingeLoss
from .cross_entropy import CrossEntropyLoss, CrossEntropyWithLogitsLoss

__all__ = ['MSELoss', 'HingeLoss',
           'CrossEntropyLoss', 'CrossEntropyWithLogitsLoss']
