import numpy as np

from simple_autodiff.nn.layers import Layer, AsLayer
import simple_autodiff.nn.functional as F
from simple_autodiff import ops


class ReLU(AsLayer):
    """Layer: Element-wise ReLU of tensor."""
    def __init__(self):
        super().__init__(ops.ReLU)


class Sigmoid(AsLayer):
    """Layer: Element-wise sigmoid operation of tensor."""
    def __init__(self):
        super().__init__(ops.Sigmoid)


class Softmax(AsLayer):
    """Layer: Softmax along axis 1 of tensor of shape [N, C]."""
    def __init__(self):
        super().__init__(ops.Softmax)


class Tanh(AsLayer):
    pass


class LeakyReLU(AsLayer):
    pass


class PReLU(Layer):
    pass


class ELU(AsLayer):
    pass


class Swish(Layer):
    pass