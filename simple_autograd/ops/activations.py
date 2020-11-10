import numpy as np

from simple_autograd.ops.tensor_op import TensorOp


class ReLU(TensorOp):
    """Element-wise ReLU of tensor: max(0, t)"""
    def _forward(self, tensor_data):
        mask = tensor_data < 0 
        tensor_data[mask] = 0
        return tensor_data, mask

    def _backward(self, upstream_grad, mask):
        upstream_grad[mask] = 0
        return (upstream_grad,)


class Sigmoid(TensorOp):
    """Element-wise sigmoid operation of tensor: 1 / (1 + e^-t)"""
    def _forward(self, tensor_data):
        out = 1 / (1 + np.exp(-tensor_data))
        return out, out

    def _backward(self, upstream_grad, out):
        output = out * (1 - out) * upstream_grad
        return (output,)


class Softmax(TensorOp):
    pass


class Tanh(TensorOp):
    pass