import numpy as np

from simple_autodiff.ops.tensor_op import TensorOp


class ReLU(TensorOp):
    """Element-wise ReLU of tensor: max(0, t)"""
    def _forward(self, tensor_data):
        return np.maximum(0, tensor_data), tensor_data

    def _backward(self, upstream_grad, tensor_data):
        upstream_grad[tensor_data < 0] = 0
        return (upstream_grad,)


class Sigmoid(TensorOp):
    """Element-wise sigmoid operation of tensor: 1 / (1 + e^-t)"""
    def _forward(self, tensor_data):
        out = 1 / (1 + np.exp(-tensor_data))
        return out, out

    def _backward(self, upstream_grad, out):
        grad = out * (1 - out) * upstream_grad
        return (grad,)


class Softmax(TensorOp):
    """Softmax over second dimension for data of shape
    [n examples, n classes]: e^xi / sum(e^xi), for i in axis.
    """
    def _forward(self, tensor_data):
        exp_x = np.exp(tensor_data)
        softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        cache = (tensor_data.shape[1], softmax)
        return softmax, cache

    def _backward(self, upstream_grad, cache):
        C, softmax = cache
        ones = np.eye(C, dtype=np.float32).reshape(1, C, C)
        softmax_repeated = np.repeat(softmax[:, :, np.newaxis], C, axis=2)
        grad = softmax * np.sum(ones - softmax_repeated, axis=1)
        return (grad * upstream_grad,)


class Tanh(TensorOp):
    pass


class LeakyRelu(TensorOp):
    pass


class ELU(TensorOp):
    pass