import numpy as np

from simple_autograd.ops.tensor_op import TensorOp


class Transpose(TensorOp):
    """Transpose of (2D) matrix."""
    def _forward(self, tensor_data):
        return tensor_data.T, ()

    def _backward(self, upstream_grad, _):
        return (upstream_grad.T,)


class Squeeze(TensorOp):
    """Remove any size 1 dimensions."""
    def _forward(self, tensor_data):
        idxs = np.where(np.array(tensor_data.shape) == 1)[0].tolist()
        return np.squeeze(tensor_data), idxs

    def _backward(self, upstream_grad, idxs):
        grad = np.expand_dims(upstream_grad, tuple(idxs))
        return (grad,)


class UnSqueeze(TensorOp):
    """Add a size one dimension at given location."""
    def _forward(self, tensor_data, axis=0):
        return np.expand_dims(tensor_data, axis), ()

    def _backward(self, upstream_grad, _, axis=0):
        grad = np.squeeze(upstream_grad, axis=axis)
        return (grad,)


class Repeat(TensorOp):
    """Copy a tensor n times along an existing axis."""
    def _forward(self, tensor_data, axis=0, n=1):
        out = np.concatenate([tensor_data] * n, axis=axis)
        return out, ()

    def _backward(self, upstream_grad, _, axis=0):
        grad = np.mean(upstream_grad, axis=axis, keepdims=True)
        return (grad,)


class Flatten(TensorOp):
    """Flattens a tensor."""
    def _forward(self, tensor_data):
        original_shape = tensor_data.shape
        return tensor_data.flatten(), original_shape
    
    def _backward(self, upstream_grad, original_shape):
        grad = upstream_grad.reshape(original_shape)
        return (grad,)