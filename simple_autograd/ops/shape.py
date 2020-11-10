import numpy as np

from simple_autograd.ops.tensor_op import TensorOp


class Transpose(TensorOp):
    """Transpose of (2D) matrix."""
    def _forward(self, tensor_data):
        return tensor_data.T, ()

    def _backward(self, upstream_grad):
        return upstream_grad.T


class Squeeze(TensorOp):
    """Remove any size 1 dimensions."""
    def _forward(self, tensor_data):
        idxs = np.where(np.array(tensor_data.shape) == 1)[0].tolist()
        return np.squeeze(tensor_data), idxs

    def _backward(self, upstream_grad, idxs):
        return np.expand_dims(upstream_grad, tuple(idxs))