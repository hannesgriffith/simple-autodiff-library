import numpy as np

from simple_autodiff.ops.tensor_op import TensorOp


class SoftmaxCrossEntropyFromLogits(TensorOp):
    """Combined softmax and cross-entropy loss on logits."""
    def _forward(self, logits, labels, axis=1):
        assert logits.shape == labels.shape
        labels = labels.astype(np.float32)
        exp_sum = np.sum(np.exp(logits), axis=axis, keepdims=True)
        softmax = np.exp(logits) / exp_sum
        out = -labels * (logits - np.log(exp_sum))
        cache = (softmax, labels)
        return out, cache

    def _backward(self, upstream_grad, cache, **kwargs):
        softmax, labels = cache
        grads = softmax - labels
        return (grads * upstream_grad,)