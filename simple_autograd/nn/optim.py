import numpy as np

from simple_autograd.autograd import get_all_parameter_tensors


class SGDOptimiser:
    """Base class for SGD optimisers. To implement your own SGD optimiser just
    sub-class this class and implement _calculate_update method."""
    def __init__(self, grad_mod=None):
        self.param_tensors = get_all_parameter_tensors()
        self.grad_mod = grad_mod if grad_mod is not None else lambda x: x

    def _calculate_update(self, grad, idx):
        raise NotImplemented("Must implement to subclass SGDOptimiser.")

    def step(self):
        for idx, param_tensor in enumerate(self.param_tensors):
            grad = self.grad_mod(param_tensor.grads)
            update = self._calculate_update(grad, idx)
            param_tensor.data -= update


class SGD(SGDOptimiser):
    """Vanilla SGD + momentum"""
    def __init__(self, lr=0.0001, momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.momentum = momentum
        self.history = [np.zeros_like(t) for t in self.param_tensors]

    def _calculate_update(self, grad, idx):
        update = self.momentum * self.history[idx] + self.lr * grad
        self.history[idx] = update.copy()
        return update