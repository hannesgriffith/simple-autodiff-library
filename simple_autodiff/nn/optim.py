import numpy as np

from simple_autodiff.autodiff import get_all_parameter_tensors


class SGDOptimiser:
    """Base class for SGD optimisers. To implement your own SGD optimiser just
    sub-class this class and implement _calculate_update method."""
    def __init__(self, grad_mod=None):
        self.param_tensors = get_all_parameter_tensors()
        self.grad_mod = grad_mod if grad_mod is not None else lambda x: x

    def step(self):
        for param_idx, param_tensor in enumerate(self.param_tensors):
            grad = self.grad_mod(param_tensor.grad)
            update = self._calculate_update(grad, param_idx)
            param_tensor.data += update
            # print(param_tensor.data, grad, update)

    def _calculate_update(self, grad, param_idx):
        raise NotImplemented("Must implement to subclass SGDOptimiser.")


class SGD(SGDOptimiser):
    """Vanilla SGD + momentum"""
    def __init__(self, lr=0.0001, momentum=0.9, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.momentum = momentum
        self.history = [np.zeros_like(t) for t in self.param_tensors]

    def _calculate_update(self, grad, param_idx):
        update = self.momentum * self.history[param_idx] - self.lr * grad
        self.history[param_idx] = update.copy()
        return update