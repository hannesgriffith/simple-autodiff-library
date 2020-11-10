import numpy as np

from simple_autograd import ops
from simple_autograd.autograd import backward, add_tensor_to_registry


class Tensor:
    """Tensor base class. All intermediate tensors in the computational graph
    will be of this class and *will not* be updated by the optimiser."""
    tensor_count = 0

    def __init__(self, data, name=None, input_tensor=False,
                    param_tensor=False):
        self.data = data.astype(np.float32)
        self.grads = np.zeros_like(data)
        self.input_tensor = input_tensor
        self.param_tensor = param_tensor

        unique_id = self.get_unique_id()
        name = name if name is not None else self.__class__.__name__ 
        self.name = f"Tensor:{name}:{unique_id}"

    @classmethod
    def get_unique_id(cls):
        cls.tensor_count += 1
        return cls.tensor_count

    def backward(self):
        backward(self.name)

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name

    def __str__(self):
        return str(self.name)

    def __add__(self, other_tensor):
        return ops.Add(self, other_tensor)

    def __sub__(self, other_tensor):
        return ops.Subtract(self, other_tensor)

    def __mul__(self, other_tensor):
        return ops.MatMul(self, other_tensor)

    def __pow__(self, other_scalar):
        return ops.Power(self, other_scalar)

    def transpose(self):
        return ops.Transpose(self)

    @property
    def T(self):
        return self.transpose()

    def sum(self):
        return ops.Sum(self)

    def mean(self):
        return ops.Mean(self)

    def squeeze(self):
        return ops.Squeeze(self)


class InputTensor(Tensor):
    """Tensor used for Inputs; defined before the computational graph is \
    built and that *will not* be updated by the optimiser."""
    def __init__(self, *args, input_tensor=True, **kwargs):
        super().__init__(
            *args,
            name="Input",
            input_tensor=input_tensor,
            **kwargs
            )
        add_tensor_to_registry(self)


class ParameterTensor(Tensor):
    """Tensor used for Parameters; defined before the computational graph is \
    built and that *will* be updated by the optimiser."""
    def __init__(self, *args, param_tensor=True, **kwargs):
        super().__init__(
            *args,
            name="Parameter",
            param_tensor=param_tensor,
            **kwargs
            )
        add_tensor_to_registry(self)