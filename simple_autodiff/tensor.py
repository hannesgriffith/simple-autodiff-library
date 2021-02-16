import numpy as np

from simple_autodiff import ops
from simple_autodiff.autodiff import add_tensor_to_registry
from simple_autodiff.autodiff import backward, reset_graph, visualise_graph


class Tensor:
    """Tensor base class. All intermediate tensors in the computational graph
    will be of this class and *will not* be updated by the optimiser.
    """
    tensor_count = 0

    def __init__(self, data, name=None, input_tensor=False,
                    param_tensor=False):
        self.data = data.astype(np.float32)
        self.grad = np.zeros_like(data)
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
        """Calling backward will finalise the computational graph, calculate
        grads for all the tensors in the graph with backprop and finally clean
        the graph and retain any parameter tensors in the tensor registry. To
        recalculate grads, get a result or visualise the graph you will need
        to build it again. If you are training remember to step the optimiser
        with these new grads.
        """
        backward(self.name)

    def result(self):
        """Calling result will return the data value for this tensor as an
        output. After returning the value the computational graph is finalised
        and cleaned. For further results build the computational graph again.
        """
        reset_graph()
        return self.data

    def visualise(self):
        """Plots visualisation of the computational graph. Calling visualise
        will add edges to the graph, plot it and then clean it up. To continue
        training or get a result you will need to build the graph again.
        """
        visualise_graph()

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name

    def __str__(self):
        return str(self.name)

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, other_tensor):
        return ops.Add(self, other_tensor)

    def __sub__(self, other_tensor):
        return ops.Subtract(self, other_tensor)

    def __mul__(self, other_tensor):
        return ops.MatMul(self, other_tensor)

    def __pow__(self, scalar_exponent):
        return ops.Power(self, scalar_exponent=scalar_exponent)

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

    def unsqueeze(self, axis=0):
        return ops.Unsqueeze(self, axis=axis)

    def repeat(self, axis=0, n=1):
        return ops.Repeat(self, axis=axis, n=n)

    def flatten(self):
        return ops.Flatten(self)


class InputTensor(Tensor):
    """Tensor used for Inputs; defined before the computational graph is
    built and that *will not* be updated by the optimiser.
    """
    def __init__(self, *args, input_tensor=True, **kwargs):
        super().__init__(
            *args,
            name="Input",
            input_tensor=input_tensor,
            **kwargs
            )
        add_tensor_to_registry(self)


class ParameterTensor(Tensor):
    """Tensor used for Parameters; defined before the computational graph is
    built and that *will* be updated by the optimiser.
    """
    def __init__(self, *args, param_tensor=True, **kwargs):
        super().__init__(
            *args,
            name="Parameter",
            param_tensor=param_tensor,
            **kwargs
            )
        add_tensor_to_registry(self)