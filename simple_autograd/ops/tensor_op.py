from simple_autograd.tensor import Tensor
from simple_autograd.autograd import add_op_to_graph
from simple_autograd.graph import ComputationalGraphOp

def not_constant_scalar(t):
    return type(t) not in [int, float]


class TensorOp:
    """To create a new Op just subclass TensorOp and implement _forward and
    _backward methods. Inputs for _forward is the data from the input tensors
    for the operation. Return any values you would like to cache for backprop
    in a tuple after the output value. The input to _backward is the upstream
    gradient and the cache from the forward pass. Calculate the gradients for
    each tensor and return them. NOTE: only ops that produce a single output
    tensor are currently supported.

    IMPORTANT: return the grads for the input tensors in a tuple in the same
    order they are given as input in forward."""
    op_count = 0

    def __new__(cls, *input_tensors):
        tensorop_instance = super().__new__(cls)
        cls.__init__(tensorop_instance)

        if not isinstance(tensorop_instance, ComputationalGraphOp):
            raise TypeError(f"{cls.__class__.__name__} doesn't have correct \
                interface to add to computational graph.")

        output_tensor = tensorop_instance.forward(*input_tensors)
        input_tensors = [t for t in input_tensors if isinstance(t, Tensor)]

        if not isinstance(output_tensor, Tensor):
            raise TypeError(f"Only Tensor objects can be added to tensor \
                registry, cannot add: {output_tensor} of type \
                {type(output_tensor)}.")

        add_op_to_graph(tensorop_instance, input_tensors, output_tensor)

        return output_tensor

    def __init__(self):
        unique_id = self.get_unique_id()
        self.name = f"Op:{self.__class__.__name__}:{unique_id}"

    @staticmethod
    def get_unique_id():
        TensorOp.op_count += 1
        return TensorOp.op_count

    def __hash__(self):
        return hash(self.name)

    def forward(self, *input_tensors, **kwargs):
        input_tensor_data = []
        for tensor in input_tensors:
            if isinstance(tensor, Tensor):
                tensor_data = tensor.data.copy()
            elif isinstance(tensor, int) or isinstance(tensor, float):
                tensor_data = tensor
            else:
                raise TypeError(f"Illegal input type: {type(tensor)}")

            input_tensor_data.append(tensor_data)

        output_data, self.cache = self._forward(*input_tensor_data, **kwargs)
        return Tensor(output_data, name=self.__class__.__name__)

    def backward(self, upstream_grads):
        return self._backward(upstream_grads, self.cache)

    def _forward(self, *input_tensor_data):
        raise NotImplemented("Must implement _forward to subclass TensorOp.")

    def _backward(self, upstream_grads, cache):
        raise NotImplemented("Must implement _backward to subclass TensorOp.")