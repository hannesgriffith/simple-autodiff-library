from pprint import pprint

import numpy as np

from simple_autograd.graph import ComputationalGraph, GraphDefinitionError


class AutoGradManager:
    """Holds the computational graph and references to all tensors. Executes
    backpropogation on the computational graph."""
    def __init__(self):
        self.graph = ComputationalGraph()
        self.tensor_registry = TensorRegistry()
        self.saved_parameters = {}

    def add_op_to_graph(self, op, input_tensors, output_tensor):
        self.tensor_registry.add_tensor(output_tensor)
        self.graph.add_op(op, input_tensors, output_tensor)

    def backward(self, start_tensor_name):
        self.tensor_registry.clear_grads()
        start_tensor = self.tensor_registry[start_tensor_name]
        start_grad = np.ones_like(start_tensor.data)
        self.tensor_registry[start_tensor_name].grad = start_grad

        for backprop_node in self.graph.backprop_iterator(start_tensor_name):
            op = backprop_node["op"]
            input_tensor_names = backprop_node["in"]
            output_tensor_name = backprop_node["out"]

            grads = op.backward(self.tensor_registry[output_tensor_name].grad)

            for input_tensor_name, grad in zip(input_tensor_names, grads):
                self.tensor_registry[input_tensor_name].grad += grad
                # print(input_tensor_name, grad)

        self.clean_up()

    def clean_up(self):
        self.graph.clear()

        params_to_carry_over = self.tensor_registry.parameter_tensors
        self.tensor_registry = TensorRegistry()

        for param_tensor in params_to_carry_over:
            self.tensor_registry.add_tensor(param_tensor)


class TensorRegistry(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op_already_added_to_graph = False

    def add_tensor(self, tensor):
        if tensor.input_tensor or tensor.param_tensor:
            if self.op_already_added_to_graph:
                raise GraphDefinitionError(f"Must add all input and param \
                    tensors to the graph before defining any ops. Cannot \
                    add: {tensor} of type {type(tensor)}.")
        else:
            self.op_already_added_to_graph = True

        self[tensor.name] = tensor

    def clear_grads(self):
        for tensor in self.values():
            tensor.grad = np.zeros_like(tensor.data)

    @property
    def parameter_tensors(self):
        return [t for t in self.values() if t.param_tensor]


def new_session():
    global GLOBAL_MANAGER
    GLOBAL_MANAGER = AutoGradManager()

def add_op_to_graph(op, input_tensors, output_tensor):
    return GLOBAL_MANAGER.add_op_to_graph(op, input_tensors, output_tensor)

def add_tensor_to_registry(tensor):
    return GLOBAL_MANAGER.tensor_registry.add_tensor(tensor)

def backward(tensor_name):
    return GLOBAL_MANAGER.backward(tensor_name)

def get_all_parameter_tensors():
    return GLOBAL_MANAGER.tensor_registry.parameter_tensors

def print_tensor_registry():
    pprint(GLOBAL_MANAGER.tensor_registry)

def draw_computational_graph():
    GLOBAL_MANAGER.graph.visualise_nice()

def _debug_get_tensor_registry():
    return GLOBAL_MANAGER.tensor_registry

def _debug_get_computational_graph():
    return GLOBAL_MANAGER.graph

def _debug_finalise_graph():
    GLOBAL_MANAGER.graph._finalise_graph()

def _debug_clean_up():
    GLOBAL_MANAGER.clean_up()

new_session()