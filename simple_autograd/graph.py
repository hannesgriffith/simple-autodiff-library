from abc import ABCMeta, abstractmethod
from functools import partial

import networkx as nx
from networkx.algorithms.dag import descendants
import matplotlib.pyplot as plt


class GraphDefinitionError(Exception):
    """Error due to defining graph illegally."""
    pass


class ComputationalGraph(nx.DiGraph):
    """Stores relationship between ops and tensors during forward pass and
    works out order of execution for backprop."""

    def add_op(self, op, input_tensors, output_tensor):
        node = {
            "op": op,
            "in": [t.name for t in input_tensors],
            "out": output_tensor.name
        }
        self.add_node(op.name, **node)

    @staticmethod
    def _get_op_number(op_name):
        return op_name.split(":")[-1]

    def _after_in_graph(self, start_op_name, op_name):
        return self._get_op_number(op_name) > self._get_op_number(start_op_name)

    def _add_edges(self):
        ordered_ops = sorted(self.nodes, key=self._get_op_number)

        for start_op in ordered_ops:
            start_tensor = self.nodes[start_op]["out"]

            filter_func = partial(self._after_in_graph, start_op)
            subsequent_ops = filter(filter_func, self.nodes)

            for end_op in subsequent_ops:
                if start_tensor in self.nodes[end_op]["in"]:
                    self.add_edge(start_op, end_op)

    def _finalise_graph(self):
        self._add_edges()

    def _get_graph_without_attributes(self):
        graph_without_attributes = nx.DiGraph()
        graph_without_attributes.add_nodes_from(self.nodes)
        graph_without_attributes.add_edges_from(self.edges)
        return graph_without_attributes

    def _get_node_by_ouput_tensor(self, start_tensor_name):
        for node in self.nodes:
            if self.nodes[node]["out"] == start_tensor_name:
                return node

    def backprop_iterator(self, start_tensor_name):
        self._finalise_graph()
        start_node = self._get_node_by_ouput_tensor(start_tensor_name)
        backprop_graph = self._get_graph_without_attributes().reverse()
        contributing_nodes = descendants(backprop_graph, start_node)

        for node in nx.topological_sort(backprop_graph):
            if node in contributing_nodes:
                yield self.nodes[node]

    def visualise(self, graph=None):
        graph = graph if graph is not None else self
        nx.draw(graph, with_labels=True, font_weight='bold')
        plt.show()


class ComputationalGraphOp(metaclass=ABCMeta):
    @abstractmethod
    def forward(self):
        """
        Inputs: input tensors required for operation
        Output: returns new tensor/s dependent on op

        Calculate and return op result in forward pass. Store values required
        for backprop as well as references to input tensors to accumulate 
        gradients in during backprop.
        """
        pass

    @abstractmethod
    def backward(self):
        """
        Input: gradient calculated from all upstream ops in backprop graph
        Output: combined gradient for next op node in backprop graph 

        Calculate grad of input/s wrt output/s and add gradients to them.
        Return result * upstream grad to next op in backprop graph.
        """
        pass

    @classmethod
    def __subclasshook__(cls, C):
        if cls is ComputationalGraphOp:
            attributes = set(dir(C))
            if set(cls.__abstractmethods__) <= attributes:
                return True

        return NotImplemented