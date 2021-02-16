from simple_autodiff.tensor import InputTensor, ParameterTensor
from simple_autodiff.nn.initialisers import normal_initialiser
from simple_autodiff.nn.initialisers import zeros_initialiser

from simple_autodiff.nn.layers.layer import Layer


class Linear(Layer):
    """Linear layer, as would be used in an MLP. Weight matrix of shape
    [units in, units out]. Option to add bias to result, of size [units
    out]. Initialisers can be specified.
    """
    def __init__(self, channels_in, channels_out, use_bias=True,
            weights_initialiser=normal_initialiser,
            bias_initialiser=zeros_initialiser):

        start_weights = weights_initialiser(channels_in, channels_out)
        self.W = ParameterTensor(start_weights)

        bias_type = ParameterTensor if use_bias else InputTensor
        start_bias = bias_initialiser((channels_out,))
        self.b = bias_type(start_bias)
    
    def forward(self, input_tensor):
        assert input_tensor.data.ndim == 2
        return input_tensor * self.W + self.b