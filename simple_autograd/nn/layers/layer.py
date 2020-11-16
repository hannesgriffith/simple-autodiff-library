class Layer:
    def __call__(self, *input_tensors, **kwargs):
        self.forward(*input_tensors, **kwargs)

class AsLayer(Layer):
    """Wrapper for functions using ops, so they can be used in Sequential"""
    def __init__(self, func):
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)