class Layer:
    """Base class for layers. Layers can be combined to be added to a
    sequential object and run sequentially. Layers represent some kind of 
    useful function in particular for deep learning, perhaps combining several
    ops, and handles setting up the associated trainable parameters.
    """
    def __call__(self, *input_tensors, **kwargs):
        if not hasattr(self, 'forward'):
            raise AttributeError("All Layers require a forward method.")

        self.forward(*input_tensors, **kwargs)

class AsLayer(Layer):
    """Wrapper for functions using ops, so they can be used in Sequential"""
    def __init__(self, func):
        self.func = func

    def forward(self, *input_tensors, **kwargs):
        return self.func(*input_tensors, **kwargs)