class Sequential:
    def __init__(self, *layers):
        self.layers = layers

        for layer in self.layers:
            if not hasattr(layer, 'forward'):
                raise AttributeError("All layers require forward method")
    
    def __call__(self, x):

        for layer in self.layers:
            x = layer.forward(x)

        return x