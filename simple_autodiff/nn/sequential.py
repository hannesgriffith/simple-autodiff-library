# from simple_autodiff.nn.layers.losses import SoftmaxCrossEntropyLossFromLogits
# from simple_autodiff.nn.layers.activations import ReLU, Softmax
# from simple_autodiff.nn.layers.mlp import Linear


class Sequential:
    """Use to stack Ops, Layers or other Sequential objects and then call them
    sequentially.
    """
    def __init__(self, *items):
        self.items = items
    
    def __call__(self, x):

        for item in self.items:
            x = item(x)

        return x


# base_model = Sequential(
#         Linear(8, 32),
#         ReLU(),
#         Linear(32, 10)
#     )

# def build_model(base_model, train=False):
#     out_activation = SoftmaxCrossEntropyLossFromLogits if train else Softmax
#     return Sequential(base_model, out_activation())