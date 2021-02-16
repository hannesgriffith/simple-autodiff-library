import simple_autodiff.nn.functional as F
from simple_autodiff.ops.losses import SoftmaxCrossEntropyFromLogits

from simple_autodiff.nn.layers.layer import AsLayer


class MSELoss(AsLayer):
    """Mean square error from functional."""
    def __init__(self):
        super().__init__(F.mean_squared_error)


class SoftmaxCrossEntropyLossFromLogits(AsLayer):
    """Compounded function, first softmax on logits as input, then the
    cross-entropy loss on the resulting probabilities.
    """
    def __init__(self):
        super().__init__(SoftmaxCrossEntropyFromLogits)