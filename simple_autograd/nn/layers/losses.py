import simple_autograd.nn.functional as F

from simple_autograd.nn.layers import AsLayer


class MSELoss(AsLayer):
    """Mean square error from functional."""
    def __init__(self):
        super().__init__(F.mean_squared_error)