import numpy as np

from simple_autograd.nn.functional import mean_squared_error


class AsLayer:
    """Wrapper for functions using ops, so they can be used in Sequential"""
    def __init__(self, func):
        self.func = func
    
    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class MSELoss:
    """Mean square error from functional."""
    def __init__(self):
        super().__init__(F.mean_squared_error)