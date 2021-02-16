from simple_autodiff import ops

def mean_squared_error(output, target):
    diff = (output - target) ** 2.
    return diff.mean()

def cross_entropy_loss(output, target):
    pass

def swish(x):
    pass