from simple_autograd import ops

def mean_squared_error(output, target):
    diff = (output - target) ** 2.
    return diff.mean()

def swish(x):
    pass