import numpy as np

def zeros_initialiser(tensor_shape):
    return np.zeros(shape=tensor_shape)

def uniform_initialiser(tensor_shape):
    return np.random.uniform(size=tensor_shape)

def normal_initialiser(tensor_shape, mean=0.0, std=1.0):
    return np.random.normal(loc=mean, scale=std, size=tensor_shape)

def xavier_initialiser(tensor_shape, num_inputs, num_outputs):
    n_avg = (num_inputs + num_outputs) / 2.
    return normal_initialiser(tensor_shape, std=(1. / n_avg) ** 0.5)

def he_initialiser(tensor_shape, num_inputs, num_outputs):
    n_avg = (num_inputs + num_outputs) / 2.
    return normal_initialiser(tensor_shape, std=(2. / n_avg) ** 0.5)