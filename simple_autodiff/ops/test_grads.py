from functools import partial

import numpy as np
from numpy.random import rand

from simple_autodiff.ops import activations
from simple_autodiff.ops import losses
from simple_autodiff.ops import maths
# from simple_autodiff.ops import shape

def normalise(X):
    return X / np.sum(X, axis=1, keepdims=True)

def one_hot(y, num_classes):
    one_hot_y = np.zeros((y.shape[0], num_classes))
    idxs = np.where(y == np.max(y, axis=1, keepdims=True))
    one_hot_y[idxs[0], idxs[1]] = 1.0
    return one_hot_y

def calculate_numerical_grad(f, xs, h=1e-5, **input_kwargs):
    # Based on https://cs231n.github.io/optimization-1/#numerical
    grads = [np.zeros(x.shape) for x in xs]

    for i, grad in enumerate(grads):

        it = np.nditer(xs[i], flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:

            ix = it.multi_index
            old_value = xs[i][ix]
            xs[i][ix] = old_value + h
            fxhp, _ = f(*xs, **input_kwargs)
            xs[i][ix] = old_value - h
            fxhm, _ = f(*xs, **input_kwargs)
            xs[i][ix] = old_value

            grad[ix] = (fxhp.sum() - fxhm.sum()) / (2. * h)
            it.iternext()

    return filter(lambda x: x is not None, grads)

def calculate_analytical_grad(forward, backward, input_tensors, input_kwargs):
    output, cache = forward(*input_tensors, **input_kwargs)
    return backward(np.ones_like(output), cache, **input_kwargs)

def test_op(Op, input_tensors, input_kwargs):
    forward = partial(Op._forward, None)
    backward = partial(Op._backward, None)

    numerical_grads = calculate_numerical_grad(
        forward, input_tensors, **input_kwargs)
    analytical_grads = calculate_analytical_grad(
        forward, backward, input_tensors, input_kwargs)

    total_diff = 0.
    for numerical_grad, analytical_grad in zip(numerical_grads, 
            analytical_grads):
        total_diff += np.abs(numerical_grad - analytical_grad).mean()

    return total_diff / float(len(analytical_grads))

def test_ops(ops_to_test, threshold=1e-6):
    num_ops = len(ops_to_test)
    num_passed = 0

    for op, input_tensors, input_kwargs in ops_to_test:
        diff = test_op(op, input_tensors, input_kwargs)

        op_name = op.__name__
        pass_ = diff < threshold
        status = "PASS" if pass_ else "FAIL"
        print(f"{status}: \t{op_name},    \t{diff:.3E}")

        if pass_:
            num_passed += 1
    
    print(f"{num_passed} / {num_ops} Passed")

# To add op to test, add a tuple below including:
# (name of op to test, list of example input tensor data, dict of kwargs)
ops_to_test = [
    (activations.ReLU, [rand(100, 100)], {}),
    (activations.Sigmoid, [rand(100, 100)], {}),
    (activations.Softmax, [rand(100, 10)], {}),
    (maths.Add, [rand(100, 100), rand(100, 100)], {}),
    (maths.Subtract, [rand(100, 100), rand(100, 100)], {}),
    (maths.MatMul, [rand(100, 100), rand(100, 100)], {}),
    (maths.Power, [rand(100, 100)], {"scalar_exponent": rand(1) * 10.}),
    (maths.Sum, [rand(100, 100)], {}),
    (maths.Mean, [rand(100, 100)], {}),
    (maths.Exp, [rand(100, 100)], {}),
    (losses.SoftmaxCrossEntropyFromLogits, 
        [100 * rand(100, 10), one_hot(rand(100, 10), 10)], {}), # one-hot labels
    (losses.SoftmaxCrossEntropyFromLogits, 
        [100 * rand(100, 10), normalise(rand(100, 10))], {}), # soft labels
]

test_ops(ops_to_test)