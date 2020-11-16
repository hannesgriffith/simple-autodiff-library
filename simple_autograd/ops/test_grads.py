from functools import partial

import numpy as np
from numpy.random import rand

from simple_autograd.ops import activations
from simple_autograd.ops import losses
from simple_autograd.ops import maths
from simple_autograd.ops import shape

def calculate_numerical_grad(f, xs, h=1e-5):
    # Based on https://cs231n.github.io/optimization-1/#numerical
    grads = []
    for x in xs:
        if isinstance(x, np.ndarray):
            grads.append(np.zeros(x.shape))
        else:
            grads.append(None)

    for i, grad in enumerate(grads):
        if grad is None:
            continue

        it = np.nditer(xs[i], flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:

            ix = it.multi_index
            old_value = xs[i][ix]
            xs[i][ix] = old_value + h
            fxhp, _ = f(*xs)
            xs[i][ix] = old_value - h
            fxhm, _ = f(*xs)
            xs[i][ix] = old_value

            grad[ix] = (fxhp.sum() - fxhm.sum()) / (2. * h)
            it.iternext()

    return filter(lambda x: x is not None, grads)

def calculate_analytical_grad(forward, backward, inputs):
    output, cache = forward(*inputs)
    return backward(np.ones_like(output), cache)

def test_op(Op, inputs):
    forward = partial(Op._forward, None)
    backward = partial(Op._backward, None)

    numerical_grads = calculate_numerical_grad(forward, inputs)
    analytical_grads = calculate_analytical_grad(forward, backward, inputs)

    total_diff = 0.
    for ng, ag in zip(numerical_grads, analytical_grads):
        total_diff += np.abs(ng - ag).mean()

    return total_diff / float(len(analytical_grads))

def test_ops(ops_to_test, threshold=1e-6):
    num_ops = len(ops_to_test)
    num_passed = 0

    for op, num_inputs in ops_to_test:
        diff = test_op(op, num_inputs)

        op_name = op.__name__
        pass_ = diff < threshold
        status = "PASS" if pass_ else "FAIL"
        print(f"{status}: \t{op_name},    \t{diff:.3E}")

        if pass_:
            num_passed += 1
    
    print(f"{num_passed} / {num_ops} Passed")

ops_to_test = [
    (activations.ReLU, [rand(100, 100)]),
    (activations.Sigmoid, [rand(100, 100)]),
    (activations.Softmax, [rand(100, 10)]),
    (maths.Add, [rand(100, 100), rand(100, 100)]),
    (maths.Subtract, [rand(100, 100), rand(100, 100)]),
    (maths.MatMul, [rand(100, 100), rand(100, 100)]),
    (maths.Power, [rand(100, 100), 2]),
    (maths.Sum, [rand(100, 100)]),
    (maths.Mean, [rand(1000, 100)])
]

test_ops(ops_to_test)