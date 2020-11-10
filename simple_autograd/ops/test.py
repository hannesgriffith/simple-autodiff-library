import numpy as np

from simple_autograd.ops import activations
from simple_autograd.ops import losses
from simple_autograd.ops import maths
from simple_autograd.ops import shape

def test_grad(Op, input_shapes, delta=1e-8):
    inputs = []
    # inputs_ = [np.random.normal(size=shape) for shape in input_shapes]

    for input_shape in input_shapes:
        if isinstance(input_shape, tuple):
            input_ = np.random.normal(size=input_shape)
        else:
            input_ = input_shape # just take input directly

        inputs.append(input_)

    inputs_plus = [i + delta for i in inputs]
    inputs_minus = [i - delta for i in inputs]

    output_plus, _ = Op._forward(None, *inputs_plus)
    output_minus, _ = Op._forward(None, *inputs_minus)
    finite_diff_grads = (output_plus - output_minus) / (2. * delta)

    output, cache = Op._forward(None, *inputs)
    op_grads = Op._backward(None, np.ones_like(output), cache)

    total_diff = 0.
    for fd, op in zip(finite_diff_grads, op_grads):
        total_diff += abs(fd.mean() - op.mean())

    return total_diff / float(len(op_grads))

def test_ops(ops_to_test, threshold=1e-8):
    num_ops = len(ops_to_test)
    num_passed = 0
    
    for op_to_test, num_inputs in ops_to_test:
        diff = test_grad(op_to_test, num_inputs)

        op_name = op_to_test.__name__
        pass_ = diff < threshold
        status = "PASS" if pass_ else "FAIL"
        print(f"{status}: \t{op_name},    \t{diff:.3E}")

        if pass_:
            num_passed += 1
    
    print(f"{num_passed} / {num_ops} Passed")

ops_to_test = [
    (activations.ReLU, [(3, 3, 3, 3)]),
    (activations.Sigmoid, [(3, 3, 3, 3)]),
    (maths.Add, [(3, 3, 3, 3), (3, 3, 3, 3)]),
    (maths.Subtract, [(3, 3, 3, 3), (3, 3, 3, 3)]),
    (maths.MatMul, [(3, 3, 3, 3), (3, 3, 3, 3)]),
    (maths.Power, [(3, 3, 3, 3), 2])
]

test_ops(ops_to_test)