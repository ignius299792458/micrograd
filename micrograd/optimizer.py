from micrograd.value import ScalarValue


def optimize_params(learning_rate: float, params: list[ScalarValue]):
    for param in params:
        param.data -= learning_rate * param.grad
