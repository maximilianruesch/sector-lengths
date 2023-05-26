import math
from functools import partial, reduce

import jax.numpy as np
from jax import jit, grad, vmap
from qgeo import ptrace, sector_len_f, GHZ, make_dm
from sympy import symbols, lambdify, Sum, srepr, Identity
from sympy.abc import i
from sympy.physics.quantum import Ket, TensorProduct, Dagger, qapply, Bra
from sympy.physics.quantum.density import Density
from sympy.physics.quantum.trace import Tr

STATE_DIMENSION = 2
TARGET_DIMENSION = 4


def run():
    state_parameters = symbols('c0:%d' % (2 ** STATE_DIMENSION))
    state = reduce(lambda a, b: a + b, [state_parameters[k] * Ket(k) for k in range(0, 2 ** STATE_DIMENSION)])
    state = (state * Dagger(state)).expand()

    print(state)

    partial_trace_state = partial_trace(state, 1, 1)

    print(partial_trace_state)

    return

    """
        state = state\
            .subs(state_parameters[0], 1 / math.sqrt(2))\
            .subs(state_parameters[1], 0)\
            .subs(state_parameters[2], 0) \
            .subs(state_parameters[3], 1 / math.sqrt(2))
        """

    trace = Tr(A)

    f = lambdify(A, trace, "numpy")
    print(f)
    state = make_dm(GHZ(STATE_DIMENSION))
    print(state)

    print(Tr(state, [1]))


def partial_trace(state, subsystem, n_subsystems):
    return TensorProduct(Bra(0), Identity) * state * TensorProduct(Ket(0), Identity) + TensorProduct(Bra(1), Identity) * state * TensorProduct(Ket(1), Identity)


"""
@partial(jit)
def calc_sector_lengths():
    state = GHZ(STATE_DIMENSION)
    trace_subsystems = [i for i in range(TARGET_DIMENSION, STATE_DIMENSION)]
    print(f"Tracing over {trace_subsystems}")

    traced_state = ptrace(state, trace_subsystems)
    sectors = sector_len_f(traced_state)

    print(sectors)
"""

if __name__ == '__main__':
    run()
