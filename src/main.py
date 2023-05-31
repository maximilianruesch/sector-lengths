import jax.numpy as np
import numpy
from jax import jit
from qgeo import GHZ, make_dm, ket
from about_time import about_time

"""
from sympy import symbols, lambdify, Sum, srepr, Identity, Array
from sympy.abc import i
from sympy.physics.quantum import Ket, TensorProduct, Dagger, qapply, Bra
from sympy.physics.quantum.density import Density
from sympy.physics.quantum.trace import Tr
"""

STATE_DIMENSION = 15
TARGET_DIMENSION = 4


def run():
    state = GHZ(STATE_DIMENSION)
    trace_subsystems = tuple([j for j in range(TARGET_DIMENSION, STATE_DIMENSION)])

    f = jit(calc_partial_trace, static_argnums=(1, 2,))

    t1 = about_time(block_partial_trace, f, state, trace_subsystems, STATE_DIMENSION)
    print(f"First pass: {t1.duration} seconds")

    t2 = about_time(block_partial_trace, f, state, trace_subsystems, STATE_DIMENSION)
    print(f"Second pass: {t2.duration} seconds")

    state = ket("0" * STATE_DIMENSION, 2)
    t3 = about_time(block_partial_trace, f, state, trace_subsystems, STATE_DIMENSION)
    print(f"Third pass: {t3.duration} seconds")

    t4 = about_time(block_partial_trace, f, state, trace_subsystems, STATE_DIMENSION)
    print(f"Fourth pass: {t4.duration} seconds")

    return

def block_partial_trace(partial_trace_callable, *args):
    result = partial_trace_callable(*args)
    result.block_until_ready()

    return result


def calc_partial_trace(rho, trace_over, n):  # former ptrace_dim
    """ partial trace over subsystems specified in trace_over for arbitrary
        n-quDit systems (also of heteregeneous dimensions)
        e.g. ptrace(rho_ABC, [1]) = rhoA_C
        if pad is True, state will be tensored by identity to its original size
    args:       rho :   ndarray
                trace_over: list of subsystems to trace out, counting starts at 0
                d :         int, dimension (default is 2 for qubits), or list of
                            dimensions (for heterogeneous systems)
                pad :    bool, pad by identity if True
    returns:    rho_tr :    ndarray
    """
    rho = make_dm(rho)
    if len(trace_over) == 0:  # if no subsystem gets trace out.
        return rho

    # create list of dimension [d1, d1, d2, d2, ...]
    ddims = [2] * 2 * n

    trace_over = numpy.sort(numpy.array(trace_over))

    # reshaped matrix, two indices per local system
    rho2 = rho.reshape(ddims)

    idx = numpy.arange(2 * n)
    for i in trace_over:
        idx[n + i] = i

    trace = np.einsum(rho2, idx.tolist())

    # calculate new number of particles and new dimensions and reshape traced out state
    d_new = 2 ** (n - len(trace_over))

    return trace.reshape(d_new, d_new)


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

"""

def partial_trace(state, subsystem, n_subsystems):
    return TensorProduct(Bra(0), Identity(1)) * state * TensorProduct(Ket(0), Identity(1))\
        + TensorProduct(Bra(1), Identity(1)) * state * TensorProduct(Ket(1), Identity(1))

    start_time = time.time()
    state_parameters = symbols('c0:%d' % (2 ** STATE_DIMENSION))
    state = reduce(lambda a, b: a + b, [state_parameters[k] * Ket(k) for k in range(0, 2 ** STATE_DIMENSION)])
    state = (state * Dagger(state)).expand()

    #print(state)

    partial_trace_state = partial_trace(state, 1, 1)

    #print(partial_trace_state.expand())
    print(f"Took {time.time() - start_time} seconds!")

    return
"""
