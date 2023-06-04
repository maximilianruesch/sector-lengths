from functools import partial

import jax
import jax.numpy as np
import jax_qgeo
import numpy
from jax import jit, grad, value_and_grad, make_jaxpr
from jax_qgeo import GHZ, make_dm, ket, number_of_qudits, purity
import qgeo
import jax_qgeo.qcode_enum as qce
from about_time import about_time
import itertools as itt  # for combinatorics, choice, permutations, etc

from sympy import symbols, lambdify

"""
from sympy import symbols, lambdify, Sum, srepr, Identity, Array
from sympy.abc import i
from sympy.physics.quantum import Ket, TensorProduct, Dagger, qapply, Bra
from sympy.physics.quantum.density import Density
from sympy.physics.quantum.trace import Tr
"""

STATE_DIMENSION = 4
TARGET_DIMENSION = 4


def run():
    print("Starting random state construct")
    with about_time() as t2:
        state = jax_qgeo.rand_pure_state(STATE_DIMENSION)
    print(f"Random state construct took {t2.duration} seconds")
    #print(f"Using state: {state}")
    #qgeo_state = qgeo.RCL(STATE_DIMENSION)

    #trace_subsystems = tuple([j for j in range(TARGET_DIMENSION, STATE_DIMENSION)])

    #f = jit(calc_partial_trace, static_argnums=(1, 2,))
    #f2 = jit(calc_sector_lengths)
    #traced_state = f(state, trace_subsystems, STATE_DIMENSION)
    #qgeo_traced_state = qgeo.ptrace(qgeo_state, trace_subsystems)

    #print(qgeo.sector_len_f(qgeo_traced_state))
    # print(f2(traced_state))

    print("Grading function")
    grad_func = grad(calc_target_sector_of_state)
    print("JITing function")
    grad_func_jitted = jit(grad_func)
    #print(make_jaxpr(grad_func_jitted)(state))

    print("Computing Result")
    result = grad_func_jitted(state)
    result.block_until_ready()
    print(f"Result: {result}")

    #for i in range(10):
    #    with about_time() as t:
            #result = grad_func_jitted(state)
            #result[0].block_until_ready()
            #print(f"Result: {result}")

        #print(f"[Dim {STATE_DIMENSION}, Pass {i}]: {t.duration} seconds")

    # t1 = about_time(block_partial_trace, f, state, trace_subsystems)
    # print(f"First pass: {t1.duration} seconds")
    #
    # t2 = about_time(block_partial_trace, f, state, trace_subsystems)
    # print(f"Second pass: {t2.duration} seconds")
    #
    # state = ket("0" * STATE_DIMENSION, 2)
    # t3 = about_time(block_partial_trace, f, state, trace_subsystems)
    # print(f"Third pass: {t3.duration} seconds")
    #
    # t4 = about_time(block_partial_trace, f, state, trace_subsystems)
    # print(f"Fourth pass: {t4.duration} seconds")

    return

def block_partial_trace(partial_trace_callable, *args):
    result = partial_trace_callable(*args)
    result.block_until_ready()

    return result

@partial(jit, static_argnums=(1,2,))
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

@jit
def calc_sector_lengths(rho):
    """ obtains sector lenghts / weights trough purities and Rains transform
            faster, and for arbitrary dimensions
        """
    n = TARGET_DIMENSION

    Ap = np.zeros(n + 1)

    for k in range(n + 1):
        for red in all_kRDMs(rho, n=n, k=k):
            Ap = Ap.at[k].set(Ap[k] + purity(red))

    # transform (Rains) unitary to Shor-Laflamme primary enumerator
    x_symbols = symbols("x0:%d" % (n + 1))

    Wp = qce.make_enum(x_symbols, n)
    W = qce.unitary_to_primary(Wp)
    # A = qce.poly_coeffs(W)
    A = qce.W_coeffs(W)
    subs_A = [lambdify(x_symbols, el)(*Ap) for el in A]

    return subs_A


def all_kRDMs(rho, n, k=2, verbose=False):
    """ generator of all reduced states of size k,
        optionally padded with identities
    """
    l = numpy.arange(n)

    for to in itt.combinations(l, n - k):
        to = list(to)
        if verbose:
            print('trace over', to)

        rho_red = calc_partial_trace(rho, tuple(to), n)

        yield rho_red

def calc_target_sector_of_state(state):
    trace_subsystems = tuple([j for j in range(TARGET_DIMENSION, STATE_DIMENSION)])
    traced_state = calc_partial_trace(state, trace_subsystems, STATE_DIMENSION)

    sector_lengths = calc_sector_lengths(traced_state)

    return sector_lengths[TARGET_DIMENSION]

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

"""
Spherical Manifolds: https://eprints.whiterose.ac.uk/78407/1/SphericalFinal.pdf
Riemann Optimization: https://andbloch.github.io/Stochastic-Gradient-Descent-on-Riemannian-Manifolds/
"""
