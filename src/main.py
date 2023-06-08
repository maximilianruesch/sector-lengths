from functools import partial

import alive_progress
import jax.numpy as np
import jax_qgeo
import numpy
from jax import jit, value_and_grad
from jax_qgeo import make_dm, purity
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

STATE_DIMENSION = 8
TARGET_DIMENSION = 4
STEP_SIZE_ALPHA = 0.00001
NUM_ITERATIONS = 1000


def run():
    print("Grading function...")
    grad_func = value_and_grad(calc_target_sector_of_state)
    print("JITing function...")
    grad_func_jitted = jit(grad_func)

    print("Starting random state construct...")
    with about_time() as t2:
        state = jax_qgeo.rand_pure_state(STATE_DIMENSION)
    print(f"Random state construct took {t2.duration} seconds")
    print(f"Size of the state: {len(state)}")
    print(f"Purity of the state: {qgeo.purity(qgeo.make_dm(state))}")
    print(f"Initial sector length {TARGET_DIMENSION}: {qgeo.sector_len_f(state)[TARGET_DIMENSION]}")

    print("Initially grading...")
    with about_time() as grad_time:
        result = grad_func_jitted(state)
        result[1].block_until_ready()
    print(f"Initially grading took {grad_time.duration} seconds")
    print("-----------------------------------------------------------------")
    for _ in alive_progress.alive_it(range(NUM_ITERATIONS), force_tty=True):
        result = grad_func_jitted(state)
        state = compute_new_state(state, result[1])

    print(f"Resulting sector length {TARGET_DIMENSION}: {grad_func_jitted(state)[0]}")
    print(f"Original sector length {TARGET_DIMENSION}: {qgeo.sector_len_f(state)[TARGET_DIMENSION]}")
    print(f"Resulting purity: {jax_qgeo.purity(jax_qgeo.make_dm(state))}")

    return


@partial(jit, static_argnums=(1,2,))
def calc_partial_trace(rho, trace_over, n):  # former ptrace_dim
    """ partial trace over subsystems specified in trace_over for arbitrary
        n-quDit systems (also of heteregeneous dimensions)
        e.g. ptrace(rho_ABC, [1]) = rhoA_C
        if pad is True, state will be tensored by identity to its original size
    args:       rho :   ndarray
                trace_over: list of subsystems to trace out, counting starts at 0
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

def calc_a():
    n = STATE_DIMENSION
    # transform (Rains) unitary to Shor-Laflamme primary enumerator
    x_symbols = symbols("x0:%d" % (n + 1))
    Wp = qce.make_enum(x_symbols, n)
    W = qce.unitary_to_primary(Wp)
    A = qce.W_coeffs(W)

    return A[TARGET_DIMENSION], x_symbols[0:TARGET_DIMENSION + 1]

def calc_target_sector_of_state(rho):
    """ obtains sector lenghts / weights trough purities and Rains transform
            faster, and for arbitrary dimensions
        """
    (target_a, x_symbols_for_target) = calc_a()

    Ap = np.zeros(TARGET_DIMENSION + 1)
    for k in range(TARGET_DIMENSION + 1):
        for rho_red in all_kRDMs(rho, n=STATE_DIMENSION, k=k):
            Ap = Ap.at[k].set(Ap[k] + purity(rho_red))

    return lambdify(x_symbols_for_target, target_a)(*Ap)


def all_kRDMs(rho, n, k=2, verbose=False):
    """ generator of all reduced states of size k """
    l = numpy.arange(n)

    for to in itt.combinations(l, n - k):
        to = list(to)
        if verbose:
            print('trace over', to)

        rho_red = calc_partial_trace(rho, tuple(to), n)

        yield rho_red

def compute_new_state(old_state, gradient):
    normal_proj = old_state * np.vdot(gradient, old_state) / np.linalg.norm(old_state)
    tangent_proj = gradient - normal_proj

    new_tangent_state = old_state + STEP_SIZE_ALPHA * tangent_proj

    tangent_norm = np.linalg.norm(tangent_proj)
    new_state = np.cos(tangent_norm) * new_tangent_state + np.sin(tangent_norm) * (tangent_proj / tangent_norm)

    return new_state


if __name__ == '__main__':
    run()

"""
Spherical Manifolds: https://eprints.whiterose.ac.uk/78407/1/SphericalFinal.pdf
Riemann Optimization: https://andbloch.github.io/Stochastic-Gradient-Descent-on-Riemannian-Manifolds/
"""
