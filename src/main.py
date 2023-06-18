import itertools as itt  # for combinatorics, choice, permutations, etc
import json
import math
import os.path
from datetime import datetime
from functools import partial

import alive_progress
import jax.numpy as np
import jax_qgeo
import jax_qgeo.qcode_enum as qce
import numpy
import qgeo
from about_time import about_time
from jax import jit, value_and_grad
from jax_qgeo import make_dm, purity
from sympy import symbols, lambdify

from lib.jax_qgeo import ket, string_permutations_unique

OUTPUT_DIRECTORY = os.path.abspath('../data')

STATE_DIMENSION = 5
TARGET_DIMENSION = 4
STEP_SIZE_ALPHA = 0.0005
NUM_ITERATIONS = 1000


def run():
    states = AllPureStates()
    print(f"[I] Using {states.__class__}")

    print("Grading function...")
    grad_func = value_and_grad(states.calc_target_sector)
    print("JITing function...")
    grad_func_jitted = jit(grad_func)

    print("Starting random state construct...")
    with about_time() as t2:
        state_params = states.construct_random()
    print(f"Random state construct took {t2.duration} seconds")

    # state_params = numpy.array([0.32302107, 0.65313237, 0.20613978], dtype=complex)
    states.stats(state_params)

    print("Initially grading...")
    with about_time() as grad_time:
        result = grad_func_jitted(state_params)
        result[1].block_until_ready()
    print(f"Initially grading took {grad_time.duration} seconds")
    print("-----------------------------------------------------------------")
    for _ in alive_progress.alive_it(range(NUM_ITERATIONS), force_tty=True):
        result = grad_func_jitted(state_params)
        print((result[0], numpy.average(result[1])))
        state_params = states.new_state_params(state_params, result[1])

    print(f"[T] (N k) witness: {math.comb(STATE_DIMENSION, TARGET_DIMENSION)}")

    # Final state normalization
    state_params = states.normalize(state_params)
    states.final_stats(state_params, final_sector_length=grad_func_jitted(state_params)[0])

    #file_name = export_state(states.get_file_prefix(), state, exact_sector_length)
    #print(f"Saved result to {file_name}")

    print("Calculating symmetric properties of the state...")
    print(states.symmetric_bins(state_params))
    print("-----------------------------------------------------------------------------------------------------------")

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

def calc_a(n):
    # transform (Rains) unitary to Shor-Laflamme primary enumerator
    x_symbols = symbols("x0:%d" % (n + 1))
    Wp = qce.make_enum(x_symbols, n)
    W = qce.unitary_to_primary(Wp)
    A = qce.W_coeffs(W)

    return A[TARGET_DIMENSION], x_symbols[0:TARGET_DIMENSION + 1]

def all_kRDMs(rho, n, k=2, verbose=False):
    """ generator of all reduced states of size k """
    l = numpy.arange(n)

    for to in itt.combinations(l, n - k):
        to = list(to)
        if verbose:
            print('trace over', to)

        rho_red = calc_partial_trace(rho, tuple(to), n)

        yield rho_red

def export_state(file_prefix, state, sector_length):
    timestamp=int(datetime.timestamp(datetime.now()))
    file_name = f"{file_prefix}result_s{STATE_DIMENSION}_t{TARGET_DIMENSION}_{timestamp}_{sector_length}.json"
    file_path = os.path.join(OUTPUT_DIRECTORY, file_name)

    with open(file_path, 'w') as file:
        json.dump(
            [str(x) for x in state],
            file,
        )

    return file_name

class AllPureStates:
    def construct_random(self):
        random_numbers = numpy.random.rand(2 ** STATE_DIMENSION) + 1j * numpy.random.rand(2 ** STATE_DIMENSION)

        return self.normalize(random_numbers)

    def normalize(self, state_params):
        return state_params / np.linalg.norm(state_params)

    def get_file_prefix(self):
        return ""

    def stats(self, state_params):
        print(f"Size of the state: {len(state_params)}")
        # print(f"Purity of the state: {qgeo.purity(qgeo.make_dm(state_params))}")
        print(f"Initial sector length {TARGET_DIMENSION}: {qgeo.sector_len_f(state_params)[TARGET_DIMENSION]}")
        print(f"Initial purity: {jax_qgeo.purity(jax_qgeo.make_dm(state_params))}")

    def final_stats(self, final_state_params, final_sector_length):
        exact_sector_length = qgeo.sector_len_f(final_state_params)[TARGET_DIMENSION]
        print(f"[N] Resulting sector length {TARGET_DIMENSION}: {final_sector_length}")
        print(f"[N] Original sector length {TARGET_DIMENSION}: {exact_sector_length}")
        print(f"[N] Resulting purity: {jax_qgeo.purity(jax_qgeo.make_dm(final_state_params))}")

    def symmetric_bins(self, state_params):
        bins = [numpy.array([]) for _ in range(STATE_DIMENSION + 1)]
        for index, j in enumerate(state_params):
            bins[index.bit_count()] = numpy.append(bins[index.bit_count()], [j])

        min_max_bins = [(numpy.min(sym_bin), numpy.max(sym_bin)) for sym_bin in bins]

        return [mmbin[1] - mmbin[0] for mmbin in min_max_bins]

    def calc_target_sector(self, state_params):
        """ obtains sector lenghts / weights trough purities and Rains transform
                    faster, and for arbitrary dimensions
                """
        (target_a, x_symbols_for_target) = calc_a(STATE_DIMENSION)

        Ap = np.zeros(TARGET_DIMENSION + 1)
        for k in range(TARGET_DIMENSION + 1):
            for rho_red in all_kRDMs(state_params, n=STATE_DIMENSION, k=k):
                Ap = Ap.at[k].set(Ap[k] + purity(rho_red))

        return lambdify(x_symbols_for_target, target_a)(*Ap)

    def new_state_params(self, old_params, gradient):
        # compute tangent projection of gradient
        normal_proj = old_params * np.vdot(gradient, old_params) / np.linalg.norm(old_params)
        tangent_proj = gradient - normal_proj

        # step along geodesic on hypersphere
        tangent_norm = np.linalg.norm(tangent_proj)
        new_state = np.cos(tangent_norm * STEP_SIZE_ALPHA) * old_params +\
                    np.sin(tangent_norm * STEP_SIZE_ALPHA) * (tangent_proj / tangent_norm)

        return new_state


class SymmetricPureStates(AllPureStates):
    def _denormalized_dicke(self, n, k):
        s = k * '1' + (n - k) * '0'
        s_list = string_permutations_unique(s)
        psi = numpy.sum(numpy.array([ket(el) for el in s_list]), axis=0)
        return psi

    def _construct_dicke(self, state_params):
        dicke_states = []
        for k in range(STATE_DIMENSION + 1):
            dicke_states.append(self._denormalized_dicke(STATE_DIMENSION, k))

        return np.dot(state_params, np.array(dicke_states))

    def construct_random(self):
        return self.normalize(numpy.random.rand(STATE_DIMENSION + 1))

    def normalize(self, state_params):
        blow_up_factor = [math.comb(STATE_DIMENSION, k) for k in range(STATE_DIMENSION + 1)]
        blown_up_params = numpy.multiply(state_params, blow_up_factor)

        normed_blown_up_params = super().normalize(blown_up_params)

        return numpy.divide(normed_blown_up_params, numpy.sqrt(blow_up_factor))

    def get_file_prefix(self):
        return "symm_"

    def stats(self, state_params):
        super().stats(self._construct_dicke(state_params))

    def final_stats(self, final_state_params, final_sector_length):
        super().final_stats(self._construct_dicke(final_state_params), final_sector_length)

    def symmetric_bins(self, state_params):
        return super().symmetric_bins(self._construct_dicke(state_params))

    def calc_target_sector(self, state_params):
        """ calculates sector lengths only for a single RDM and multiplies by the RDM count """
        rho = self._construct_dicke(state_params)
        (target_a, x_symbols_for_target) = calc_a(TARGET_DIMENSION)

        trace_over = [i for i in range(TARGET_DIMENSION, STATE_DIMENSION)]
        rho_red = calc_partial_trace(rho, trace_over=tuple(trace_over), n=STATE_DIMENSION)

        Ap = np.zeros(TARGET_DIMENSION + 1)
        for k in range(TARGET_DIMENSION + 1):
            for rho_red_red in all_kRDMs(rho_red, n=TARGET_DIMENSION, k=k):
                Ap = Ap.at[k].set(Ap[k] + purity(rho_red_red))

        return lambdify(x_symbols_for_target, target_a)(*Ap) * math.comb(STATE_DIMENSION, TARGET_DIMENSION)

    def new_state_params(self, old_params, gradient):
        blow_up_factor = numpy.sqrt([math.comb(STATE_DIMENSION, k) for k in range(STATE_DIMENSION + 1)])
        blown_up_params = numpy.multiply(old_params, blow_up_factor)
        blown_up_gradient = numpy.multiply(gradient, blow_up_factor)

        # compute tangent projection of gradient
        normal_proj = blown_up_params * np.vdot(blown_up_gradient, blown_up_params) / np.linalg.norm(blown_up_params)
        tangent_proj = blown_up_gradient - normal_proj

        # step along tangent space
        scaled_projection = STEP_SIZE_ALPHA * tangent_proj

        # exp_map new tangent point to hypersphere surface
        tangent_norm = np.linalg.norm(scaled_projection)
        new_state = np.cos(tangent_norm) * blown_up_params + np.sin(tangent_norm) * (scaled_projection / tangent_norm)

        if tangent_norm < 1E-6:
            return old_params

        return (new_state / np.linalg.norm(new_state)) / numpy.sqrt(blow_up_factor)


if __name__ == '__main__':
    run()

"""
Spherical Manifolds: https://eprints.whiterose.ac.uk/78407/1/SphericalFinal.pdf
Riemann Optimization: https://andbloch.github.io/Stochastic-Gradient-Descent-on-Riemannian-Manifolds/
"""
