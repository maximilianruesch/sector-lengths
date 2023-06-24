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
from jax.experimental.compilation_cache import compilation_cache as cc
from jax_qgeo import make_dm, purity
from sympy import symbols, lambdify
from matplotlib.axes import Axes
from matplotlib import cm, colors, pyplot as plt

from lib.jax_qgeo import ket, string_permutations_unique

OUTPUT_DIRECTORY = os.path.abspath('../data')

STATE_DIMENSION = 2
TARGET_DIMENSION = 1
STEP_SIZE_ALPHA = 0.005
NUM_ITERATIONS = 30


def run():
    states = SymmetricPureStates()
    print(f"[I] Using {states.__class__}")

    print("Grading function...")
    grad_func = value_and_grad(states.calc_target_sector)
    print("JITing function...")
    grad_func_jitted = jit(grad_func)

    print("Starting random state construct...")
    with about_time() as t2:
        state_params = states.construct_random()
    print(f"Random state construct took {t2.duration} seconds")

    # state_params = states.normalize(numpy.array([0.75, 0.23, 0.27]))
    states.stats(state_params)

    print("Initially grading...")
    with about_time() as grad_time:
        result = grad_func_jitted(state_params)
        result[1].block_until_ready()
    print(f"Initially grading took {grad_time.duration} seconds")
    print("-----------------------------------------------------------------")

    states.axes = generate_plot(grad_func_jitted)

    for _ in alive_progress.alive_it(range(NUM_ITERATIONS), force_tty=True):
        result = grad_func_jitted(state_params)
        print((result[0], numpy.average(result[1]), numpy.linalg.norm(result[1])))
        state_params = states.new_state_params(state_params, result[1])

    print(f"[T] (N k) witness: {math.comb(STATE_DIMENSION, TARGET_DIMENSION)}")

    # Final state normalization
    state_params = states.normalize(state_params)
    final_sector_length = grad_func_jitted(state_params)[0]
    states.final_stats(state_params, final_sector_length=final_sector_length)

    file_name = export_state(states.get_file_prefix(), state_params, final_sector_length)
    print(f"Saved result to {file_name}")

    print("Calculating symmetric properties of the state...")
    print(states.symmetric_bins(state_params))
    print("-----------------------------------------------------------------------------------------------------------")

    plt.ioff()
    plt.show()

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
    axes: Axes = None

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

        self.axes.quiver(*old_params, *(gradient * 0.1), color="g")
        self.axes.quiver(*old_params, *(tangent_proj * 0.1), color="r")

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
        return self.normalize(numpy.random.rand(STATE_DIMENSION + 1))  # + 1j * numpy.random.rand(STATE_DIMENSION + 1)

    def normalize(self, state_params):
        blow_up_factor = numpy.sqrt([math.comb(STATE_DIMENSION, k) for k in range(STATE_DIMENSION + 1)])
        blown_up_params = state_params * blow_up_factor

        normed_blown_up_params = super().normalize(blown_up_params)

        return normed_blown_up_params / blow_up_factor

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

        blown_up_params = old_params * blow_up_factor
        blown_up_gradient = gradient

        new_blown_up_params = super().new_state_params(blown_up_params, blown_up_gradient)

        return self.normalize(new_blown_up_params / blow_up_factor)


def generate_plot(sector_len_f):
    resolution = 100
    u = numpy.linspace(0, 2 * numpy.pi, resolution)
    v = numpy.linspace(0, numpy.pi, resolution)

    x = numpy.outer(numpy.cos(u), numpy.sin(v))
    y = numpy.outer(numpy.sin(u), numpy.sin(v)) / numpy.sqrt(2)
    z = numpy.outer(numpy.ones(numpy.size(u)), numpy.cos(v))

    raw = numpy.stack((x, y, z), axis=2)

    # Remap to sphere
    y = y * numpy.sqrt(2)

    values = [numpy.zeros(resolution) for _ in range(resolution)]
    for i, j in itt.product(range(resolution), range(resolution)):
        coords = raw[i][j]

        values[i][j] = sector_len_f(np.array(coords))[0]

        # new_r = values[i][j] * 0.1
        # x[i][j] *= (1 + new_r)
        # y[i][j] *= (1 + new_r)
        # z[i][j] *= (1 + new_r)


    ax: Axes = plt.axes(projection="3d")
    ax.plot_surface(x, y, z, cmap="plasma", facecolors=cm.plasma(colors.Normalize()(values)), rstride=1, cstride=1, antialiased=False)

    m = cm.ScalarMappable(cmap=cm.plasma)
    m.set_array(numpy.linspace(numpy.min(values), numpy.max(values), 10))
    plt.colorbar(m, ax=ax)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    return ax



if __name__ == '__main__':
    cc.initialize_cache("cache")

    run()

"""
Spherical Manifolds: https://eprints.whiterose.ac.uk/78407/1/SphericalFinal.pdf
Riemann Optimization: https://andbloch.github.io/Stochastic-Gradient-Descent-on-Riemannian-Manifolds/
"""
