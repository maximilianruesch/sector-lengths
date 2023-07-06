import itertools as itt  # for combinatorics, choice, permutations, etc
import json
import math
import os.path
from datetime import datetime
from functools import partial
import logging
import alive_progress
import jax.numpy as np
import jax_qgeo
import numpy
import qgeo
import hydra
from omegaconf import DictConfig
from about_time import about_time
from jax import jit, value_and_grad
from jax.experimental.compilation_cache import compilation_cache as cc
from jax_qgeo import make_dm, purity, qcode_enum as qce
from sympy import symbols, lambdify
from matplotlib.axes import Axes
from matplotlib import cm, colors, pyplot as plt

@hydra.main(version_base=None, config_path="../conf", config_name=".config.yaml")
def run(config: DictConfig):
    states = None
    with about_time() as t:
        match config.states.stateType:
            case "all": states = AllPureStates(config)
            case "symmetric": states = SymmetricPureStates(config)
    print(f"State object construct took {t.duration} seconds")
    print(f"[I] Using {states.__class__}")

    print("Grading function...")
    grad_func = value_and_grad(states.calc_target_sector)
    print("JITing function...")
    grad_func_jitted = jit(grad_func)

    print("Starting random state construct...")
    with about_time() as t2:
        state_params = states.construct_random()
    print(f"Random state construct took {t2.duration} seconds")

    # states.stats(state_params)

    print("Initially grading...")
    with about_time() as grad_time:
        grad_func_jitted(state_params)[1].block_until_ready()
    print(f"Initially grading took {grad_time.duration} seconds")
    print("-----------------------------------------------------------------")

    if config.plot:
        states.axes = generate_plot(grad_func_jitted, states)

    def do_iteration(it_sp):
        it_r = grad_func_jitted(it_sp)
        print((it_r[0], numpy.average(it_r[1]), numpy.linalg.norm(it_r[1])))
        return it_r[0], states.new_state_params(it_sp, it_r[1])

    if config.iterations == 'auto':
        sector_lengths = []
        with alive_progress.alive_bar(force_tty=True) as bar:
            for _ in range(config.maxIterations):
                sector_length, state_params = do_iteration(state_params)

                sector_lengths.append(sector_length)
                last_n_sector_lengths = sector_lengths[-10:]
                if len(last_n_sector_lengths) == 10 \
                        and numpy.max(last_n_sector_lengths) - numpy.min(last_n_sector_lengths) < 5e-3:
                    break
                bar()
    else:
        for _ in alive_progress.alive_it(range(config.iterations), force_tty=True):
            __, state_params = do_iteration(state_params)


    print(f"[T] (N k) witness: {math.comb(config.qubitCount, config.target)}")

    # Final state normalization
    state_params = states.normalize(state_params)
    final_sector_length = grad_func_jitted(state_params)[0]
    # states.final_stats(state_params, final_sector_length=final_sector_length)

    print("Calculating symmetric properties of the state...")
    print(states.symmetric_bins(state_params))

    if config.export.enabled == 'ask' and input("Export state? [y/n]: ") == "y"\
            or config.export.enabled is True:
        print(f"Saved result to {states.export(state_params, final_sector_length)}")

    print("-----------------------------------------------------------------------------------------------------------")

    if config.plot:
        plt.ioff()
        plt.show()

    return


@partial(jit, static_argnums=(1,2,))
def calc_partial_trace(rho, trace_over, n):  # former ptrace_dim
    """ partial trace over subsystems specified in trace_over for arbitrary
        n-quDit systems (also of heterogeneous dimensions)
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

def calc_a(n, t):
    # transform (Rains) unitary to Shor-Laflamme primary enumerator
    x_symbols = symbols("x0:%d" % (n + 1))
    Wp = qce.make_enum(x_symbols, n)
    W = qce.unitary_to_primary(Wp)
    A = qce.W_coeffs(W)

    return A[t], x_symbols[0:t + 1]

def all_kRDMs(rho, n, k=2, verbose=False):
    """ generator of all reduced states of size k """
    l = numpy.arange(n)

    for to in itt.combinations(l, n - k):
        to = list(to)
        if verbose:
            print('trace over', to)

        rho_red = calc_partial_trace(rho, tuple(to), n)

        yield rho_red

class AllPureStates:
    _config: DictConfig

    axes: Axes = None
    _file_prefix: str = ""

    _adam_state: dict = {}

    def __init__(self, config: DictConfig):
        self._config = config

    def construct_random(self):
        random_numbers = numpy.random.rand(2 ** self._config.qubitCount)
        if not self._config.onlyRealValues:
            random_numbers = random_numbers + 1j * numpy.random.rand(2 ** self._config.qubitCount)

        return self.normalize(random_numbers)

    def normalize(self, state_params):
        return state_params / np.linalg.norm(state_params)

    def export(self, state_params, sector_length):
        directory = os.path.join(os.path.abspath('../'), self._config.export.directory)
        os.makedirs(directory, exist_ok=True)

        timestamp = int(datetime.timestamp(datetime.now()))
        file_name = f"{self._file_prefix}result_s{self._config.qubitCount}_t{self._config.target}_{timestamp}_{sector_length}.json"
        file_path = os.path.join(directory, file_name)

        with open(file_path, 'w') as file:
            json.dump(
                {
                    "state": [str(x) for x in state_params],
                    "sector_length": { str(self._config.target): float(sector_length) },
                    "purity": float(jax_qgeo.purity(jax_qgeo.make_dm(state_params)))
                },
                file,
            )

        return file_name

    def stats(self, state_params):
        print(f"Size of the state: {len(state_params)}")
        print(f"Initial sector length {self._config.target}: {qgeo.sector_len_f(state_params)[self._config.target]}")
        print(f"Initial purity: {jax_qgeo.purity(jax_qgeo.make_dm(state_params))}")

    def final_stats(self, final_state_params, final_sector_length):
        exact_sector_length = qgeo.sector_len_f(final_state_params)[self._config.target]
        print(f"[N] Resulting sector length {self._config.target}: {final_sector_length}")
        print(f"[N] Original sector length {self._config.target}: {exact_sector_length}")
        print(f"[N] Resulting purity: {jax_qgeo.purity(jax_qgeo.make_dm(final_state_params))}")

    def symmetric_bins(self, state_params):
        bins = [numpy.array([]) for _ in range(self._config.qubitCount + 1)]
        for index, j in enumerate(state_params):
            bins[index.bit_count()] = numpy.append(bins[index.bit_count()], [j])

        min_max_bins = [(numpy.min(sym_bin), numpy.max(sym_bin)) for sym_bin in bins]

        return [min_max_bin[1] - min_max_bin[0] for min_max_bin in min_max_bins]

    def calc_target_sector(self, state_params):
        """ obtains sector lengths / weights trough purities and Rains transform
                    faster, and for arbitrary dimensions
                """
        (target_a, x_symbols_for_target) = calc_a(self._config.qubitCount, self._config.target)

        Ap = np.zeros(self._config.target + 1)
        for k in range(self._config.target + 1):
            for rho_red in all_kRDMs(state_params, n=self._config.qubitCount, k=k):
                Ap = Ap.at[k].set(Ap[k] + purity(rho_red))

        return lambdify(x_symbols_for_target, target_a)(*Ap)

    def new_state_params(self, old_params, gradient):
        if self._config.adam.enabled:
            return self._adam_step(old_params, gradient)

        # compute tangent projection of gradient
        tangent_proj = self._proj_u_on_x_tangent(old_params, gradient)

        if self._config.plot:
            self.axes.quiver(*old_params, *(gradient * 0.1), color="g")
            self.axes.quiver(*old_params, *(tangent_proj * 0.1), color="r")

        # step along geodesic on hypersphere
        return self._exp_map(old_params, tangent_proj, step_size=self._config.states.stepSize)

    def _proj_u_on_x_tangent(self, x, u):
        return u - (np.vdot(x, u) / np.linalg.norm(x)) * x

    def _exp_map(self, cur_point, tangent_direction, step_size):
        tangent_norm = np.linalg.norm(tangent_direction)
        return np.cos(tangent_norm * step_size) * cur_point +\
            np.sin(tangent_norm * step_size) * (tangent_direction / tangent_norm)

    def _adam_step(self, old_params, gradient):
        config = self._config.adam
        state = self._adam_state

        # state initialization
        if len(state) == 0:
            state["step"] = 0
            # Exponential moving average of gradient values
            state["m_t"] = numpy.zeros_like(old_params)
            # Exponential moving average of squared gradient values
            state["v_t"] = numpy.zeros_like(old_params)

        state["step"] += 1

        # actual step
        g_t = self._proj_u_on_x_tangent(old_params, gradient + old_params)

        betas = config.betas
        m_t = state["m_t"] * betas[0] + (1 - betas[0]) * g_t
        v_t = state["v_t"] * betas[1] + (1 - betas[1]) * numpy.inner(g_t, g_t)
        bias_correction1 = 1 - betas[0] ** state["step"]
        bias_correction2 = 1 - betas[1] ** state["step"]

        # get the direction for ascend
        direction = (m_t / bias_correction1) / (np.sqrt(v_t / bias_correction2) + config.eps)
        # step along geodesic on hypersphere
        new_point = self._exp_map(old_params, direction, step_size=config.learningRate)
        # transport the exponential averaging to the new point. this is an approximation by changing the tangent space with projection
        m_t_new = self._proj_u_on_x_tangent(new_point, m_t)

        state["m_t"] = m_t_new
        state["v_t"] = v_t

        self._adam_state = state

        return self.normalize(new_point)

class SymmetricPureStates(AllPureStates):
    _file_prefix: str = "symm_"
    _sphere_mapping: numpy.ndarray
    _dicke_states: numpy.ndarray

    def __init__(self, config):
        super().__init__(config)
        self._sphere_mapping = numpy.sqrt([math.comb(self._config.qubitCount, k) for k in range(self._config.qubitCount + 1)])

        def _fast_denormalized_dicke(n, k):
            ret = numpy.zeros(2 ** self._config.qubitCount)

            powers = [1 << e for e in range(n)]
            for bits in itt.combinations(powers, k):
                ret[sum(bits)] = 1

            return ret

        self._dicke_states = numpy.array([_fast_denormalized_dicke(self._config.qubitCount, k) for k in range(self._config.qubitCount + 1)])

    def _construct_dicke(self, state_params):
        return np.dot(super().normalize(state_params) / self._sphere_mapping, self._dicke_states)

    def construct_random(self):
        random_numbers = numpy.random.rand(self._config.qubitCount + 1)
        if not self._config.onlyRealValues:
            random_numbers = random_numbers + 1j * numpy.random.rand(self._config.qubitCount + 1)

        return self.normalize(random_numbers)

    def normalize(self, state_params):
        return super().normalize(state_params * self._sphere_mapping) / self._sphere_mapping

    def stats(self, state_params):
        super().stats(self._construct_dicke(state_params))

    def final_stats(self, final_state_params, final_sector_length):
        super().final_stats(self._construct_dicke(final_state_params), final_sector_length)

    def symmetric_bins(self, state_params):
        return super().symmetric_bins(self._construct_dicke(state_params))

    def calc_target_sector(self, state_params):
        """ calculates sector lengths only for a single RDM and multiplies by the RDM count """
        rho = self._construct_dicke(state_params)
        (target_a, x_symbols_for_target) = calc_a(self._config.target, self._config.target)

        trace_over = [i for i in range(self._config.target, self._config.qubitCount)]
        rho_red = calc_partial_trace(rho, trace_over=tuple(trace_over), n=self._config.qubitCount)

        Ap = np.zeros(self._config.target + 1)
        for k in range(self._config.target + 1):
            for rho_red_red in all_kRDMs(rho_red, n=self._config.target, k=k):
                Ap = Ap.at[k].set(Ap[k] + purity(rho_red_red))

        return lambdify(x_symbols_for_target, target_a)(*Ap) * math.comb(self._config.qubitCount, self._config.target)

    def new_state_params(self, old_params, gradient):
        return self.normalize(super().new_state_params(old_params=super().normalize(old_params), gradient=gradient))

def generate_plot(sector_len_f, states):
    resolution = 100
    u = numpy.linspace(0, 2 * numpy.pi, resolution)
    v = numpy.linspace(0, numpy.pi, resolution)

    x = numpy.outer(numpy.cos(u), numpy.sin(v))
    y = numpy.outer(numpy.sin(u), numpy.sin(v))
    z = numpy.outer(numpy.ones(numpy.size(u)), numpy.cos(v))

    raw = numpy.stack((x, y, z), axis=2)

    values = [numpy.zeros(resolution) for _ in range(resolution)]
    for i, j in itt.product(range(resolution), range(resolution)):
        coords = states.normalize(np.array(raw[i][j]))
        values[i][j] = sector_len_f(coords)[0]

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
    logging.getLogger('jax._src.dispatch').setLevel(logging.WARNING)
    cc.initialize_cache("cache")

    run()

"""
Spherical Manifolds: https://eprints.whiterose.ac.uk/78407/1/SphericalFinal.pdf
Riemann Optimization: https://andbloch.github.io/Stochastic-Gradient-Descent-on-Riemannian-Manifolds/
"""
