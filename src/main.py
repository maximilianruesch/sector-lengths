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
import wandb
from omegaconf import OmegaConf, DictConfig
from about_time import about_time
from jax import jit, value_and_grad, random, config as jconfig
from jax.experimental.compilation_cache import compilation_cache as cc
from jax_qgeo import make_dm, purity, qcode_enum as qce
from sympy import symbols, lambdify
from matplotlib.axes import Axes
from matplotlib import cm, colors, pyplot as plt

from src.optimizer import Optimizer, GradientDescent, Adam
from src.scheduler import ConstantScheduler, LinearScheduler, InverseSqrtScheduler, NegSinScheduler


@hydra.main(version_base=None, config_path="../conf", config_name=".config.yaml")
def run(config: DictConfig):
    jconfig.update("jax_enable_x64", config.float64)

    if config.report.enabled:
        wandb.init(
            project='sector-lengths',
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        )

    states = None
    with about_time() as t:
        match config.states.stateType:
            case "all": states = AllPureStates(config)
            case "symmetric": states = SymmetricPureStates(config)
    print(f"State object construct took {t.duration} seconds")
    print(f"[I] Using {states.__class__}")

    scheduler = None
    match config.scheduler.type:
        case "constant":
            scheduler = ConstantScheduler(config)
        case "linear":
            scheduler = LinearScheduler(config)
        case "inverseSqrt":
            scheduler = InverseSqrtScheduler(config)
        case "negSin":
            scheduler = NegSinScheduler(config)

    if config.adam.enabled:
        optimizer = Adam(config, states)
    else:
        optimizer = GradientDescent(config, states)
    states.set_optimizer(optimizer)

    print("Grading function...")
    grad_func = value_and_grad(states.get_calc_target_sector())
    print("JITing function...")
    grad_func_jitted = jit(grad_func)

    print("Starting state loading...")
    with about_time() as t2:
        if config.states.init == "random":
            state_params = states.construct_random()
        elif config.states.init == "real":
            if config.onlyRealValues:
                state_params = states.construct_random()
            else:
                print(f"Starting state with real values...")
                state_params = states._expand_complex(np.real(states._contract_complex(states.construct_random())))
        elif config.states.init == "symmetric":
            if isinstance(states, SymmetricPureStates):
                state_params = states.construct_random()
            else:
                print(f"Kickstarting state with symmetric...")
                ms = SymmetricPureStates(config)
                if config.onlyRealValues:
                    state_params = ms._construct_dicke(ms.construct_random())
                else:
                    state_params = ms._expand_complex(ms._construct_dicke(ms._contract_complex(ms.construct_random())))
        else:
            print(f"Trying to open state from: {config.states.init}")
            with open(config.states.init) as f:
                state_params = numpy.array(json.load(f)['state'], dtype=numpy.float64)
    print(f"State loading took {t2.duration} seconds")

    print("Starting lower and compile...")
    with about_time() as lower_timer:
        lowered = grad_func_jitted.lower(state_params)
    print(f"[TIME] Lowering took {lower_timer.duration} seconds")
    if config.report.enabled:
        wandb.run.summary["lowering_time"] = lower_timer.duration
    with about_time() as compile_timer:
        compiled = lowered.compile()
    print(f"[TIME] Compilation took {compile_timer.duration} seconds")
    if config.report.enabled:
        wandb.run.summary["compile_time"] = compile_timer.duration
    grad_func_jitted = compiled

    print("Initially grading...")
    with about_time() as grad_time:
        grad_func_jitted(state_params)[1].block_until_ready()
    print(f"Initially grading took {grad_time.duration} seconds")
    if config.report.enabled:
        wandb.run.summary["grading_time"] = grad_time.duration
    print("-----------------------------------------------------------------")

    if config.plot:
        states.axes = generate_plot(grad_func_jitted, states)

    with about_time() as optimize_timer:
        for step in alive_progress.alive_it(range(config.iterations), force_tty=True):
            it_r = grad_func_jitted(state_params)

            lr = scheduler.getLR(step)
            if config.report.enabled:
                wandb.log({
                    'sector': it_r[0],
                    'lr': lr
                })

            print((it_r[0], numpy.average(it_r[1]), numpy.linalg.norm(it_r[1]), lr))
            state_params = states.new_state_params(state_params, it_r[1], step_size=lr)
    print(f"Optimizing took {optimize_timer.duration} seconds")
    if config.report.enabled:
        wandb.run.summary["optimize_time"] = optimize_timer.duration

    print(f"[T] (N k) witness: {math.comb(config.qubitCount, config.target)}")

    # Final state normalization
    state_params = states.normalize(state_params)
    final_sector_length = grad_func_jitted(state_params)[0]
    # states.final_stats(state_params, final_sector_length=final_sector_length)

    # print("Calculating symmetric properties of the state...")
    # print(states.symmetric_bins(state_params))

    if config.export.enabled == 'ask' and input("Export state? [y/n]: ") == "y"\
            or config.export.enabled is True:
        print(f"Saved result to {states.export(state_params, final_sector_length)}")

    print("-----------------------------------------------------------------------------------------------------------")

    if config.plot:
        plt.ioff()
        plt.show()

    if config.report.enabled:
        wandb.finish()

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
    _optimizer: Optimizer = None
    _file_prefix: str = ""

    def __init__(self, config: DictConfig):
        self._config = config

    def set_optimizer(self, optimizer: Optimizer):
        self._optimizer = optimizer

    def construct_random(self):
        random_numbers = numpy.random.rand(2 ** self._config.qubitCount)
        if not self._config.onlyRealValues:
            random_numbers = self._expand_complex(random_numbers + 1j * numpy.random.rand(2 ** self._config.qubitCount))

        return self.normalize(random_numbers)

    def normalize(self, state_params):
        return state_params / np.linalg.norm(state_params)

    def export(self, state_params, sector_length):
        directory = os.path.join(os.path.abspath('../'), self._config.export.directory)
        os.makedirs(directory, exist_ok=True)

        timestamp = int(datetime.timestamp(datetime.now()))
        value_prefix = 'real_' if self._config.onlyRealValues else 'complex_'
        file_name = f"{self._file_prefix}{value_prefix}result_s{self._config.qubitCount}_t{self._config.target}_{timestamp}_{sector_length}.json"
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

    def get_calc_target_sector(self):
        if self._config.onlyRealValues:
            return lambda x: self._calc_target_sector(x)
        else:
            return lambda x: self._calc_target_sector(self._contract_complex(x))

    def _calc_target_sector(self, state_params):
        """ obtains sector lengths / weights trough purities and Rains transform
                    faster, and for arbitrary dimensions
                """
        (target_a, x_symbols_for_target) = calc_a(self._config.qubitCount, self._config.target)

        Ap = np.zeros(self._config.target + 1)
        for k in range(self._config.target + 1):
            for rho_red in all_kRDMs(state_params, n=self._config.qubitCount, k=k):
                Ap = Ap.at[k].set(Ap[k] + purity(rho_red))

        return lambdify(x_symbols_for_target, target_a)(*Ap)

    def _expand_complex(self, arg: np.ndarray) -> numpy.ndarray:
        res = numpy.empty(2 * arg.size)
        res[0::2] = numpy.real(arg)
        res[1::2] = numpy.imag(arg)
        return res

    def _contract_complex(self, res: np.ndarray):
        real = res[0::2]
        imag = res[1::2]
        return real + 1j * imag

    def new_state_params(self, old_params, gradient, step_size):
        if self._optimizer is None:
            raise ValueError("No optimizer set. Cannot step for new state parameters.")

        return self._optimizer.step(old_params, gradient, step_size=step_size)

class SymmetricPureStates(AllPureStates):
    _file_prefix: str = "symm_"
    _sphere_mapping: numpy.ndarray
    _dicke_states: numpy.ndarray

    def __init__(self, config: DictConfig):
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
        def rand_nums(subkey):
            # Note that this will not actually be x64 if JAX x64 mode is not enabled
            return random.uniform(subkey, (self._config.qubitCount + 1,), dtype=np.float64)

        subkey_1, subkey_2 = random.split(random.PRNGKey(numpy.random.randint(1_000_000_000)), 2)
        random_numbers = rand_nums(subkey_1)
        if not self._config.onlyRealValues:
            random_numbers = self._expand_complex(random_numbers + 1j * rand_nums(subkey_2))

        return self.normalize(random_numbers)

    def normalize(self, state_params):
        sp = state_params if self._config.onlyRealValues else self._contract_complex(state_params)
        n_sp = super().normalize(sp * self._sphere_mapping) / self._sphere_mapping
        return n_sp if self._config.onlyRealValues else self._expand_complex(n_sp)

    def stats(self, state_params):
        sp = state_params if self._config.onlyRealValues else self._contract_complex(state_params)
        super().stats(self._construct_dicke(sp))

    def final_stats(self, final_state_params, final_sector_length):
        super().final_stats(self._construct_dicke(final_state_params), final_sector_length)

    def symmetric_bins(self, state_params):
        return super().symmetric_bins(self._construct_dicke(state_params))

    def get_calc_target_sector(self):
        if not self._config.onlyRealValues:
            return lambda x: self._calc_target_sector(self._contract_complex(x))
        else:
            return lambda x: self._calc_target_sector(x)

    def _calc_target_sector(self, state_params):
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

    def new_state_params(self, old_params, gradient, step_size):
        return self.normalize(super().new_state_params(old_params=super().normalize(old_params), gradient=gradient, step_size=step_size))

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
