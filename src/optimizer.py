import numpy
import jax.numpy as np
from omegaconf import DictConfig


class Optimizer:
    """
    Resembles an optimizer for optimizing on a sphere of size log_2(`old_params`).
    """
    _config: DictConfig
    _states = None

    def __init__(self, config: DictConfig, states):
        self._config = config
        self._states = states

    def _proj_u_on_x_tangent(self, x, u):
        return u - (np.vdot(x, u) / np.linalg.norm(x)) * x

    def _exp_map(self, cur_point, tangent_direction, step_size):
        tangent_norm = np.linalg.norm(tangent_direction)
        return np.cos(tangent_norm * step_size) * cur_point +\
            np.sin(tangent_norm * step_size) * (tangent_direction / tangent_norm)

    def step(self, old_params, gradient):
        raise NotImplementedError("The step function is not implemented in the general `Optimizer`")


class GradientDescent(Optimizer):
    """
    A simple gradient descent algorithm without any state (thus "local).
    """
    def step(self, old_params, gradient):
        # compute tangent projection of gradient
        tangent_proj = self._proj_u_on_x_tangent(old_params, gradient)

        # step along geodesic on hypersphere
        return self._exp_map(old_params, tangent_proj, step_size=self._config.states.stepSize)


class Adam(Optimizer):
    """
    The Adam algorithm generalized to work on spheres. Has a state for keeping latest momentum values.
    """
    _adam_state: dict

    def __init__(self, config: DictConfig, states):
        super().__init__(config, states)
        self._adam_state = {}

    def step(self, old_params, gradient):
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

        return self._states.normalize(new_point)
