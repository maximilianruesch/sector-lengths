import numpy
from omegaconf import DictConfig

class Scheduler:
    """
    Resembles a scheduler for learning rates.
    """
    config: DictConfig
    init: float

    def __init__(self, config: DictConfig):
        self.config = config
        self.init = config.scheduler.init

    def getLR(self, step: int) -> float:
        raise NotImplementedError('Cannot call "getLR" on base scheduler')


class ConstantScheduler(Scheduler):
    def getLR(self, step: int) -> float:
        return self.init


class LinearScheduler(Scheduler):
    totalSteps: int

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.totalSteps = config.iterations

    def getLR(self, step: int) -> float:
        return self.init * (1 - float(step) / self.totalSteps)


class InverseSqrtScheduler(Scheduler):
    def getLR(self, step: int) -> float:
        return self.init / numpy.sqrt(numpy.maximum(1, step))


class NegSinScheduler(Scheduler):
    totalSteps: int

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.totalSteps = config.iterations

    def getLR(self, step: int) -> float:
        return self.init * -1 * numpy.sin(numpy.pi * step / (2 * self.totalSteps)) + self.init
