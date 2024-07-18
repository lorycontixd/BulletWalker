import enum
import numpy as np
from abc import ABC, abstractmethod


class Callback(ABC):
    """Abstract class for simulation callbacks."""

    def __init__(self, simulator):
        self.simulator = simulator

    @abstractmethod
    def on_simulation_start(self):
        pass

    @abstractmethod
    def on_simulation_step(self):
        pass

    @abstractmethod
    def on_simulation_end(self):
        pass


class EarlyStoppingCallback(Callback):
    """Callback to stop the simulation early.
    ...
    """

    class EarlyStoppingMode(enum.IntEnum):
        MIN = enum.auto()
        MAX = enum.auto()
        EXACT = enum.auto()
        AUTO = enum.auto()

    def _init_(
        self,
        monitor: str = "index",
        min_delta: float = 0.0,
        mode: EarlyStoppingMode = EarlyStoppingMode.AUTO,
    ) -> None:

        self.monitor = monitor
        self.mode = mode
        self.min_delta = abs(min_delta)
        self.stopping_step = 0
        self.monitor_operation = None
        self.best = None

        if self.mode == EarlyStoppingCallback.EarlyStoppingMode.MIN:
            self.monitor_operation = np.less
        elif self.mode == EarlyStoppingCallback.EarlyStoppingMode.MAX:
            self.monitor_operation = np.greater
        elif self.mode == EarlyStoppingCallback.EarlyStoppingMode.EXACT:
            self.best_value = np.equal
        else:
            raise NotImplementedError(
                f"Early stopping mode {self.mode} not implemented yet"
            )

        if self.monitor_operation == np.greater:
            self.min_delta *= 1
        elif self.monitor_operation == np.less:
            self.min_delta *= -1
        else:
            self.min_delta = 0

    def on_simulation_start(self):
        self.stopping_step = 0
        self.best = np.inf if self.monitor_op == np.less else -np.inf

    def on_simulation_step(self, simulation_step):
        current = getattr(simulation_step, self.monitor)
        if current is None:
            raise ValueError(f"Early stopping monitor {self.monitor} not found")
        if self._is_improvement(current, self.best):
            self.best = current
            self.stopping_step = 0
        else:
            self.stopping_step += 1
            if self.stopping_step >= self.patience:
                self.simulator.should_stop = True

    def _is_improvement(self, monitor_value, reference_value) -> bool:
        return self.monitor_operation(monitor_value - self.min_delta, reference_value)
