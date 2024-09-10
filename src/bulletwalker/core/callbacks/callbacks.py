import enum
import numpy as np
import pathlib
from typing import List, Dict
from bulletwalker import logging as log
from bulletwalker.core.models.model import Model
from .scores import ScoreFunction
from abc import ABC, abstractmethod


class Callback(ABC):
    """Abstract class for simulation callbacks."""

    @abstractmethod
    def __init__(self, simulator):
        self.simulator = simulator

    @abstractmethod
    def on_simulation_start(self):
        pass

    @abstractmethod
    def on_simulation_step(self, simulation_step):
        pass

    @abstractmethod
    def on_simulation_end(self):
        pass


class ScoreCallback(Callback):
    """Callback to calculate a score for each simulation step.
    The score is calculated using a ScoreFunction object that is passed as an argument.
    If the direction is set to "minimize", the score will be initialized to np.inf and

    Args:
        Callback ([type]): [description]
    """

    def __init__(
        self,
        simulator,
        score_function: ScoreFunction,
        tracked_models: list = None,
        direction="minimize",
        multiplier: float = 1.0,
    ):
        super().__init__(simulator)
        self.score_function = score_function
        self.direction = direction
        if (
            tracked_models is None or len(tracked_models) == 0
        ) and simulator is not None:
            tracked_models = [model.name for model in simulator.models]
        self.tracked_models = tracked_models
        self.history = []
        self.multiplier = multiplier

    def on_simulation_start(self):
        self.history = []
        self.score_function.set_tracked_models(self.tracked_models)

    def on_simulation_step(self, simulation_step):
        r = self.score_function.calculate_score(simulation_step, self.tracked_models)
        r.total_score *= self.multiplier
        r.scores = {k: v * self.multiplier for k, v in r.scores.items()}
        r.contributions = {
            k: {kk: vv * self.multiplier for kk, vv in v.items()}
            for k, v in r.contributions.items()
        }
        self.result = r
        self.history.append(self.result)

    def on_simulation_end(self):
        pass

    def reset(self):
        self.history = []
        self.result = {}


class EarlyStoppingCallback(Callback):
    """Callback to stop the simulation early.
    The callback will stop the simulation if the monitored value does not improve for a number of steps,
    or when the monitored value reaches a certain value.
    """

    class EarlyStoppingMode(enum.IntEnum):
        MIN = enum.auto()
        MAX = enum.auto()
        EXACT = enum.auto()
        AUTO = enum.auto()

    def __init__(
        self,
        simulator,
        monitor: str = "index",
        min_delta: float = 0.0,
        mode: EarlyStoppingMode = EarlyStoppingMode.AUTO,
        patience: float = 0.0,
    ) -> None:
        super().__init__(simulator=simulator)

        self.monitor = monitor
        self.mode = mode
        self.min_delta = abs(min_delta)
        self.patience = patience
        self.stopping_step = 0
        self.monitor_operation = None
        self.best = None

        if self.mode == EarlyStoppingCallback.EarlyStoppingMode.MIN:
            self.monitor_operation = np.less
        elif self.mode == EarlyStoppingCallback.EarlyStoppingMode.MAX:
            self.monitor_operation = np.greater
        elif self.mode == EarlyStoppingCallback.EarlyStoppingMode.EXACT:
            self.monitor_operation = np.equal
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
        self.started = False

    def on_simulation_start(self):
        # Reset values
        pass

    def on_simulation_step(self, simulation_step):
        current = self.get_monitor_value(simulation_step)
        print(f"curr: {current}")

    def on_simulation_end(self):
        pass

    def get_monitor_value(self, simulation_step):
        monitor_value = simulation_step.__dict__[self.monitor]
        if monitor_value is None:
            log.warning(f"No value found for monitor {self.monitor} in simulation step")
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)


class PrinterCallback(Callback):
    """Callback to print tracked variables when included in the simulator.

    Args:
        Callback ([type]): [description]

    """

    def __init__(self, simulator, variables):
        super().__init__(simulator)
        self.variables = variables

    def on_simulation_start(self):
        print("Simulation started")

    def on_simulation_step(self, simulation_step):
        for var in self.variables:
            try:
                log.info(
                    f"[PrinterCallback] ==> {var}: {simulation_step.__dict__[var]}"
                )
            except AttributeError:
                log.warning(f"Variable {var} not found in simulation step")

    def on_simulation_end(self):
        print("Simulation ended")


class TrackerCallback(Callback):
    """Callback to track variables in the simulation and store them in a list.

    Args:
        Callback ([type]): [description]

    """

    def __init__(self, simulator, model_names: List[str]):
        super().__init__(simulator)
        self.model_names = model_names
        self.history = {}
        self.t = []

    def on_simulation_start(self):
        self.t = []
        self.history = {model: [] for model in self.model_names}

    def on_simulation_step(self, simulation_step):
        self.t.append(simulation_step.sim_time)
        for model in self.model_names:
            if model not in simulation_step.model_states:
                log.warning(f"Model {model} not found in simulation step")
                continue
            if model not in self.history:
                self.history[model] = []
            self.history[model].append(simulation_step.model_states[model])

    def on_simulation_end(self):
        print(f"[TrackerCallback] {list(self.history.keys())}")

    def reset(self):
        self.history = {}
        self.t = []

    def get_observable_values(self, model_name: str, observable: str):
        if not self.history:
            log.warning("No history found. Run simulation first.")
            return
        if model_name not in self.history:
            log.warning(f"Model {model_name} not found in history")
            return
        x = [state.__dict__[observable] for state in self.history[model_name]]
        return x
