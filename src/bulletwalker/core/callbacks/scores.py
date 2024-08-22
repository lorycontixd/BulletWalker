from abc import ABC, abstractmethod
import numpy as np
from bulletwalker import logging as log
from bulletwalker.data.simulation_step import SimulationStep
from bulletwalker.data.score_result import ScoreResult


class ScoreFunction(ABC):
    """Score function base class.
    The score returned by the score function should be a normalized value which should be minimized.
    """

    @abstractmethod
    def __init__(self, multiplier: float = 1.0):
        self.multiplier = multiplier
        self.score_func_weight = 1.0
        self.tracked_models: list = []

    @abstractmethod
    def calculate_score(
        self,
        simulation_step: SimulationStep,
        tracked_models: list = None,
        *args,
        **kwargs,
    ):
        # if self.tracked_models is None or len(self.tracked_models) == 0:
        #    self.tracked_models = simulation_step.model_states.keys()
        if tracked_models is None:
            tracked_models = self.tracked_models

    def set_tracked_models(self, tracked_models: list):
        self.tracked_models = tracked_models

    def __add__(self, other):
        if not isinstance(other, ScoreFunction):
            raise ValueError(
                f"Cannot add ScoreFunction object with object of type {type(other)}"
            )
        return CombinedScoreFunction([self, other])


class RootUpwardsScore(ScoreFunction):
    """Score function that calculates a score based on the root upwards value of the robot, with respect to a reference value."""

    def __init__(
        self, root_up_reference_value: float, a: float, multiplier: float = 1.0
    ):
        super().__init__(multiplier=multiplier)
        self.root_up_reference_value = root_up_reference_value
        if a <= 0:
            log.warning(
                f"Value of a should be strictly greater than 0, but received value: {a}. Setting a to 1.0"
            )
            a = 1.0
        self.a = a

    def calculate_score(
        self, simulation_step: SimulationStep, tracked_models: list
    ) -> np.ndarray:
        super().calculate_score(simulation_step, tracked_models)
        if tracked_models is None:
            tracked_models = list(simulation_step.model_states.keys())
        root_up_values = np.array(
            [
                simulation_step.model_states[m].base_position[2]
                for m in simulation_step.model_states
                if m in tracked_models or len(tracked_models) == 0
            ]
        )
        # vals = self.multiplier * np.exp(
        #     np.negative(self.a * np.abs(root_up_values - self.root_up_reference_value))
        # )
        vals = self.multiplier * np.exp(
            np.negative(
                (1.0 / self.a) * (root_up_values - self.root_up_reference_value) ** 2
            )
        )

        scores = dict(zip(tracked_models, vals))

        return ScoreResult(
            scores=scores,
            contributions=None,
        )


class ForwardScore(ScoreFunction):
    """Score function that calculates a score based on the forward value of the robot, with respect to a reference value."""

    def __init__(
        self,
        forward_reference_value: float,
        a: float,
        b: float,
        multiplier: float = 1.0,
    ):
        super().__init__(multiplier=multiplier)
        self.forward_reference_value = forward_reference_value
        if a <= 0:
            log.warning(
                f"Value of a should be strictly greater than 0, but received value: {a}. Setting a to 1.0"
            )
            a = 1.0
        self.a = a
        self.b = b

    def calculate_score(
        self, simulation_step: SimulationStep, tracked_models: list = None
    ):
        super().calculate_score(simulation_step, tracked_models)
        if tracked_models is None:
            tracked_models = list(simulation_step.model_states.keys())
        scores = {
            # m: (2.0 * self.multiplier / np.pi)
            # * np.arctan(
            #     np.abs(
            #         simulation_step.model_states[m].base_position[0]
            #         - self.forward_reference_value
            #     )
            # )
            m: self.multiplier
            * self.b
            * (
                simulation_step.model_states[m].base_position[0]
                - self.forward_reference_value
            )
            ** (2 * self.a)
            for m in simulation_step.model_states
            if m in tracked_models or len(tracked_models) == 0
        }
        return ScoreResult(
            scores=scores,
            contributions=None,
        )


class VelocityChangeScore(ScoreFunction):
    def __init__(self, velocity_delta: float = 0.01):
        super().__init__()
        self.velocity_delta = velocity_delta
        self.reset()
        self.last_angular_velocities = {}
        self.last_time = None

    def reset(self):
        self.score = 0

    def calculate_score(
        self, simulation_step: SimulationStep, tracked_models: list = None
    ):
        super().calculate_score(simulation_step, tracked_models)
        if tracked_models is None:
            tracked_models = list(simulation_step.model_states.keys())
        if self.last_angular_velocities is not None:
            scores = {}
            for model in tracked_models:
                if model not in self.last_angular_velocities:
                    log.warning(
                        f"Model {model} not found in last angular velocities. Skipping..."
                    )
                    continue
                current_angular_velocity = simulation_step.model_states[
                    model
                ].base_angular_velocity
                if current_angular_velocity * self.last_angular_velocities[model] < 0:
                    scores[model] = 1.0
                else:
                    scores[model] = 0.0
            self.last_angular_velocities = {
                model: simulation_step.model_states[model].base_angular_velocity
                for model in tracked_models
            }
            self.last_time = simulation_step.time
        else:
            self.last_angular_velocities = {
                model: simulation_step.model_states[model].base_angular_velocity
                for model in tracked_models
            }
            self.last_time = simulation_step.time
            scores = dict.fromkeys(tracked_models, 0.0)
        return ScoreResult(
            scores=scores,
            contributions=None,
            total_score=np.mean(list(scores.values())),
        )


class CombinedScoreFunction(ScoreFunction):
    """Score function that combines multiple score functions into a single score function."""

    def __init__(self, score_functions: list = [], weights: list = []):
        super().__init__()
        self.score_functions = score_functions
        try:
            self.weights = np.array(weights)
        except Exception as e:
            log.error(
                f"Error while setting weights for CombinedScoreFunction: {e}. Setting all weights to 1.0"
            )
            self.weights = np.ones(len(score_functions)) / len(score_functions)
        if len(self.weights) == 0:
            log.warning(
                f"[CombinedScoreFunction] No weights provided while {len(score_functions)} score funcs were provided. Setting same weights for all score functions"
            )
            self.weights = (1.0 / len(score_functions)) * np.ones(len(score_functions))
        else:
            if not np.isclose(np.sum(self.weights), 1.0):
                log.warning(
                    "[CombinedScoreFunction] Weights provided do not sum to 1.0. Normalizing the weights."
                )
                self.weights = self.weights / np.sum(self.weights)

    def calculate_score(
        self,
        simulation_step: SimulationStep,
        tracked_models: list = None,
    ):
        super().calculate_score(simulation_step, tracked_models)
        if tracked_models is None:
            tracked_models = list(simulation_step.model_states.keys())
        # Return a normalized weighted sum of the scores
        models_scores = dict.fromkeys(tracked_models, 0.0)
        contributions = dict.fromkeys(tracked_models, {})

        for i0, model in enumerate(tracked_models):
            contributions[model] = {}
            for i1, score_func in enumerate(self.score_functions):
                score = score_func.calculate_score(simulation_step, [model])
                contributions[model][score_func.__class__.__name__] = (
                    score.scores[model] * self.weights[i1]
                )
                models_scores[model] += score.scores[model] * self.weights[i1]

        return ScoreResult(
            scores=models_scores,
            contributions=contributions,
        )

    def __add__(self, other) -> "CombinedScoreFunction":
        if isinstance(other, ScoreFunction):
            self.score_functions.append(other)
            self.weights.append(1.0)
            return self
        elif isinstance(other, CombinedScoreFunction):
            self.score_functions.extend(other.score_functions)
            self.weights.extend(other.weights)
            return self
        else:
            raise TypeError(
                f"Cannot add CombinedScoreFunction object with object of type {type(other)}"
            )

    def set_weights(self, weights: list):
        if len(weights) != len(self.score_functions):
            raise ValueError(
                f"Number of weights provided ({len(weights)}) does not match the number of score functions ({len(self.score_functions)})"
            )
        self.weights = weights
        return self
