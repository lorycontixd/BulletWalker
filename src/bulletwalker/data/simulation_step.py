from typing import Dict
from dataclasses import dataclass
from .model_state import ModelState


@dataclass
class SimulationStep:
    """
    Data class for a single simulation step.

    Args:
        index (int): Index of the simulation step
        sim_time (float): Simulation time
        real_time (float): Real time
        loss (float): Loss value
        model_states (Dict[str, ModelState]): Dictionary of model states
    """

    index: int
    sim_time: float
    real_time: float
    loss: float
    model_states: Dict[str, ModelState]
