from typing import Dict
from dataclasses import dataclass
from .model_state import ModelState


@dataclass
class SimulationStep:
    index: int
    sim_time: float
    real_time: float
    loss: float
    model_states: Dict[str, ModelState]
