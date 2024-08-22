import pybullet
import numpy as np
import dataclasses
from dataclasses import dataclass
from bulletwalker.data.joint_state import JointState
from bulletwalker.core.math.quaternion import Quaternion


@dataclass
class ModelState:
    base_position: np.ndarray
    base_orientation: Quaternion
    base_linear_velocity: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(3)
    )
    base_angular_velocity: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(3)
    )
    joint_states: dict[str, JointState] = dataclasses.field(default_factory=dict)
