import pybullet
import enum
from dataclasses import dataclass


class ControlMetric(enum.IntEnum):
    POSITION = pybullet.POSITION_CONTROL
    VELOCITY = pybullet.VELOCITY_CONTROL
    TORQUE = pybullet.TORQUE_CONTROL
    PD = pybullet.PD_CONTROL


@dataclass
class JointData:
    """Interface only to set initial joint values"""

    name: str
    initial_position: float = 0
    initial_velocity: float = 0
    control_metric: ControlMetric = ControlMetric.POSITION
