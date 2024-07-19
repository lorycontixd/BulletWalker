from dataclasses import dataclass
import numpy as np
import enum
from bulletwalker import logging as log


class BodyType(enum.IntEnum):
    RIGIDBODY = 1
    MULTIBODY = 2
    SOFTBODY = 3


class DynamicsInfo:
    def __init__(self, *args):
        if len(args) == 1:
            info = args[0]
            if not isinstance(info, tuple):
                raise TypeError(
                    f"[DynamicsInfo] Passed single argument must be a tuple, not {type(info)}"
                )
            # The tuple retured by pybyllet.getDynamicsInfo
            self.mass = float(info[0])
            self.lateral_friction = float(info[1])
            self.local_inertial_diagonal = np.array(info[2])
            self.local_inertial_position = np.array(info[3])
            self.local_inertial_orn = np.array(info[4])
            self.restitution = float(info[5])
            self.rolling_friction = float(info[6])
            self.spinning_friction = float(info[7])
            self.contact_damping = float(info[8])
            self.contact_stiffness = float(info[9])
            self.body_type_int = int(info[10])
            self.body_type = BodyType(info[10])
            self.collision_margin = float(info[11])
        else:
            log.error(f"[DynamicsInfo] Invalid arguments. Expected 1, got {len(args)}")

    def __str__(self):
        return f"DynamicsInfo: {self.mass=}, {self.lateral_friction=}, {self.local_inertial_diagonal=}, {self.local_inertial_position=}, {self.local_inertial_orn=}, {self.restitution=}, {self.rolling_friction=}, {self.spinning_friction=}, {self.contact_damping=}, {self.contact_stiffness=}, {self.body_type=}, {self.collision_margin=}"
