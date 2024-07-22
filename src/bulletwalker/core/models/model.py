import numpy as np
import pybullet
from typing import Dict
from bulletwalker import logging as log
from bulletwalker.data.joint_info import JointInfo
from bulletwalker.core.math.quaternion import Quaternion
from abc import ABC, abstractmethod
from typing import Sequence


class Model(ABC):
    @abstractmethod
    def __init__(self, name: str, urdf_path: str, **kwargs):
        self.name = name
        self.id = -1
        self.dofs = -1  # Remains -1 in non-robot models
        self.urdf_path = urdf_path if isinstance(urdf_path, str) else str(urdf_path)
        self._validate_kwargs(**kwargs)
        self.position = kwargs.get("position", np.zeros(3))
        self.orientation = kwargs.get("orientation", Quaternion.Identity())
        self.velocity = kwargs.get("velocity", np.zeros(3))
        self.joints: Dict[str, JointInfo] = {}  # Remains empty in non-robot models

    def _validate_kwargs(self, **kwargs):
        valid_kwargs = ("position", "orientation", "velocity", "joints")
        for key in kwargs:
            print(f"Checking key {key} for model {self.name}")
            if key not in valid_kwargs:
                raise ValueError(
                    f"Invalid keyword argument {key}. Valid arguments: {valid_kwargs}"
                )

    @abstractmethod
    def load(self, model_id: int) -> None:
        if model_id < 0:
            raise ValueError("Invalid ID for model")
        self.id = model_id
        log.info(f"Loaded model {self.name} with ID {self.id}")

    def reset_position(
        self, position: Sequence = None, call_pybullet: bool = False
    ) -> None:
        if position is None:
            log.warning("No initial position provided. Setting position to [0, 0, 0]")
            position = np.zeros(3)
        else:
            try:
                position = np.array(position, dtype=float)
            except ValueError:
                raise ValueError(
                    "Invalid type for position. Expecting sequence of floats (3)"
                )
        if not position.shape == (3,):
            raise ValueError(
                f"Invalid shape of initial position: {position.shape}. Expecting shape (3,)"
            )

        print(f"Setting robot {self.name} ({self.id}) to position {position}")
        self.position = position

        if call_pybullet and self.id >= 0:
            pybullet.resetBasePositionAndOrientation(
                self.id, self.position, self.orientation.elements
            )

    def reset_orientation(
        self, orientation: Sequence | Quaternion = None, call_pybullet: bool = False
    ) -> None:
        if orientation is None:
            orientation = Quaternion.Identity().elements
        else:
            if isinstance(orientation, Quaternion):
                orientation = orientation.elements
            else:
                try:
                    orientation = np.array(orientation, dtype=float)
                    if not orientation.shape == (4,):
                        raise ValueError(
                            f"Invalid shape of orientation: {orientation.shape}. Expecting shape (4,)"
                        )
                except ValueError:
                    raise ValueError(
                        "Invalid type for orientation. Expecting sequence of floats (4)"
                    )
        self.orientation = orientation

        if call_pybullet and self.id >= 0:
            pybullet.resetBasePositionAndOrientation(
                self.id, self.position, self.orientation.elements
            )

    def reset_pose(
        self,
        position: Sequence[float] = None,
        orientation: Sequence[float] | Quaternion = None,
    ) -> None:
        self.reset_position(position)
        self.reset_orientation(orientation)
