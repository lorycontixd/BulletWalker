import numpy as np
import pybullet
import pathlib
from abc import ABC, abstractmethod
from typing import List


class Model(ABC):
    @abstractmethod
    def __init__(self, urdf_path: str | pathlib.Path, name: str, **kwargs):
        raise ValueError("Abstract class Model cannot be instantiated")

    def init(self, urdf_path: str | pathlib.Path, name: str, **kwargs) -> None:
        self.name = name if name else "Model"
        self.urdf_path = urdf_path if isinstance(urdf_path, str) else str(urdf_path)
        self.id = pybullet.loadURDF(urdf_path)
        self.dof = pybullet.getNumJoints(self.id)

        valid_kwargs = (
            "initial_position",
            "initial_rotation",
            "initial_velocity",
        )
        for key in kwargs:
            if key not in valid_kwargs:
                raise ValueError(f"Invalid keyword argument: {key}")
        self.reset_position(kwargs.get("initial_position"))
        self.reset_orientation(kwargs.get("initial_rotation"))

    def reset_position(self, position: np.ndarray | List[float] = None) -> None:
        if position is None:
            position = np.zeros(3)
        if not position.shape == (3,):
            raise ValueError(
                f"Invalid shape of initial position: {position.shape}. Expecting shape (3,)"
            )
        pybullet.resetBasePositionAndOrientation(self.id, position, [0, 0, 0, 1])

    def reset_orientation(self, orientation: np.ndarray | List[float] = None) -> None:
        if orientation is None:
            orientation = np.array([0, 0, 0, 1])
        if not orientation.shape == (4,):
            raise ValueError(
                f"Invalid shape of initial orientation: {orientation.shape}. Expecting shape (4,)"
            )
        pybullet.resetBasePositionAndOrientation(self.id, [0, 0, 0], orientation)

    def reset_pose(
        self,
        position: np.ndarray | List[float] = None,
        orientation: np.ndarray | List[float] = None,
    ) -> None:
        self.reset_position(position)
        self.reset_orientation(orientation)
