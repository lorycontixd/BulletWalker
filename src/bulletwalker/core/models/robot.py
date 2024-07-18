import pathlib
import pybullet
import numpy as np
from typing import List
from .model import Model


class Robot(Model):
    def __init__(
        self,
        urdf_path: str | pathlib.Path,
        name: str = "Robot Model",
        **kwargs,
    ) -> None:
        self.init(urdf_path, name, **kwargs)
        self.reset_joints(kwargs.get("initial_joints"))

    def reset_joints(self, poses: np.ndarray | List[float] = None) -> None:
        if poses is None:
            poses = np.zeros(self.dof)
        if not poses.shape == (self.dof,):
            raise ValueError(
                f"Invalid shape of initial joint poses: {poses.shape}. Expecting shape ({self.dof},)"
            )
        pass

    def get_total_mass(self) -> float:
        total_mass: float = 0
        base_mass = pybullet.getDynamicsInfo(self.id, -1)[0]
        total_mass += base_mass

        for i in range(self.dof):
            joint_mass = pybullet.getDynamicsInfo(self.id, i)[0]
            total_mass += joint_mass
        return total_mass
