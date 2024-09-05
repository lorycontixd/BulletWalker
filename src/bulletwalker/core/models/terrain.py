import numpy as np
import trimesh
import os
import pathlib
import pybullet
from dataclasses import dataclass
from bulletwalker.core.models.model import Model
from bulletwalker.core.math.quaternion import Quaternion
from bulletwalker.data.dynamics_info import DynamicsInfo
from bulletwalker.data.model_state import ModelState
from bulletwalker.data.joint_state import JointState
from abc import ABC, abstractmethod


@dataclass
class TerrainDynamicParameters:
    lateral_friction: float = 1.0
    spinning_friction: float = 1.0
    restitution: float = 0.0
    linear_damping: float = 0.04
    angular_damping: float = 0.04

    def __post_init__(self):
        self.lateral_friction = max(0.0, self.lateral_friction)
        self.spinning_friction = max(0.0, self.spinning_friction)
        self.restitution = max(0.0, self.restitution)
        self.linear_damping = max(0.0, self.linear_damping)
        self.angular_damping = max(0.0, self.angular_damping)


class Terrain(Model, ABC):
    @abstractmethod
    def __init__(
        self,
        terrain_urdf: str,
        dynamic_parameters: TerrainDynamicParameters,
        **kwargs,
    ):
        super().__init__(self.__class__.__name__, terrain_urdf, **kwargs)
        self.velocity = np.zeros(3)
        self.dynamic_parameters = dynamic_parameters

    def load(self, index: int):
        super().load(index)
        self._set_dynamics()

    def _set_dynamics(self):
        pybullet.changeDynamics(
            self.id,
            -1,
            lateralFriction=self.dynamic_parameters.lateral_friction,
            spinningFriction=self.dynamic_parameters.spinning_friction,
            restitution=self.dynamic_parameters.restitution,
            linearDamping=self.dynamic_parameters.linear_damping,
            angularDamping=self.dynamic_parameters.angular_damping,
        )

    def get_model_state(model: Model) -> ModelState:
        _pos, _or = pybullet.getBasePositionAndOrientation(model.id)
        return ModelState(
            base_position=np.array(_pos),
            base_orientation=Quaternion(_or),
            joint_states=[],
        )

    def step(self, t: float) -> None:
        pass


class PlaneTerrain(Terrain):
    def __init__(
        self,
        dynamic_parameters: TerrainDynamicParameters = TerrainDynamicParameters(),
        **kwargs,
    ):
        """Initialize a PlaneTerrain object

        Args:
            path (str): Path to the URDF file of the terrain
            rotation (np.ndarray | Quaternion, optional): Initial rotation of the terrain. Defaults to None.

        Raises:
            ValueError: Invalid shape of initial rotation
            ValueError: Invalid type for rotation
        Note:
            The rotation can be passed as a numpy array of shape (3,) representing an euler rotation or as a Quaternion object
        """
        # Use built-in resource later on
        path = str(
            pathlib.Path(__file__).parents[2]
            / "assets"
            / "urdfs"
            / "terrains"
            / "plane_terrain.urdf"
        )
        super().__init__(path, dynamic_parameters, **kwargs)


class HeightmapTerrain(Terrain):
    def __init__(self) -> None:
        pass


class RandomTerrain(Terrain):
    def __init__(self):
        """Initialize a RandomTerrain object

        Args:
            size (np.ndarray): Size of the terrain in meters
            seed (int, optional): Seed for the random generation. Defaults to 0.
            disturbance (float, optional): Magnitude of the disturbance. Defaults to 1.0.
        """
        pass
