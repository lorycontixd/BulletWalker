import numpy as np
import trimesh
import os
from bulletwalker.core.models.model import Model
from bulletwalker.core.math.quaternion import Quaternion
from abc import ABC, abstractmethod


class Terrain(Model, ABC):
    @abstractmethod
    def __init__(self, name, terrain_urdf: str, **kwargs):
        super().__init__(name, terrain_urdf, **kwargs)
        self.velocity = np.zeros(3)


class PlaneTerrain(Terrain):
    def __init__(self, path: str, **kwargs):
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

        super().__init__(self.__class__.__name__, path, **kwargs)


class HeightmapTerrain(Terrain):
    pass


class RandomTerrain(Terrain):
    def __init__(
        self, size: np.ndarray, seed: int = 0, disturbance: float = 1.0, **kwargs
    ):
        """Initialize a RandomTerrain object

        Args:
            size (np.ndarray): Size of the terrain in meters
            seed (int, optional): Seed for the random generation. Defaults to 0.
            disturbance (float, optional): Magnitude of the disturbance. Defaults to 1.0.
        """
        pass
