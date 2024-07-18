import numpy as np
import pathlib
from bulletwalker.core.models.model import Model
from abc import ABC, abstractmethod


class Terrain(Model, ABC):
    @abstractmethod
    def __init__(self, terrain_urdf: str, **kwargs):
        self.init(terrain_urdf, "PlaneTerrain", **kwargs)


class PlaneTerrain(Terrain):
    def __init__(self, path: str, rotation: np.ndarray = np.array([0, 0, 0, 1])):
        # Use built-in resource later on
        super().__init__(str(path))
        self.reset_orientation(rotation)


class HeightmapTerrain(Terrain):
    pass


class RandomTerrain(Terrain):
    pass
