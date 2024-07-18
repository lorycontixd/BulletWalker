import pybullet
import pathlib
import numpy as np
import time
from bulletwalker import logging as log
from bulletwalker.core.models.model import Model
from bulletwalker.core.models.terrain import Terrain, PlaneTerrain
from bulletwalker.core.callbacks import Callback
from typing import Union, Tuple, List
from datetime import timedelta
from dataclasses import dataclass


@dataclass
class SimulationStep:
    index: int
    time: float
    score: float
    loss: float
    base_position: np.ndarray


class Simulator:
    def __init__(
        self,
        name: str = "Simulator",
        use_gui: bool = False,
        real_time: bool = False,
        gravity: Tuple[float, float, float] = (0, 0, -9.81),
        models: List[Model] = [],
    ) -> None:
        self.name = name if name else "Simulator"
        self.gravity = np.array(gravity)
        if use_gui:
            self.client: int = pybullet.connect(pybullet.GUI)
        else:
            self.client: int = pybullet.connect(pybullet.DIRECT)

        pybullet.setGravity(
            self.gravity[0],
            self.gravity[1],
            self.gravity[2],
            physicsClientId=self.client,
        )
        if real_time:
            pybullet.setRealTimeSimulation(1, physicsClientId=self.client)

        self.models: List[Model] = models
        self.terrain = None

        self.history: List[SimulationStep] = []
        self.running = False
        self.should_stop = False

    def add_model(self, model: Model) -> None:
        if not isinstance(model, Model):
            raise TypeError("Model must be an instance of the Model class")
        self.models.append(model)

    def add_terrain(self, terrain: Terrain = None) -> None:
        # Pass either a built terrain, a resource or a path to a terrain file
        # For now just load a plane
        if terrain is None:
            path = (
                pathlib.Path(__file__).parents[1]
                / "assets"
                / "urdfs"
                / "terrains"
                / "plane_terrain.urdf"
            )
            terrain = PlaneTerrain(path, rotation=np.array([0, 0, 0, 1]))
        if self.terrain is not None:
            log.warning(
                "Terrain already exists in the simulator. Overwriting the current terrain."
            )
        self.terrain = terrain

    def step(self):
        pybullet.stepSimulation(physicsClientId=self.client)

    def reset(self) -> None:
        self.should_stop = False
        self.t: float = 0.0

    def run(
        self, dt: float = 0.001, tf: float = 2.0, callbacks: List[Callback] = []
    ) -> None:
        self.reset()
        start = time.time()
        self.running = True
        log.info(f"Starting simulation {self.name}")
        for callback in callbacks:
            callback.on_simulation_start(self)
        while self.running:
            if self.should_stop:
                self.running = False
                break
            tik = time.time()
            self.step()
            tok = time.time()
            for callback in callbacks:
                callback.on_simulation_step(
                    self, SimulationStep(index=0, time=self.t, score=0.0, loss=0.0)
                )

            self.t += dt
            if self.t >= tf:
                self.running = False
            time.sleep(dt)
        end = time.time()
        log.info(
            f"Simulation finished in {timedelta(seconds=time.time()-start)} ({end-start:.2f} seconds) with {self.t:.2f} simulated seconds."
        )

    def close(self) -> None:
        pybullet.disconnect(physicsClientId=self.client)
