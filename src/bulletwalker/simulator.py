import pybullet
import pathlib
import numpy as np
import time
import pybullet_data
from bulletwalker import logging as log
from bulletwalker.core.models.model import Model
from bulletwalker.core.models.terrain import Terrain, PlaneTerrain
from bulletwalker.core.callbacks import Callback
from bulletwalker.data.simulation_step import SimulationStep
from bulletwalker.data.model_state import ModelState
from bulletwalker import logging as log
from typing import Union, Tuple, List, Dict
from datetime import timedelta
from dataclasses import dataclass


@dataclass
class PhysicsEngineParameters:
    timestep: float = 1.0 / 240


class Simulator:
    def __init__(
        self,
        name: str = "Simulator",
        gravity: Tuple[float, float, float] = (0, 0, -9.81),
        use_gui: bool = False,
        real_time: bool = False,
        use_gl2: bool = False,
        models: List[Model] = [],
    ) -> None:
        self.name = name if name else "Simulator"
        self.gravity = np.array(gravity)
        if use_gui:
            self.client: int = pybullet.connect(
                pybullet.GUI, options="--opengl2" if use_gl2 else ""
            )
        else:
            self.client: int = pybullet.connect(pybullet.DIRECT)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setGravity(
            self.gravity[0],
            self.gravity[1],
            self.gravity[2],
            physicsClientId=self.client,
        )
        self.real_time: bool = real_time
        if real_time:
            pybullet.setRealTimeSimulation(1, physicsClientId=self.client)

        self.models: List[Model] = models if models else []
        self._validate_model_input()
        self.terrain = None

        self.history: List[SimulationStep] = []
        self.reset()
        self.state = ""

    def _validate_model_input(self):
        model_names = []
        for i, model in enumerate(self.models):
            if not isinstance(model, Model):
                raise TypeError("Model must be an instance of the Model class")
            if model.name in model_names:
                log.warning(
                    f"Two models share the same name: {model.name}. Renaming model at index {i} to {model.name}_{i}"
                )
                model.name = f"{model.name}_{i}"

            model_names.append(model.name)

    def add_model(self, model: Model) -> None:
        if not isinstance(model, Model):
            raise TypeError("Model must be an instance of the Model class")
        if not model.id == -1:
            log.warning(
                f"Model {model.name} already exists in the simulator. Overwriting the current model."
            )
        self.models.append(model)

    def add_terrain(self, terrain: Terrain = None) -> None:
        # Pass either a built terrain, a resource or a path to a terrain file
        # For now just load a plane
        if terrain is None:
            terrain = PlaneTerrain()
        else:
            if not isinstance(terrain, Terrain):
                raise TypeError(
                    f"Terrain must be an instance of the Terrain class, but got {type(terrain)}"
                )

        if self.terrain is not None:
            log.warning(
                "Terrain already exists in the simulator. Overwriting the current terrain."
            )

        self.terrain = terrain

    def load_models(self):
        if self.terrain is not None:
            self.terrain.load(pybullet.loadURDF(self.terrain.urdf_path))
            self.terrain.reset_position(self.terrain.position)
            self.terrain.reset_orientation(self.terrain.orientation)

            pybullet.resetBasePositionAndOrientation(
                self.terrain.id, self.terrain.position, self.terrain.orientation
            )

        if len(self.models) == 0:
            log.warning(
                "No models to load into the simulator. If you defined some models, make sure you add them before calling load_models()"
            )
            return

        for model in self.models:
            model.load(pybullet.loadURDF(model.urdf_path))
            log.debug(
                f"Loaded model {model.name} with id {model.id} into simulator {self.name}"
            )
            model.reset_position(model.position)
            model.reset_orientation(model.orientation)
            # print(
            #     f"Resetting model {model.name} to position {model.position} and orientation {model.orientation}"
            # )

            pybullet.resetBasePositionAndOrientation(
                model.id, model.position, model.orientation
            )

    def get_model_states(self) -> Dict[str, ModelState]:
        model_states = {}
        for model in self.models:
            model_states[model.name] = model.get_model_state()
        return model_states

    def step(self, index: int = 0, callbacks=[]) -> SimulationStep:
        for model in self.models:
            model.step()
        pybullet.stepSimulation(physicsClientId=self.client)

        # Build simulation step
        step = SimulationStep(
            index,
            self.t,
            time.time() - self.sim_start if self.sim_start else 0.0,
            0.0,
            self.get_model_states(),
        )

        for callback in callbacks:
            callback.on_simulation_step(step)

        return step

    def reset(self) -> None:
        self.should_stop = False
        self.t: float = 0.0
        self._iter: int = 0
        self.history = []
        self.sim_start = None
        self.running = False

    def run(
        self,
        dt: float = 1.0 / 240,
        tf: float = 2.0,
        frozen_run: bool = False,
        max_iterations=-1,
        callbacks: List[Callback] = [],
    ) -> None:
        if dt <= 0:
            raise ValueError("Time step must be greater than zero.")
        if dt > 0.1:
            log.warning(
                "Time step is greater than 0.1 seconds. Simulations may produce unrealistic results."
            )
        elif dt < 1e-4:
            log.warning(
                "Time step is smaller than 1e-4 seconds. Simulations may be slow or may diverge."
            )
        pybullet.setTimeStep(dt, physicsClientId=self.client)
        self.reset()
        self.sim_start = time.time()
        self.running = True
        log.info(f"Starting simulation {self.name}")
        for callback in callbacks:
            callback.on_simulation_start()

        if frozen_run:
            while True:
                pass

        while self.running:
            if self.should_stop:
                self.running = False
                break
            # tik = time.time()
            step = self.step(index=self._iter, callbacks=callbacks)
            self.history.append(step)
            # tok = time.time()

            self.t += dt
            if self.t >= tf:
                self.running = False
            self._iter += 1
            if max_iterations > 0 and self._iter > max_iterations:
                self.running = False
            time.sleep(dt)
        end = time.time()
        for callback in callbacks:
            callback.on_simulation_end()
        log.info(
            f"Simulation finished in {timedelta(seconds=time.time()-self.sim_start)} ({end-self.sim_start:.2f} seconds) with {self.t:.2f} simulated seconds."
        )

    def close(self) -> None:
        log.debug("Closing simulator")
        pybullet.disconnect(physicsClientId=self.client)
