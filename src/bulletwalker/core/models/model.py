import numpy as np
import pybullet
from typing import Dict
from bulletwalker import logging as log
from bulletwalker.data.joint_info import JointInfo
from bulletwalker.data.model_state import ModelState
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
        self.velocities = kwargs.get("velocities", dict())
        self.forces = kwargs.get("forces", dict())

        # Ensure that velocities is a dictionary of callbacks that take a float as input and return an array of shape (6,)
        if not isinstance(self.velocities, dict):
            raise TypeError(
                f"Invalid type for velocities: {type(self.velocities)}. Expecting dictionary"
            )
        for key, value in self.velocities.items():
            if not callable(value):
                raise ValueError(
                    f"Invalid type for velocities callback {key}: {type(value)}. Expecting callable"
                )
            velocity = value(0.0)
            if not isinstance(velocity, np.ndarray):
                raise TypeError(
                    f"Invalid type for velocities callback on {key}. Expecting numpy array, but got {type(velocity)}"
                )
            if not velocity.shape == (6,):
                raise ValueError(
                    f"Invalid shape of velocities callback {key}: {velocity.shape}. Expecting shape (6,)"
                )
        # Ensure that force is a dictionary of callbacks that take a float as input and return an array of shape (3,)
        if not isinstance(self.forces, dict):
            raise TypeError(
                f"Invalid type for forces: {type(self.forces)}. Expecting dictionary"
            )
        for key, value in self.forces.items():
            if not callable(value):
                raise ValueError(
                    f"Invalid type for force callback {key}: {type(value)}. Expecting callable"
                )
            force = value(0.0)
            if not isinstance(force, np.ndarray):
                raise TypeError(
                    f"Invalid type for force callback {key}. Expecting numpy array"
                )
            if not force.shape == (3,):
                raise ValueError(
                    f"Invalid shape of force callback {key}: {force.shape}. Expecting shape (3,)"
                )
        self.unfound_links = []  # Keep track of links that were not found in the model

        # Joints are only relevant for robot models
        self.joints: Dict[str, JointInfo] = {}  # Remains empty in non-robot models

    def _validate_kwargs(self, **kwargs):
        valid_kwargs = (
            "position",
            "orientation",
            "velocities",
            "joints",
            "forces",
        )
        for key in kwargs:
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

    @abstractmethod
    def step(self, t: float) -> None:
        if self.id < 0:
            raise ValueError("[Model.step] Model ID is not set. Load model first.")

        # Apply velocities
        for key, value in self.velocities.items():
            try:
                link_id = (
                    [item for item in self.get_link_ids() if item[0] == key][0][1]
                    if key != "base"
                    else -1
                )
                velocity = value(t)
                if not velocity.shape == (6,):
                    raise ValueError(
                        f"Invalid shape of velocity {key}: {velocity.shape}. Expecting shape (6,)"
                    )

                if not np.equal(velocity, np.zeros(6)).all():
                    log.debug(
                        f"Applying velocity {velocity} to link {key} of model {self.name} ({self.id})"
                    )
                    pybullet.resetBaseVelocity(self.id, velocity[:3], velocity[3:6])
            except IndexError:
                if key not in self.unfound_links:
                    self.unfound_links.append(key)
                    log.warning(
                        f"Link {key} not found in model {self.name}. Skipping velocity application."
                    )

        # Apply forces
        for key, value in self.forces.items():
            try:
                link_id = (
                    [item for item in self.get_link_ids() if item[0] == key][0][1]
                    if key != "base"
                    else -1
                )
                force = value(t)
                if not force.shape == (3,):
                    raise ValueError(
                        f"Invalid shape of force {key}: {force.shape}. Expecting shape (3,)"
                    )
                pybullet.applyExternalForce(
                    self.id,
                    link_id,
                    force,
                    [0, 0, 0],
                    pybullet.LINK_FRAME,
                )
            except IndexError:
                if key not in self.unfound_links:
                    self.unfound_links.append(key)
                    log.warning(
                        f"Link {key} not found in model {self.name}. Skipping force application."
                    )

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
        self.position = position

        if call_pybullet and self.id >= 0:
            log.debug(
                f"Setting position of model {self.name} ({self.id}) to {self.position}"
            )
            pybullet.resetBasePositionAndOrientation(
                self.id, self.position, self.orientation.elements
            )

    def reset_orientation(
        self, orientation: Sequence | Quaternion = None, call_pybullet: bool = False
    ) -> None:
        if orientation is None:
            orientation = Quaternion.Identity()
        else:
            if isinstance(orientation, Quaternion):
                orientation = orientation
            else:
                try:
                    orientation = Quaternion(*np.array(orientation, dtype=float))
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
            log.debug(
                f"Setting orientation of model {self.name} ({self.id}) to {self.orientation}"
            )
            pybullet.resetBasePositionAndOrientation(
                self.id, self.position, self.orientation.elements
            )

    def reset_velocity(
        self,
        linear_velocity: Sequence = None,
        angular_velocity: Sequence = None,
        call_pybullet: bool = False,
    ) -> None:
        if linear_velocity is None:
            linear_velocity = np.zeros(3)
        else:
            try:
                linear_velocity = np.array(linear_velocity, dtype=float)
                if not linear_velocity.shape == (3,):
                    raise ValueError(
                        f"Invalid shape of linear velocity: {linear_velocity.shape}. Expecting shape (3,)"
                    )
            except ValueError:
                raise ValueError(
                    "Invalid type for linear velocity. Expecting sequence of floats (3)"
                )
        self.linear_velocity = linear_velocity

        if angular_velocity is None:
            angular_velocity = np.zeros(3)
        else:
            try:
                angular_velocity = np.array(angular_velocity, dtype=float)
                if not angular_velocity.shape == (3,):
                    raise ValueError(
                        f"Invalid shape of angular velocity: {angular_velocity.shape}. Expecting shape (3,)"
                    )
            except ValueError:
                raise ValueError(
                    f"Invalid type for angular velocity: {type(angular_velocity)} ({angular_velocity.shape}). Expecting sequence of floats (3)"
                )
        self.angular_velocity = angular_velocity

        if call_pybullet and self.id >= 0:
            log.debug(
                f"Setting velocity of model {self.name} ({self.id}) to linear: {linear_velocity} and angular: {angular_velocity}"
            )
            pybullet.resetBaseVelocity(self.id, linear_velocity, angular_velocity)

    def reset_pose(
        self,
        position: Sequence[float] = None,
        orientation: Sequence[float] | Quaternion = None,
    ) -> None:
        self.reset_position(position)
        self.reset_orientation(orientation)

    # def apply_initial_force(
    #     self, force: Sequence[float] = None, multiplier: float = 1.0
    # ) -> None:
    #     if force is None:
    #         log.warning("No initial force provided. Applying zero force")
    #         force = np.zeros(6)
    #     else:
    #         try:
    #             force = np.array(force, dtype=float)
    #             if not force.shape == (3,):
    #                 raise ValueError(
    #                     f"Invalid shape of force: {force.shape}. Expecting shape (3,)"
    #                 )
    #         except ValueError:
    #             raise ValueError(
    #                 "Invalid type for force. Expecting sequence of floats (3)"
    #             )
    #     log.debug(
    #         f"Applying initial force {self.force} to model {self.name} ({self.id})"
    #     )
    #     pybullet.applyExternalForce(
    #         self.id,
    #         -1,
    #         np.array(self.force) * multiplier,
    #         [0, 0, 0],
    #         pybullet.LINK_FRAME,
    #     )

    @abstractmethod
    def get_model_state(self) -> ModelState:
        pass

    def get_info(self) -> None:
        pass

    def get_link_names(self) -> list[str]:
        return [
            pybullet.getJointInfo(self.id, i)[12].decode("utf-8")
            for i in range(pybullet.getNumJoints(self.id))
        ]

    def get_link_ids(self) -> list[(str, int)]:
        return [
            (pybullet.getJointInfo(self.id, i)[12].decode("utf-8"), i)
            for i in range(pybullet.getNumJoints(self.id))
        ]
