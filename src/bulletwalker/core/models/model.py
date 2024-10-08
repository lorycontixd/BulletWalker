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
        self.velocity = kwargs.get("velocity", np.zeros(6))
        self.force = kwargs.get("force", np.zeros(3))

        # Ensure that velocity is a numpy array of shape (6,)
        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity, dtype=float)
        if self.velocity.shape == (3,):
            self.velocity = np.concatenate((self.velocity, np.zeros(3)))
        if not self.velocity.shape == (6,):
            raise ValueError(
                f"Invalid shape of velocity: {self.velocity.shape}. Expecting shape (6,)"
            )

        # Ensure that force is a numpy array of shape (6,)
        if not isinstance(self.force, np.ndarray):
            self.force = np.array(self.force, dtype=float)
        if not self.force.shape == (3,):
            raise ValueError(
                f"Invalid shape of force: {self.force.shape}. Expecting shape (3,)"
            )

        # Joints are only relevant for robot models
        self.joints: Dict[str, JointInfo] = {}  # Remains empty in non-robot models

    def _validate_kwargs(self, **kwargs):
        valid_kwargs = ("position", "orientation", "velocity", "joints", "force")
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

    def apply_initial_force(
        self, force: Sequence[float] = None, multiplier: float = 1.0
    ) -> None:
        if force is None:
            log.warning("No initial force provided. Applying zero force")
            force = np.zeros(6)
        else:
            try:
                force = np.array(force, dtype=float)
                if not force.shape == (3,):
                    raise ValueError(
                        f"Invalid shape of force: {force.shape}. Expecting shape (3,)"
                    )
            except ValueError:
                raise ValueError(
                    "Invalid type for force. Expecting sequence of floats (3)"
                )
        log.debug(
            f"Applying initial force {self.force} to model {self.name} ({self.id})"
        )
        pybullet.applyExternalForce(
            self.id,
            -1,
            np.array(self.force) * multiplier,
            [0, 0, 0],
            pybullet.LINK_FRAME,
        )

    @abstractmethod
    def get_model_state(self) -> ModelState:
        pass
