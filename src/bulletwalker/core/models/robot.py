import pathlib
import pybullet
from typing import List, Dict
from .model import Model
from bulletwalker.data.joint_info import JointInfo
from bulletwalker.data.joint_data import JointData, ControlMetric
from bulletwalker.data.joint_state import JointState
from bulletwalker import logging as log


class Robot(Model):
    def __init__(
        self,
        urdf_path: str | pathlib.Path,
        name: str = "Robot Model",
        **kwargs,
    ) -> None:
        super().__init__(name, urdf_path, **kwargs)
        i_j: List[JointData] = kwargs.get("joints", [])
        self.initial_joints: Dict[str, JointData] = {j.name: j for j in i_j}
        print(f"Initial rot of {self.name}: {self.orientation.elements}")

        self.i = 0

    def load(self, model_id: int) -> None:
        super().load(model_id)
        self._initialize_joints()

    def _validate_joints(self) -> None:
        joint_names = [
            pybullet.getJointInfo(self.id, i)[1].decode("utf-8")
            for i in range(self.dofs)
        ]
        for joint in self.initial_joints:
            if joint not in joint_names:
                raise ValueError(
                    f"Passed joint {joint} is not present in the robot model. Available joints: {joint_names}"
                )

    def _initialize_joints(self):
        if self.id < 0:
            raise ValueError("Robot ID is not set. Load model first.")

        self.dofs = pybullet.getNumJoints(self.id)
        self._validate_joints()
        for i in range(self.dofs):
            joint_info = pybullet.getJointInfo(self.id, i)
            joint_name = str(joint_info[1].decode("utf-8"))
            self.joints[joint_name] = JointInfo(
                joint_info,
                initial_position=(
                    self.initial_joints[joint_name].initial_position
                    if joint_name in self.initial_joints
                    else 0
                ),
                control_metric=(
                    self.initial_joints[joint_name].control_metric
                    if joint_name in self.initial_joints
                    else ControlMetric.POSITION
                ),
            )
            print(f"Joint name: {joint_name}, Type: {self.joints[joint_name].type}")
            pybullet.enableJointForceTorqueSensor(self.id, i, enableSensor=True)
            log.debug(f"Setting joint {joint_name} to initial value")
            pybullet.resetJointState(
                self.id,
                i,
                self.joints[joint_name].initial_position,
            )

        log.info(f"Loaded joints for robot {self.name}. DOFs: {self.dofs}")

    def _update_joints(self, joint_info: tuple) -> None:
        if self.id < 0:
            raise ValueError("Robot ID is not set. Load model first.")

        for i in range(self.dofs):
            new_info = pybullet.getJointInfo(self.id, i)
            joint_name = str(new_info[1].decode("utf-8"))
            self.joints[joint_name] = JointInfo(
                new_info,
                initial_position=self.joints[joint_name].initial_position,
                control_metric=self.joints[joint_name].control_metric,
            )

    def reset_joints(self) -> None:
        if self.id < 0:
            raise ValueError("Robot ID is not set. Load model first.")
        for i in range(self.dofs):
            joint_info = pybullet.getJointInfo(self.id, i)
            joint_name = str(joint_info[1].decode("utf-8"))
            pybullet.resetJointState(
                self.id,
                i,
                (
                    self.joints[joint_name].initial_position
                    if joint_name in self.joints
                    else 0
                ),
            )

    def step(self) -> None:
        if self.id < 0:
            raise ValueError("Robot ID is not set. Load model first.")

        if self.i == 0:
            pybullet.applyExternalForce(
                self.id, -1, [20, 0, 0], [0, 0, 0], pybullet.WORLD_FRAME
            )

        joint_indices = [j.index for j in self.joints.values()]
        control_mode = ControlMetric.VELOCITY
        control_values = [0.01] * len(joint_indices)
        print(
            f"Setting joints {joint_indices} to {control_values} with control mode {control_mode.name}"
        )
        pybullet.setJointMotorControlArray(
            self.id,
            joint_indices,
            control_mode,
            forces=control_values,
        )

        rev_joints = [
            j for j in self.joints.values() if j.type == pybullet.JOINT_REVOLUTE
        ]
        for j in rev_joints:
            j: JointInfo
            state = JointState(pybullet.getJointState(self.id, j.index))
            print(f"Joint {j.name} ==> state: {state}")
        print("")
        self.i += 1

    def post_step(self) -> None:
        pass

    def get_total_mass(self) -> float:
        total_mass: float = 0
        base_mass = pybullet.getDynamicsInfo(self.id, -1)[0]
        total_mass += base_mass

        for i in range(self.dof):
            joint_mass = pybullet.getDynamicsInfo(self.id, i)[0]
            total_mass += joint_mass
        return total_mass
