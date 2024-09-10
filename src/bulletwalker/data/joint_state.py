import numpy as np


class JointState:
    def __init__(self, *args) -> None:
        if len(args) == 1:
            # Passed as tuple
            info = args[0]
            self.joint_position: float = info[0]
            self.joint_velocity: float = info[1]
            self.joint_reaction_forces: np.ndarray = np.array(info[2])
            self.applied_joint_motor_torque: float = info[3]
        elif len(args) == 4:
            # Passed as individual arguments
            self.joint_position: float = args[0]
            self.joint_velocity: float = args[1]
            self.joint_reaction_forces: np.ndarray = args[2]
            self.applied_joint_motor_torque: float = args[3]
        else:
            raise ValueError(
                f"Invalid number of arguments for JointState. Expected 1 or 4, got {len(args)}"
            )

    def __str__(self) -> str:
        return f"JointState: {self.joint_position=}, {self.joint_velocity=}, {self.joint_reaction_forces=}, {self.applied_joint_motor_torque=}"
