import numpy as np
from bulletwalker.core.math.quaternion import Quaternion


class LinkState:
    def __init__(self, *args) -> None:
        if len(args) == 1:
            info = args[0]
            self.link_world_position: np.ndarray = np.array(info[0])
            self.link_world_orientation: Quaternion = Quaternion(*info[1])
            self.local_inertial_frame_position: np.ndarray = np.array(info[2])
            self.link_world_orientation: Quaternion = Quaternion(*info[3])
            self.world_link_frame_position: np.ndarray = np.array(info[4])
            self.world_link_frame_orientation: Quaternion = Quaternion(*info[5])
            # self.world_link_linear_velocity: np.ndarray = np.array(info[6])
            # self.world_link_angular_velocity: np.ndarray = np.array(info[7])
        elif len(args) == 8:
            self.link_world_position: np.ndarray = np.array(args[0])
            self.link_world_orientation: Quaternion = Quaternion(*args[1])
            self.local_inertial_frame_position: np.ndarray = np.array(args[2])
            self.link_world_orientation: Quaternion = Quaternion(*args[3])
            self.world_link_frame_position: np.ndarray = np.array(args[4])
            self.world_link_frame_orientation: Quaternion = Quaternion(*args[5])
            # self.world_link_linear_velocity: np.ndarray = np.array(args[6])
            # self.world_link_angular_velocity: np.ndarray = np.array(args[7])
        else:
            raise ValueError(
                f"Invalid number of arguments for LinkState. Expected 1 or 8, got {len(args)}"
            )

    def __str__(self) -> str:
        return f"LinkState: {self.link_world_position=}, {self.link_world_orientation=}, {self.local_inertial_frame_position=}, {self.link_world_orientation=}, {self.world_link_frame_position=}, {self.world_link_frame_orientation=}, {self.world_link_linear_velocity=}, {self.world_link_angular_velocity=}"
