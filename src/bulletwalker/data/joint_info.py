import enum


class JointType(enum.IntEnum):
    REVOLUTE = 0
    PRISMATIC = 1
    SPHERICAL = 2
    PLANAR = 3
    FIXED = 4


class JointInfo:
    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            info = args[0]
            if not isinstance(args[0], tuple):
                raise TypeError("Invalid type for JointInfo. Expecting tuple")
            # Unpack tuple
            self.index: int = info[0]
            self.name: str = info[1]
            self.type: JointType = JointType(info[2])
            self.q_index: int = info[3]
            self.u_index: int = info[4]
            self.flags: int = info[5]
            self.damping: float = info[6]
            self.friction: float = info[7]
            self.lower_limit: float = info[8]
            self.upper_limit: float = info[9]
            self.max_force: float = info[10]
            self.max_velocity: float = info[11]
            self.link_name: str = info[12]
            self.joint_axis: tuple = info[13]
            self.parent_frame_pos: tuple = info[14]
            self.parent_frame_orn: tuple = info[15]
            self.parent_index: int = info[16]
        else:
            print("nono")

        valid_kwargs = (
            "initial_position",
            "control_metric",
        )

        for key, value in kwargs.items():
            if key in valid_kwargs:
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid keyword argument: {key}")
