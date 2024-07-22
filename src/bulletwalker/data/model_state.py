import pybullet
import numpy as np
from dataclasses import dataclass
from bulletwalker.data.joint_state import JointState
from bulletwalker.core.models.model import Model
from bulletwalker.core.math.quaternion import Quaternion


@dataclass
class ModelData:
    base_position: np.ndarray
    base_orientation: Quaternion
    joint_states: dict[str, JointState]


def get_model_data(model: Model) -> ModelData:
    _pos, _or = pybullet.getBasePositionAndOrientation(model.id)
    joints = [j for j in model.joints]
    joint_states = {
        j: JointState(pybullet.getJointState(model.id, model.joints[j].id))
        for j in joints
    }
    return ModelData(
        base_position=np.array(_pos),
        base_orientation=Quaternion(_or),
        joint_states=joint_states,
    )
