import dataclasses
import pybullet


@dataclasses.dataclass
class ContactInfo:
    """
    Data class for storing contact information between two bodies.

    Attributes:
        - bodyA (int): ID of the first body
        - bodyB (int): ID of the second body
        - linkIndexA (int): Index of the link on the first body
        - linkIndexB (int): Index of the link on the second body
        - positionOfA (tuple): Position of the contact on the first body, in world coordinates
        - positionOfB (tuple): Position of the contact on the second body, in world coordinates
        - contactNormalOnB (tuple): Normal vector of the contact on the second body, pointing towards the first body
        - contactDistance (float): Distance between the two contact points. Positive if the bodies are separated, negative if they are in contact
        - normalForce (float): Normal force applied during last 'simulationStep' call
        - lateralFriction1 (float): Lateral friction force applied in the direction of 'lateralFrictionDir1'
        - lateralFrictionDir1 (tuple): Direction of the first lateral friction force
        - lateralFriction2 (float): Lateral friction force applied in the direction of 'lateralFrictionDir2'
        - lateralFrictionDir2 (tuple): Direction of the second lateral friction force

    """

    bodyA: int
    bodyB: int
    linkIndexA: int
    linkIndexB: int
    positionOfA: tuple
    positionOfB: tuple
    contactNormalOnB: tuple
    contactDistance: float
    normalForce: float
    lateralFriction1: float
    lateralFrictionDir1: tuple
    lateralFriction2: float
    lateralFrictionDir2: tuple
    linkNameA: str = None
    linkNameB: str = None

    def __post_init__(self):
        try:
            a = pybullet.getJointInfo(self.bodyA, self.linkIndexA)
        except pybullet.error:
            a = None

        try:
            b = pybullet.getJointInfo(self.bodyB, self.linkIndexB)
        except pybullet.error:
            b = None

        if a is not None:
            self.linkNameA = a[12].decode("utf-8")
        if b is not None:
            self.linkNameB = b[12].decode("utf-8")
