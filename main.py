from bulletwalker.core import simulator
from bulletwalker.core.models.robot import Robot
from bulletwalker.core.models.terrain import PlaneTerrain
from bulletwalker.core.math.quaternion import Quaternion
from bulletwalker import logging as log
import numpy as np
import time
import pybullet

log.set_logging_level(log.LoggingLevel.DEBUG)


def main() -> None:
    log.info("Hello, BulletWalker!")
    sim = simulator.Simulator(use_gui=True, real_time=False)

    # a = pybullet.loadURDF("src/bulletwalker/assets/urdfs/terrains/plane_terrain.urdf")
    # b = pybullet.loadURDF("src/bulletwalker/assets/urdfs/models/model.urdf")
    # rot = Quaternion.from_euler(0, 30, 0).elements
    # pybullet.resetBasePositionAndOrientation(a, [0, 0, 0], rot)
    # pybullet.resetBasePositionAndOrientation(b, [0, 0, 2], [0, 0, 0, 1])

    sim.add_terrain(
        PlaneTerrain(
            path="src/bulletwalker/assets/urdfs/terrains/plane_terrain.urdf",
            orientation=Quaternion.from_euler(0, 30, 0),
        ),
    )

    sim.add_model(
        Robot(
            name="Robot",
            urdf_path="src/bulletwalker/assets/urdfs/models/model.urdf",
            position=[0, 0, 3],
        )
    )

    sim.run(tf=50.0)


if __name__ == "__main__":
    main()
