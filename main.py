import bulletwalker
from bulletwalker import simulator
from bulletwalker.core.callbacks.callbacks import ScoreCallback
from bulletwalker.core.callbacks.scores import (
    RootUpwardsScore,
    ForwardScore,
    CombinedScoreFunction,
)
from bulletwalker.core.models.robot import Robot
from bulletwalker.core.models.terrain import PlaneTerrain, TerrainDynamicParameters
from bulletwalker.core.math.quaternion import Quaternion
from bulletwalker.core.math.utils import deg_to_rad
from bulletwalker.data.joint_data import JointData, ControlMetric
from bulletwalker import logging as log

log.set_logging_level(log.LoggingLevel.DEBUG)


def main() -> None:
    log.info("Hello, BulletWalker!")
    sim = simulator.Simulator(use_gui=True, real_time=True)

    # a = pybullet.loadURDF("src/bulletwalker/assets/urdfs/terrains/plane_terrain.urdf")
    # b = pybullet.loadURDF("src/bulletwalker/assets/urdfs/models/model.urdf")
    # rot = Quaternion.from_euler(0, 30, 0).elements
    # pybullet.resetBasePositionAndOrientation(a, [0, 0, 0], rot)
    # pybullet.resetBasePositionAndOrientation(b, [0, 0, 2], [0, 0, 0, 1])

    sim.add_terrain(
        PlaneTerrain(
            orientation=Quaternion.from_euler(0, 16, 0),
            dynamic_parameters=TerrainDynamicParameters(
                lateral_friction=0.7,
            ),
        ),
    )

    sim.add_model(
        Robot(
            name="Robot",
            urdf_path="src/bulletwalker/assets/urdfs/models/model.urdf",
            position=[0, 0, 2],
            orientation=Quaternion.from_euler(0, -10, 0),
            joints=[
                JointData(
                    "left_hip",
                    deg_to_rad(0.0),
                    ControlMetric.TORQUE,
                ),
                JointData(
                    "right_hip",
                    deg_to_rad(30.0),
                    ControlMetric.TORQUE,
                ),
                JointData("left_ankle", 0, ControlMetric.POSITION),
            ],
        )
    )

    cb = bulletwalker.core.callbacks.callbacks.ScoreCallback(
        sim,
        score_function=RootUpwardsScore(2.0, 1.0) + ForwardScore(0.0, 1.0),
        tracked_models=["Robot"],
    )

    sim.load_models()
    sim.run(tf=5.0, dt=1e-4, callbacks=[cb])

    print("Score: ", cb.scores)

    sim.close()


if __name__ == "__main__":
    main()
