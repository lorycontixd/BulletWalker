import unittest
from bulletwalker.simulator import Simulator
from bulletwalker.core.models.robot import Robot


class RobotTests(unittest.TestCase):
    def test_robot_box(self):
        path = "src/bulletwalker/assets/urdfs/models/box.urdf"
        robot = Robot(name="box", urdf_path=path)
        simulator = Simulator(models=[robot])
        simulator.load_models()

        self.assertEqual(robot.dofs, 0)
        self.assertEqual(len(robot.joints), 0)

        for i in range(0, 10):
            simulator.step()


if __name__ == "__main__":
    unittest.main()
