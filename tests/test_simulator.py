import unittest
import pathlib
from bulletwalker.core import simulator


class SimulatorTests(unittest.TestCase):
    def test_simulator_creation(self):
        urdf_file = (
            pathlib.Path(__file__).parents[1] / "assets" / "urdfs" / "model.urdf"
        )
        N = 10
        sims = [simulator.Simulator(model_path=urdf_file) for _ in range(N)]

        for i in range(N):
            self.assertEqual(sims[i].client, i)
            self.assertEqual(sims[i].name, "Simulator")
            self.assertEqual(sims[i].model_path, urdf_file)


if __name__ == "__main__":
    unittest.main()
