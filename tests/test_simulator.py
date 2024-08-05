import unittest
import pathlib
from bulletwalker import simulator
from bulletwalker.core.models.terrain import PlaneTerrain


class SimulatorTests(unittest.TestCase):
    def test_simulator_creation(self):
        urdf_file = (
            pathlib.Path(__file__).parents[1] / "assets" / "urdfs" / "model.urdf"
        )
        N = 10
        sims = [simulator.Simulator() for _ in range(N)]

        pass

    def test_simulator_add_terrain(self):
        pass


if __name__ == "__main__":
    unittest.main()
