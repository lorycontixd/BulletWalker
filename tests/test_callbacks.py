import unittest
import numpy as np
from bulletwalker.simulator import Simulator
from bulletwalker.core.models.robot import Robot
from bulletwalker.core.callbacks.callbacks import ScoreCallback, PrinterCallback
from bulletwalker.core.callbacks import scores
from bulletwalker.core.math.quaternion import Quaternion
from bulletwalker.data.simulation_step import SimulationStep
from bulletwalker.data.model_state import ModelState


class CallbacksTests(unittest.TestCase):
    @unittest.skip("Not implemented")
    def test_early_stopping(self):
        box = Robot(
            name="box", urdf_path="src/bulletwalker/assets/urdfs/models/box.urdf"
        )
        box2 = Robot(
            name="box2",
            urdf_path="src/bulletwalker/assets/urdfs/models/box.urdf",
            position=[0, 3, 2],
        )
        simulator = Simulator(models=[box, box2])
        simulator.load_models()

        callback = ScoreCallback(
            simulator,
            score_function=scores.RootUpwardsScore(
                2.0,
                1.0,
            ),
        )

        simulator.run(callbacks=[callback], tf=5.0)

    def test_root_upward_scores(self):
        simulation_steps = [
            SimulationStep(
                0,
                0.0,
                0.0,
                0.0,
                {
                    "box": ModelState(
                        np.array([0.0, 0.0, 2.0]), Quaternion.from_euler(0, 0, 0), {}
                    )
                },
            ),
            SimulationStep(
                1,
                0.0,
                0.0,
                0.0,
                {
                    "box": ModelState(
                        np.array([0.0, 0.0, 1.5]), Quaternion.from_euler(0, 0, 0), {}
                    )
                },
            ),
            SimulationStep(
                2,
                0.0,
                0.0,
                0.0,
                {
                    "box": ModelState(
                        np.array([0.0, 0.0, 1.0]), Quaternion.from_euler(0, 0, 0), {}
                    )
                },
            ),
            SimulationStep(
                3,
                0.0,
                0.0,
                0.0,
                {
                    "box": ModelState(
                        np.array([0.0, 0.0, 0.5]), Quaternion.from_euler(0, 0, 0), {}
                    )
                },
            ),
            SimulationStep(
                4,
                0.0,
                0.0,
                0.0,
                {
                    "box": ModelState(
                        np.array([0.0, 0.0, 0.0]), Quaternion.from_euler(0, 0, 0), {}
                    )
                },
            ),
        ]

        # Test root upwards score

        callback = ScoreCallback(
            None,
            score_function=scores.RootUpwardsScore(
                2.0,
                1.0,
            ),
        )
        for step in simulation_steps:
            callback.on_simulation_step(step)

        self.assertEqual(callback.history[0].scores["box"], 1.0)
        self.assertEqual(callback.history[1].scores["box"], np.exp(-0.5))
        self.assertEqual(callback.history[2].scores["box"], np.exp(-1.0))
        self.assertEqual(callback.history[3].scores["box"], np.exp(-1.5))
        self.assertEqual(callback.history[4].scores["box"], np.exp(-2.0))

    def test_forward_scores(self):
        simulation_steps = [
            SimulationStep(
                0,
                0.0,
                0.0,
                0.0,
                {
                    "box": ModelState(
                        np.array([0.0, 0.0, 0.0]), Quaternion.from_euler(0, 0, 0), {}
                    )
                },
            ),
            SimulationStep(
                1,
                0.0,
                0.0,
                0.0,
                {
                    "box": ModelState(
                        np.array([1.0, 0.0, 0]), Quaternion.from_euler(0, 0, 0), {}
                    )
                },
            ),
            SimulationStep(
                2,
                0.0,
                0.0,
                0.0,
                {
                    "box": ModelState(
                        np.array([2.0, 0.0, 0.0]), Quaternion.from_euler(0, 0, 0), {}
                    )
                },
            ),
            SimulationStep(
                3,
                0.0,
                0.0,
                0.0,
                {
                    "box": ModelState(
                        np.array([3.0, 0.0, 0.0]), Quaternion.from_euler(0, 0, 0), {}
                    )
                },
            ),
            SimulationStep(
                4,
                0.0,
                0.0,
                0.0,
                {
                    "box": ModelState(
                        np.array([4.0, 0.0, 0.0]), Quaternion.from_euler(0, 0, 0), {}
                    )
                },
            ),
        ]

        callback = ScoreCallback(
            None,
            score_function=scores.ForwardScore(
                0.0,
                1.0,
            ),
        )

        for step in simulation_steps:
            callback.on_simulation_step(step)

        # Calculated by hand
        self.assertEqual(callback.history[0].scores["box"], 0.0)
        self.assertAlmostEqual(callback.history[1].scores["box"], 0.5)
        self.assertAlmostEqual(callback.history[2].scores["box"], 0.70483, places=4)
        self.assertAlmostEqual(callback.history[3].scores["box"], 0.79517, places=4)

    def test_combined_scores(self):
        simulation_steps = [
            SimulationStep(
                0,
                0.0,
                0.0,
                0.0,
                {
                    "box": ModelState(
                        np.array([0.0, 0.0, 2.0]), Quaternion.from_euler(0, 0, 0), {}
                    )
                },
            ),
            SimulationStep(
                1,
                0.0,
                0.0,
                0.0,
                {
                    "box": ModelState(
                        np.array([1.0, 0.0, 1.5]), Quaternion.from_euler(0, 0, 0), {}
                    )
                },
            ),
            SimulationStep(
                2,
                0.0,
                0.0,
                0.0,
                {
                    "box": ModelState(
                        np.array([2.0, 0.0, 1.0]), Quaternion.from_euler(0, 0, 0), {}
                    )
                },
            ),
            SimulationStep(
                3,
                0.0,
                0.0,
                0.0,
                {
                    "box": ModelState(
                        np.array([3.0, 0.0, 0.5]), Quaternion.from_euler(0, 0, 0), {}
                    )
                },
            ),
            SimulationStep(
                4,
                0.0,
                0.0,
                0.0,
                {
                    "box": ModelState(
                        np.array([4.0, 0.0, 0.0]), Quaternion.from_euler(0, 0, 0), {}
                    )
                },
            ),
        ]
        s: scores.CombinedScoreFunction = scores.RootUpwardsScore(
            2.0,
            1.0,
        ) + scores.ForwardScore(
            0.0,
            1.0,
        )
        s.set_weights([0.3, 0.7])

        callback = ScoreCallback(
            None,
            score_function=s,
        )

        for step in simulation_steps:
            callback.on_simulation_step(step)

        self.assertEqual(callback.history[0].scores["box"], 0.3)


if __name__ == "__main__":
    unittest.main()
