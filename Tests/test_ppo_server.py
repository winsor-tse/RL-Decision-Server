import unittest

import numpy as np

from Training.PPO_server import Args, log_step_metrics


class FakeWriter:
    def __init__(self):
        self.scalars = {}
        self.histograms = {}

    def add_scalar(self, tag, value, step):
        self.scalars[tag] = (value, step)

    def add_histogram(self, tag, values, step, bins):
        self.histograms[tag] = (values, step, bins)


class PPOMetricTests(unittest.TestCase):
    def test_live_environment_defaults_to_one_instance(self):
        self.assertEqual(Args().num_envs, 1)

    def test_metrics_frequency_defaults_to_every_step(self):
        self.assertEqual(Args().metrics_frequency, 1)

    def test_step_metrics_match_environment_dashboard(self):
        writer = FakeWriter()
        observation = np.zeros(10, dtype=np.float32)
        observation[3] = 0.8
        observation[8] = 0.5

        log_step_metrics(
            writer,
            global_step=10,
            reward=2.5,
            observation=observation,
            player_hp_index=3,
            enemy_hp_index=8,
            reward_components={
                "damage_dealt": 2.5,
                "damage_taken": 0.0,
                "terminal": 0.0,
            },
            action_counts=np.array([7, 3]),
            action_names=["up", "castSpell:1"],
            recent_actions=[0, 1],
        )

        expected_scalars = {
            "charts/step_reward",
            "environment/player_hp",
            "environment/enemy_hp",
            "environment/action_frequency/up",
            "environment/action_frequency/castSpell_1",
            "rewards/damage_dealt",
            "rewards/damage_taken",
            "rewards/terminal",
        }
        self.assertTrue(expected_scalars.issubset(writer.scalars))
        self.assertIn("environment/action_frequency", writer.histograms)


if __name__ == "__main__":
    unittest.main()
