import unittest
import warnings

import numpy as np

from Training.DQN_server import Args, log_step_metrics


class FakeWriter:
    def __init__(self):
        self.scalars = {}
        self.histograms = {}

    def add_scalar(self, tag, value, step):
        self.scalars[tag] = (value, step)

    def add_histogram(self, tag, values, step, bins):
        self.histograms[tag] = (values, step, bins)


class DqnMetricTests(unittest.TestCase):
    def test_step_metrics_include_requested_tensorboard_tags(self):
        writer = FakeWriter()
        observation = np.zeros(10, dtype=np.float32)
        observation[3] = 0.8
        observation[8] = 0.5

        log_step_metrics(
            writer,
            global_step=10,
            reward=2.5,
            epsilon=0.4,
            replay_buffer_size=10,
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
            "charts/epsilon",
            "training/replay_buffer_size",
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

    def test_args_have_concrete_defaults_for_current_run(self):
        args = Args()

        self.assertEqual(args.buffer_size, 400)
        self.assertEqual(args.target_network_frequency, 20)
        self.assertEqual(args.batch_size, 5)
        self.assertEqual(args.learning_starts, 400)
        self.assertEqual(args.train_frequency, 1)
        self.assertEqual(args.metrics_frequency, 1)

    def test_total_timesteps_does_not_rewrite_other_hyperparameters(self):
        with self.assertWarns(UserWarning):
            args = Args(total_timesteps=500_000)

        self.assertEqual(args.buffer_size, 400)
        self.assertEqual(args.learning_starts, 400)
        self.assertEqual(args.batch_size, 5)

    def test_non_positive_total_timesteps_only_warns(self):
        with self.assertWarnsRegex(UserWarning, "greater than zero"):
            args = Args(total_timesteps=0)

        self.assertEqual(args.total_timesteps, 0)

    def test_explicit_fine_tuning_overrides_are_preserved(self):
        with self.assertWarns(UserWarning):
            args = Args(
                total_timesteps=20_000,
                buffer_size=2_000,
                target_network_frequency=7,
                batch_size=32,
                learning_starts=0,
                train_frequency=2,
                metrics_frequency=25,
            )

        self.assertEqual(args.buffer_size, 2_000)
        self.assertEqual(args.target_network_frequency, 7)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.learning_starts, 0)
        self.assertEqual(args.train_frequency, 2)
        self.assertEqual(args.metrics_frequency, 25)

    def test_batch_size_above_buffer_only_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            args = Args(buffer_size=10, batch_size=11)

        self.assertTrue(
            any("batch_size exceeds buffer_size" in str(item.message) for item in caught)
        )
        self.assertEqual(args.batch_size, 11)
        self.assertEqual(args.buffer_size, 10)


if __name__ == "__main__":
    unittest.main()
