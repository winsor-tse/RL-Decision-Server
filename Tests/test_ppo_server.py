import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from Training.PPO_server import (
    Agent,
    Args,
    create_run_paths,
    log_step_metrics,
    save_agent,
)


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

    def test_default_checkpoint_is_created_under_current_run_name(self):
        args = Args()

        with tempfile.TemporaryDirectory() as directory:
            run_name, run_directory, checkpoint_path = create_run_paths(
                args,
                timestamp=1234,
                runs_directory=directory,
            )

            self.assertEqual(run_name, "Env16__PPO_server__1__1234")
            self.assertEqual(run_directory, Path(directory) / run_name)
            self.assertTrue(run_directory.is_dir())
            self.assertEqual(
                checkpoint_path,
                run_directory / "PPO_server.pt",
            )
            self.assertEqual(Path(args.model_path), checkpoint_path)

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

    def test_saved_agent_checkpoint_loads_into_same_architecture(self):
        env = SimpleNamespace(
            single_observation_space=SimpleNamespace(shape=(2,)),
            single_action_space=SimpleNamespace(n=3),
        )
        source_agent = Agent(env)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = save_agent(
                source_agent,
                str(Path(directory) / "ppo.pt"),
            )
            loaded_agent = Agent(env)
            loaded_agent.load_state_dict(
                torch.load(checkpoint_path, weights_only=True)
            )

        for source_parameter, loaded_parameter in zip(
            source_agent.parameters(),
            loaded_agent.parameters(),
        ):
            self.assertTrue(torch.equal(source_parameter, loaded_parameter))


if __name__ == "__main__":
    unittest.main()
