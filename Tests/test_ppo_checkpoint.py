import random
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from Utils.ppo_checkpoint import (
    load_training_checkpoint,
    restore_rng_state,
    restore_saved_hyperparameters,
    save_training_checkpoint,
)


@dataclass
class FakeArgs:
    learning_rate: float = 0.001
    total_timesteps: int = 100
    model_path: str = "model.pt"
    resume_checkpoint_path: str | None = None
    training_checkpoint_path: str | None = None
    checkpoint_interval: int = 10
    stop_after_timesteps: int = 0
    batch_size: int = 4
    minibatch_size: int = 2
    num_iterations: int = 25
    restore_model_path: str | None = None


class PPOTrainingCheckpointTests(unittest.TestCase):
    def test_round_trip_preserves_training_state_and_original_hyperparameters(self):
        torch.manual_seed(7)
        random.seed(7)
        np.random.seed(7)
        agent = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
        loss = agent(torch.ones(1, 2)).sum()
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "PPO_server_training.pt")
            save_training_checkpoint(
                path,
                algorithm="ppo",
                args=FakeArgs(),
                agent=agent,
                optimizer=optimizer,
                global_step=40,
                completed_iteration=10,
                run_name="run-1",
                run_directory=directory,
                runtime_state={"completed_episodes": 3},
            )
            _, checkpoint = load_training_checkpoint(
                path, algorithm="ppo", device=torch.device("cpu")
            )

        self.assertEqual(checkpoint["global_step"], 40)
        self.assertEqual(checkpoint["completed_iteration"], 10)
        self.assertEqual(checkpoint["runtime_state"]["completed_episodes"], 3)
        self.assertTrue(checkpoint["optimizer"]["state"])

        resumed = FakeArgs(
            learning_rate=0.5,
            total_timesteps=999,
            checkpoint_interval=99,
            stop_after_timesteps=50,
        )
        restore_saved_hyperparameters(resumed, checkpoint)
        self.assertEqual(resumed.learning_rate, 0.001)
        self.assertEqual(resumed.total_timesteps, 100)
        self.assertEqual(resumed.checkpoint_interval, 99)
        self.assertEqual(resumed.stop_after_timesteps, 50)

        first_expected = (random.random(), np.random.random(), torch.rand(1))
        restore_rng_state(checkpoint["rng_state"])
        first_actual = (random.random(), np.random.random(), torch.rand(1))
        self.assertEqual(first_expected[0], first_actual[0])
        self.assertEqual(first_expected[1], first_actual[1])
        self.assertTrue(torch.equal(first_expected[2], first_actual[2]))

    def test_algorithm_mismatch_is_rejected(self):
        agent = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(agent.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "state.pt")
            save_training_checkpoint(
                path,
                algorithm="ppo",
                args=FakeArgs(),
                agent=agent,
                optimizer=optimizer,
                global_step=0,
                completed_iteration=0,
                run_name="run",
                run_directory=directory,
                runtime_state={},
            )
            with self.assertRaisesRegex(ValueError, "not 'ppo_lstm'"):
                load_training_checkpoint(
                    path, algorithm="ppo_lstm", device=torch.device("cpu")
                )


if __name__ == "__main__":
    unittest.main()
