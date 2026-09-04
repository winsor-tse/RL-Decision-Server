import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from Offline.any_percent_bc import (
    BC_MODEL_FILENAME,
    TrainConfig,
    train,
)


class FakeDataset:
    def __init__(self, episodes):
        self.episodes = episodes

    def __iter__(self):
        return iter(self.episodes)

    def iterate_episodes(self, episode_indices=None):
        if episode_indices is None:
            return iter(self.episodes)
        selected = [
            episode
            for episode in self.episodes
            if episode.id in episode_indices
        ]
        return iter(selected)


class AnyPercentBCTrainingTests(unittest.TestCase):
    def test_training_saves_bc_model_in_generated_run_directory(self):
        episode = SimpleNamespace(
            id=0,
            observations=np.array(
                [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                dtype=np.float32,
            ),
            actions=np.array([0, 1], dtype=np.int64),
            rewards=np.array([1.0, 2.0], dtype=np.float32),
            terminations=np.array([False, True]),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainConfig(
                dataset_id="env16/BC-v0",
                update_steps=0,
                buffer_size=8,
                batch_size=2,
                top_fraction=1.0,
                checkpoints_path=temp_dir,
            )
            with mock.patch(
                "Offline.any_percent_bc.minari.load_dataset",
                return_value=FakeDataset([episode]),
            ):
                train(config)

            run_directory = Path(config.checkpoints_path)
            self.assertTrue(run_directory.is_dir())
            self.assertTrue(Path(run_directory, BC_MODEL_FILENAME).is_file())
            self.assertTrue(Path(run_directory, "config.yaml").is_file())
            self.assertFalse(Path(run_directory, "final_checkpoint.pt").exists())


if __name__ == "__main__":
    unittest.main()
