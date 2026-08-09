import os
import tempfile
import unittest
from pathlib import Path

from Utils.model_paths import (
    inference_checkpoint_path,
    training_checkpoint_path,
)


class ModelPathTests(unittest.TestCase):
    def test_training_checkpoint_defaults_to_run_directory(self):
        checkpoint = training_checkpoint_path(
            Path("runs") / "Env16__PPO_server__1__1234",
            None,
            "PPO_server.pt",
        )

        self.assertEqual(
            checkpoint,
            Path("runs/Env16__PPO_server__1__1234/PPO_server.pt"),
        )

    def test_explicit_training_checkpoint_path_is_preserved(self):
        checkpoint = training_checkpoint_path(
            Path("runs") / "ignored",
            "custom/model.pt",
            "PPO_server.pt",
        )

        self.assertEqual(checkpoint, Path("custom/model.pt"))

    def test_inference_selects_newest_matching_run_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            runs_directory = Path(directory)
            older = runs_directory / "older" / "PPO_lstm_server.pt"
            newer = runs_directory / "newer" / "PPO_lstm_server.pt"
            older.parent.mkdir()
            newer.parent.mkdir()
            older.write_bytes(b"older")
            newer.write_bytes(b"newer")
            os.utime(older, ns=(1_000_000_000, 1_000_000_000))
            os.utime(newer, ns=(2_000_000_000, 2_000_000_000))

            checkpoint = inference_checkpoint_path(
                None,
                "PPO_lstm_server.pt",
                runs_directory,
            )

        self.assertEqual(checkpoint, newer)

    def test_explicit_inference_checkpoint_must_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pt"

            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                inference_checkpoint_path(
                    str(missing),
                    "PPO_server.pt",
                    directory,
                )

    def test_missing_run_checkpoint_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "--model-path"):
                inference_checkpoint_path(
                    None,
                    "PPO_server.pt",
                    directory,
                )


if __name__ == "__main__":
    unittest.main()
