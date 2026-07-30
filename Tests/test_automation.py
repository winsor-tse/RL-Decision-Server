import io
import os
import tempfile
import sys
import unittest
from pathlib import Path

from Automation.infer import resolve_inference_command
from Automation.processes import load_config, normalize_command
from Automation.tensorboard_server import FilteredStderr, NO_TENSORFLOW_NOTICE


class AutomationConfigTests(unittest.TestCase):
    def test_default_config_selects_ppo_server(self):
        config = load_config("Automation/automation_config.yaml")
        self.assertEqual(config["rl_algorithm"], "ppo")
        self.assertEqual(
            config["ppo_command"],
            ["python", "-m", "Training.PPO_server"],
        )

    def test_python_command_uses_active_interpreter(self):
        command = normalize_command(["python", "-m", "example"])
        self.assertEqual(command, [sys.executable, "-m", "example"])

    def test_smoke_config_uses_argument_lists(self):
        config = load_config("Tests/automation_smoke.yaml")
        self.assertIsInstance(config["bridge_command"], list)
        self.assertIsInstance(config["smoke_command"], list)

    def test_empty_command_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_command([])

    def test_string_override_preserves_quoted_argument(self):
        command = normalize_command(
            'python -m Inference.dqn_eval --model_path "runs/model file.pt"'
        )
        self.assertEqual(command[-1], "runs/model file.pt")

    def test_inference_defaults_to_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            runs_directory = Path(directory)
            older = runs_directory / "older.pt"
            latest = runs_directory / "latest.pt"
            older.touch()
            latest.touch()
            older.touch()
            latest.touch()
            older_time = older.stat().st_mtime - 10
            os.utime(older, (older_time, older_time))

            command = resolve_inference_command(
                {"inference_model_path": "latest"},
                runs_directory=runs_directory,
            )

        self.assertEqual(command[-1], str(latest))

    def test_tensorboard_filter_keeps_real_errors(self):
        output = io.StringIO()
        filtered = FilteredStderr(output)

        filtered.write(NO_TENSORFLOW_NOTICE)
        filtered.write("\n")
        filtered.write("real TensorBoard error\n")

        self.assertEqual(output.getvalue(), "real TensorBoard error\n")


if __name__ == "__main__":
    unittest.main()
