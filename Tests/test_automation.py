import io
import sys
import unittest

from Automation.infer import resolve_inference_command
from Automation.processes import load_config, normalize_command
from Automation.tensorboard_server import FilteredStderr, NO_TENSORFLOW_NOTICE


class AutomationConfigTests(unittest.TestCase):
    def test_default_config_selects_ppo_server(self):
        config = load_config("Automation/automation_config.yaml")
        self.assertEqual(config["rl_algorithm"], "ppo")
        self.assertEqual(
            config["dqn_command"],
            ["python", "-m", "Training.DQN_server"],
        )
        self.assertEqual(
            config["ppo_command"],
            ["python", "-m", "Training.PPO_server"],
        )
        self.assertEqual(
            config["ppo_lstm_command"],
            ["python", "-m", "Training.PPO_lstm_server"],
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

    def test_inference_uses_explicit_configured_command(self):
        config = load_config("Automation/automation_config.yaml")

        command = resolve_inference_command(config)

        self.assertEqual(command, config["ppo_inference_command"])
        self.assertEqual(
            command,
            [
                "python",
                "-m",
                "Inference.ppo_eval",
                "--model-path",
                "runs/PPO_server.pt",
            ],
        )

    def test_dqn_inference_command_is_selectable(self):
        config = load_config("Automation/automation_config.yaml")
        config["inference_algorithm"] = "dqn"

        command = resolve_inference_command(config)

        self.assertEqual(command, config["dqn_inference_command"])
        self.assertEqual(
            command,
            [
                "python",
                "-m",
                "Inference.dqn_eval",
                "--model-path",
                "runs/DQN_server__1783138095/DQN_server.pt",
            ],
        )

    def test_missing_inference_command_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "inference_algorithm or inference_command",
        ):
            resolve_inference_command({})

    def test_tensorboard_filter_keeps_real_errors(self):
        output = io.StringIO()
        filtered = FilteredStderr(output)

        filtered.write(NO_TENSORFLOW_NOTICE)
        filtered.write("\n")
        filtered.write("real TensorBoard error\n")

        self.assertEqual(output.getvalue(), "real TensorBoard error\n")


if __name__ == "__main__":
    unittest.main()
