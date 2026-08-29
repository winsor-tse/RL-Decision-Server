import io
import subprocess
import sys
import unittest
from unittest import mock

from Automation.infer import resolve_inference_command
from Automation.processes import (
    load_config,
    normalize_command,
    run_stack,
    wait_for_interrupted_child,
)
from Automation.record import resolve_recorder_command
from Automation.tensorboard_server import FilteredStderr, NO_TENSORFLOW_NOTICE
from Automation.train import resolve_training_command


class AutomationConfigTests(unittest.TestCase):
    def test_configured_training_algorithm_resolves_its_command(self):
        config = load_config("Automation/automation_config.yaml")
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

        algorithm, command = resolve_training_command(config)
        self.assertEqual(algorithm, config["rl_algorithm"])
        self.assertEqual(command, config[f"{algorithm}_command"])

    def test_restore_model_path_is_added_to_ppo_training_commands(self):
        config = load_config("Automation/automation_config.yaml")
        for algorithm_name, checkpoint_name in (
            ("ppo", "PPO_server.pt"),
            ("ppo_lstm", "PPO_lstm_server.pt"),
        ):
            with self.subTest(algorithm=algorithm_name):
                restore_model_path = f"runs/existing/{checkpoint_name}"
                config["rl_algorithm"] = algorithm_name
                config["restore_model_path"] = restore_model_path

                algorithm, command = resolve_training_command(config)

                self.assertEqual(algorithm, algorithm_name)
                self.assertEqual(
                    command[-2:],
                    ["--restore-model-path", restore_model_path],
                )

    def test_restore_model_path_is_rejected_for_dqn(self):
        config = load_config("Automation/automation_config.yaml")
        config["rl_algorithm"] = "dqn"
        config["restore_model_path"] = "runs/existing/DQN_server.pt"

        with self.assertRaisesRegex(ValueError, "only supported"):
            resolve_training_command(config)

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

    def test_interrupted_recorder_gets_time_to_save(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.return_value = 0

        return_code = wait_for_interrupted_child(
            process,
            "offline recorder",
            timeout_seconds=30,
        )

        self.assertEqual(return_code, 0)
        process.wait.assert_called_once_with(timeout=30)

    def test_interrupted_recorder_timeout_falls_back_to_cleanup(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired(
            cmd="offline recorder",
            timeout=1,
        )

        return_code = wait_for_interrupted_child(
            process,
            "offline recorder",
            timeout_seconds=1,
        )

        self.assertIsNone(return_code)

    def test_run_stack_waits_for_recorder_after_ctrl_c(self):
        bridge_process = mock.Mock()
        recorder_process = mock.Mock()
        recorder_process.poll.return_value = None
        recorder_process.wait.side_effect = [KeyboardInterrupt(), 0]
        config = {
            "bridge_command": ["python", "bridge.py"],
            "interrupt_grace_seconds": 12,
        }

        with (
            mock.patch(
                "Automation.processes.start_process",
                side_effect=[bridge_process, recorder_process],
            ),
            mock.patch("Automation.processes.wait_until_ready"),
            mock.patch("Automation.processes.terminate_process"),
        ):
            return_code = run_stack(
                config,
                ["python", "-m", "Offline.record_minari"],
                "offline recorder",
                start_tensorboard=False,
            )

        self.assertEqual(return_code, 0)
        self.assertEqual(
            recorder_process.wait.call_args_list,
            [mock.call(), mock.call(timeout=12)],
        )

    def test_configured_inference_algorithm_resolves_its_command(self):
        config = load_config("Automation/automation_config.yaml")

        command = resolve_inference_command(config)

        algorithm = config["inference_algorithm"]
        self.assertEqual(
            command,
            config[f"{algorithm}_inference_command"],
        )

    def test_offline_recorder_command_is_configured(self):
        config = load_config("Automation/automation_config.yaml")

        self.assertEqual(
            resolve_recorder_command(config),
            ["python", "-m", "Offline.record_player"],
        )

    def test_offline_recorder_command_can_be_overridden(self):
        command = resolve_recorder_command(
            {},
            "python -m Offline.record_player --no-keyboard",
        )

        self.assertEqual(
            command,
            "python -m Offline.record_player --no-keyboard",
        )

    def test_missing_offline_recorder_command_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "recorder_command"):
            resolve_recorder_command({})

    def test_recurrent_ppo_inference_command_is_selectable(self):
        config = load_config("Automation/automation_config.yaml")
        config["inference_algorithm"] = "ppo_lstm"

        command = resolve_inference_command(config)

        self.assertEqual(command, config["ppo_lstm_inference_command"])
        self.assertEqual(
            command,
            [
                "python",
                "-m",
                "Inference.ppo_lstm_eval",
                "--model-path",
                "runs/XXXX/PPO_lstm_server.pt",
            ],
        )

    def test_feedforward_ppo_inference_command_is_selectable(self):
        config = load_config("Automation/automation_config.yaml")
        config["inference_algorithm"] = "ppo"

        command = resolve_inference_command(config)

        self.assertEqual(command, config["ppo_inference_command"])
        self.assertEqual(
            command,
            [
                "python",
                "-m",
                "Inference.ppo_eval",
                "--model-path",
                "runs/XXXX/PPO_server.pt",
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
