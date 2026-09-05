import io
import subprocess
import sys
import unittest
from unittest import mock

from Automation.infer import resolve_inference_command
from Automation.offline_rl import (
    build_evaluation_command,
    build_training_command,
    main as offline_main,
    run_offline_rl,
)
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
    def test_awac_offline_routes_options_without_bridge(self):
        with (
            mock.patch('Automation.offline_rl.run_process', return_value=7) as process,
            mock.patch('Automation.offline_rl.run_stack') as stack,
            mock.patch('Automation.offline_rl.load_config') as config,
        ):
            result = offline_main([
                '--algorithm', 'awac', '--update-steps', '12',
                '--dataset-id', 'env16/BC-v2', '--device', 'cpu',
                '--batch-size', '8', '--buffer-size', '100',
                '--hidden-dim', '32', '--learning-rate', '0.001',
                '--tau', '0.01', '--awac-lambda', '2', '--gamma', '0.95',
                '--no-normalize-state', '--checkpoints-path', 'runs/my awac',
            ])
        self.assertEqual(result, 7)
        stack.assert_not_called()
        config.assert_not_called()
        command, name = process.call_args.args
        self.assertEqual(name, 'AWAC training')
        self.assertEqual(command[:3], ['python', '-m', 'Offline.awac'])
        for flag, value in {
            '--offline-iterations': '12', '--online-iterations': '0',
            '--dataset-id': 'env16/BC-v2', '--device': 'cpu',
            '--batch-size': '8', '--buffer-size': '100',
            '--hidden-dim': '32', '--learning-rate': '0.001',
            '--tau': '0.01', '--awac-lambda': '2.0', '--gamma': '0.95',
            '--checkpoints-path': 'runs/my awac',
        }.items():
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn('--no-normalize-state', command)
        self.assertNotIn('--top-fraction', command)
        self.assertNotIn('--eval-every', command)

    def test_awac_online_starts_supervised_bridge(self):
        with (
            mock.patch('Automation.offline_rl.run_process') as process,
            mock.patch('Automation.offline_rl.run_stack', return_value=0) as stack,
            mock.patch('Automation.offline_rl.load_config', return_value={'bridge_command': ['bridge']}) as config,
        ):
            self.assertEqual(offline_main([
                '--algorithm', 'awac', '--online-iterations', '5',
                '--config', 'custom.yaml',
            ]), 0)
        process.assert_not_called()
        config.assert_called_once_with('custom.yaml')
        command = stack.call_args.args[1]
        self.assertEqual(command[command.index('--online-iterations') + 1], '5')
        self.assertNotIn('--device', command)  # auto uses AWAC's device detection
        self.assertTrue(stack.call_args.kwargs['start_tensorboard'])

    def test_invalid_awac_modes_and_online_options_fail_before_launch(self):
        cases = [
            ['--algorithm', 'awac', '--mode', 'dataset'],
            ['--algorithm', 'awac', '--mode', 'live'],
            ['--online-iterations', '5'],
            ['--algorithm', 'awac', '--online-iterations', '-1'],
            ['--algorithm', 'awac', '--top-fraction', '0.5'],
        ]
        with mock.patch('Automation.offline_rl.run_process') as process, mock.patch('Automation.offline_rl.run_stack') as stack:
            for args in cases:
                with self.subTest(args=args), self.assertRaises(ValueError):
                    offline_main(args)
        process.assert_not_called()
        stack.assert_not_called()

    def test_bc_remains_default_training_algorithm(self):
        with mock.patch('Automation.offline_rl.run_process', return_value=0) as process:
            self.assertEqual(offline_main([]), 0)
        command = process.call_args.args[0]
        self.assertEqual(command[:3], ['python', '-m', 'Offline.any_percent_bc'])
        self.assertNotIn('--online-iterations', command)

    def test_offline_training_does_not_start_bridge(self):
        with (
            mock.patch(
                "Automation.offline_rl.run_process",
                return_value=0,
            ) as run_process_mock,
            mock.patch("Automation.offline_rl.run_stack") as run_stack_mock,
        ):
            return_code = run_offline_rl(
                {},
                ["python", "-m", "Offline.any_percent_bc"],
                mode="train",
            )

        self.assertEqual(return_code, 0)
        run_process_mock.assert_called_once()
        run_stack_mock.assert_not_called()

    def test_offline_training_command_contains_dataset_and_output(self):
        command = build_training_command(
            dataset_id="env16/BC-v2",
            update_steps=10_000,
            buffer_size=50_000,
            batch_size=128,
            top_fraction=1.0,
            gamma=0.99,
            eval_every=1_000,
            normalize_state=True,
            checkpoints_path="runs",
        )

        self.assertEqual(
            command[:3],
            ["python", "-m", "Offline.any_percent_bc"],
        )
        self.assertIn("env16/BC-v2", command)
        self.assertIn("10000", command)
        self.assertIn("--normalize-state", command)
        self.assertIn("runs", command)

    def test_offline_dataset_evaluation_does_not_start_bridge(self):
        with (
            mock.patch(
                "Automation.offline_rl.run_process",
                return_value=0,
            ) as run_process_mock,
            mock.patch("Automation.offline_rl.run_stack") as run_stack_mock,
        ):
            return_code = run_offline_rl(
                {},
                ["python", "-m", "Inference.any_percent_bc_eval"],
                mode="dataset",
            )

        self.assertEqual(return_code, 0)
        run_process_mock.assert_called_once()
        run_stack_mock.assert_not_called()

    def test_offline_live_evaluation_starts_bridge(self):
        with (
            mock.patch("Automation.offline_rl.run_process") as run_process_mock,
            mock.patch(
                "Automation.offline_rl.run_stack",
                return_value=0,
            ) as run_stack_mock,
        ):
            return_code = run_offline_rl(
                {"bridge_command": ["python", "bridge.py"]},
                ["python", "-m", "Inference.any_percent_bc_eval"],
                mode="live",
            )

        self.assertEqual(return_code, 0)
        run_process_mock.assert_not_called()
        run_stack_mock.assert_called_once()

    def test_offline_evaluation_command_contains_explicit_inputs(self):
        command = build_evaluation_command(
            mode="dataset",
            checkpoint_path="runs/bc/BC_model.pt",
            dataset_id="env16/BC-v2",
            eval_episodes=3,
            top_fraction=0.5,
            gamma=0.95,
            device="cpu",
            normalize_state=False,
            output_csv="reports/predictions.csv",
        )

        self.assertEqual(
            command[:3],
            ["python", "-m", "Inference.any_percent_bc_eval"],
        )
        self.assertIn("runs/bc/BC_model.pt", command)
        self.assertIn("env16/BC-v2", command)
        self.assertIn("--no-normalize-state", command)
        self.assertEqual(command[-2:], ["--output-csv", "reports/predictions.csv"])

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
