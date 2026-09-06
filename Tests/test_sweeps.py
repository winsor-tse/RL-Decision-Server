import contextlib
from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from Sweeps import (
    bootstrap_mean, grid_search, hierarchical_mean, latin_hypercube,
    monte_carlo, summarize_runs, wilson_interval,
)
from Sweeps.__main__ import main
from Sweeps.runner import (
    _run_child, build_plan, live_lock, read_json, report, run_plan,
    run_trial, training_command, write_json,
)


class SearchTests(unittest.TestCase):
    def test_grid_is_complete_lazy_and_preserves_types(self):
        result = grid_search({"rate": [0.1, 0.2], "flag": [False, True], "mode": [None, "x"]})
        self.assertIs(iter(result), result)
        points = list(result)
        self.assertEqual(len(points), 8)
        self.assertEqual(points[0], {"rate": 0.1, "flag": False, "mode": None})
        self.assertEqual(points[-1], {"rate": 0.2, "flag": True, "mode": "x"})
        self.assertEqual(list(grid_search({})), [{}])

    def test_lhs_stratifies_every_linear_and_log_marginal(self):
        count = 37
        space = {"x": {"min": -2, "max": 4}, "y": {"min": 1e-5, "max": 1e-1, "scale": "log"}}
        points = latin_hypercube(space, count, seed=12)
        x = [(p["x"] + 2) / 6 for p in points]
        y = [(math.log(p["y"]) - math.log(1e-5)) / math.log(1e4) for p in points]
        for values in (x, y):
            self.assertEqual(sorted(int(v * count) for v in values), list(range(count)))
        self.assertEqual(points, latin_hypercube(space, count, seed=12))
        self.assertNotEqual(points, latin_hypercube(space, count, seed=13))

    def test_integer_categorical_sampling_and_rng_isolation(self):
        np.random.seed(11)
        expected = np.random.random(3)
        np.random.seed(11)
        points = latin_hypercube({"n": {"min": 1, "max": 4, "type": "int"}, "b": [False, True]}, 40)
        self.assertEqual([sum(p["n"] == k for p in points) for k in range(1, 5)], [10] * 4)
        self.assertEqual(sum(p["b"] for p in points), 20)
        np.testing.assert_equal(np.random.random(3), expected)

    def test_monte_carlo_is_reproducible_and_covers_distribution(self):
        space = {"x": {"min": 0, "max": 1}}
        points = monte_carlo(space, 10000, seed=3)
        self.assertEqual(points, monte_carlo(space, 10000, seed=3))
        values = np.array([p["x"] for p in points])
        self.assertAlmostEqual(values.mean(), 0.5, delta=0.015)
        self.assertAlmostEqual(values.var(), 1 / 12, delta=0.005)

    def test_invalid_designs_fail(self):
        for space in ({"a": []}, {"a": [1, 1]}, {"a": [float("nan")]},
                      {"a": {"min": 0, "max": 1, "scale": "log"}},
                      {"a": {"min": 0.5, "max": 3, "type": "int"}},
                      {"a": {"min": 1, "max": 1}}, {"a": {"min": 0, "max": 1, "typo": 0}}):
            with self.subTest(space=space), self.assertRaises(ValueError):
                latin_hypercube(space, 10)
        with self.assertRaises(ValueError):
            list(grid_search({"a": {"min": 0, "max": 1}}))
        with self.assertRaises(ValueError):
            monte_carlo({}, True)


class UQTests(unittest.TestCase):
    def test_wilson_known_reference_and_extremes(self):
        np.testing.assert_allclose(wilson_interval(5, 10), [0.2365930905, 0.7634069095], atol=1e-9)
        self.assertAlmostEqual(wilson_interval(0, 10)[0], 0)
        self.assertGreater(wilson_interval(0, 10)[1], 0.27)
        self.assertAlmostEqual(wilson_interval(10, 10)[1], 1)
        self.assertLess(wilson_interval(10, 10)[0], 0.73)

    def test_bootstrap_exact_two_point_distribution(self):
        result = bootstrap_mean([0, 2], resamples=5000, seed=3)
        self.assertEqual(result["mean"], 1)
        self.assertAlmostEqual(result["std"], math.sqrt(2))
        self.assertEqual(result["mean_ci"], [0, 2])
        self.assertEqual(result, bootstrap_mean([0, 2], resamples=5000, seed=3))
        self.assertIsNone(bootstrap_mean([5])["mean_ci"])

    def test_hierarchy_weights_policies_equally_and_preserves_run_variance(self):
        result = hierarchical_mean([[0] * 100, [10] * 2], resamples=1000)
        self.assertEqual(result["mean"], 5)
        self.assertEqual(result["mean_ci"], [0, 10])
        self.assertIsNone(hierarchical_mean([[0, 10]])["mean_ci"])

    def test_summary_keeps_single_policy_and_cross_seed_uncertainty_separate(self):
        runs = [{"seed": 1, "episode_returns": [0] * 20, "wins": 0},
                {"seed": 2, "episode_returns": [10] * 20, "wins": 20}]
        result = summarize_runs(runs, resamples=1000)
        self.assertEqual(result["win_rate_across_runs"]["mean_ci"], [0, 1])
        single = summarize_runs(runs[:1], resamples=10)
        self.assertIsNone(single["win_rate_across_runs"]["mean_ci"])
        self.assertGreater(single["per_run"][0]["win_rate_ci"][1], 0)
        with self.assertRaises(ValueError):
            summarize_runs([runs[0], runs[0]])

    def test_invalid_measurements_and_options(self):
        for values in ([], [float("inf")], [[1, 2]]):
            with self.assertRaises(ValueError):
                bootstrap_mean(values)
        for wins, n in ((-1, 10), (11, 10), (1.0, 10), (0, 0)):
            with self.assertRaises(ValueError):
                wilson_interval(wins, n)
        with self.assertRaises(ValueError):
            bootstrap_mean([1, 2], confidence=1)
        with self.assertRaises(ValueError):
            hierarchical_mean([[1, 2]], resamples=0)


@dataclass
class FakeArgs:
    total_timesteps: int = 20480
    num_steps: int = 128
    num_minibatches: int = 4
    learning_rate: float = 0.00025


class RunnerTests(unittest.TestCase):
    def test_plan_repeats_seeds_deduplicates_and_randomizes_reproducibly(self):
        config = {"method": "lhs", "samples": 20, "parameters": {"learning_rate": [0.1, 0.2]},
                  "training_seeds": [1, 2], "design_seed": 4}
        plan = build_plan(config)
        self.assertEqual(len(plan["trials"]), 4)
        self.assertEqual(plan["duplicate_configurations_removed"], 18)
        self.assertEqual(plan, build_plan(config))
        self.assertEqual(len({t["id"] for t in plan["trials"]}), 4)

    def test_budget_ownership_and_invalid_config_checks(self):
        for config in (
            {"parameters": {"seed": [1]}}, {"fixed": {"num_envs": 2}},
            {"fixed": {"total_timesteps": 20000}}, {"fixed": {"num_minibatches": 128}},
            {"training_seeds": [1, 1]}, {"trial_timeout_seconds": -1},
            {"fixed": {"x": 1}, "parameters": {"x": [1, 2]}},
            {"parameters": {"x": list(range(1000)), "y": list(range(1000))}, "max_trials": 10},
            {"deterministic": "false"}, {"unknown": 0},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                build_plan(config)

    def test_command_handles_booleans_paths_and_fresh_weights(self):
        trial = build_plan({"fixed": {"cuda": False, "anneal_lr": True}})["trials"][0]
        command = training_command("ppo_lstm", trial, "folder with spaces/model.pt", "trial")
        self.assertIn("--no-cuda", command)
        self.assertIn("--anneal-lr", command)
        self.assertIn("folder with spaces/model.pt", command)
        self.assertNotIn("--restore-model-path", command)

    def test_lock_excludes_second_runner_and_releases_after_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock"
            with self.assertRaisesRegex(RuntimeError, "test error"):
                with live_lock(path):
                    with self.assertRaisesRegex(RuntimeError, "lock exists"):
                        with live_lock(path):
                            self.fail("second owner acquired lock")
                    raise RuntimeError("test error")
            self.assertFalse(path.exists())

    def test_trial_orders_training_then_evaluation_and_cleans_bridge(self):
        plan = build_plan({"training_seeds": [1], "eval_episodes": 2})
        trial = plan["trials"][0]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            phases = []

            def child(command, logfile, deadline):
                phases.append(command[2])
                if command[2] == "Sweeps.evaluate":
                    self.assertIn(str(directory / "model.pt"), command)
                    write_json(directory / "evaluation.json", {"episode_returns": [1, 2], "wins": 1})

            with mock.patch("Automation.processes.start_process") as start, \
                    mock.patch("Automation.processes.wait_until_ready"), \
                    mock.patch("Automation.processes.terminate_process") as terminate, \
                    mock.patch("Sweeps.runner._run_child", side_effect=child):
                result = run_trial(plan, trial, directory, {"bridge_command": ["bridge"]})
            self.assertEqual(phases, ["Training.PPO_server", "Sweeps.evaluate"])
            self.assertEqual(result["wins"], 1)
            self.assertEqual(result["training_steps"], 20480)
            self.assertEqual(read_json(directory / "result.json")["status"], "complete")
            terminate.assert_called_once_with(start.return_value, "sweep bridge")

    def test_trial_failure_is_not_recorded_as_low_score(self):
        plan = build_plan({"training_seeds": [1]})
        with tempfile.TemporaryDirectory() as temp, \
                mock.patch("Automation.processes.start_process"), \
                mock.patch("Automation.processes.wait_until_ready"), \
                mock.patch("Automation.processes.terminate_process") as terminate, \
                mock.patch("Sweeps.runner._run_child", side_effect=TimeoutError("no game")):
            with self.assertRaises(TimeoutError):
                run_trial(plan, plan["trials"][0], temp, {"bridge_command": ["bridge"]})
            self.assertFalse((Path(temp) / "result.json").exists())
            self.assertEqual(read_json(Path(temp) / "status.json")["status"], "failed")
            terminate.assert_called_once()

    def test_resume_only_runs_missing_trials_and_rejects_changed_plan(self):
        plan = build_plan({"training_seeds": [1, 2]})
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            write_json(output / "plan.json", plan)
            complete = plan["trials"][0]
            write_json(output / complete["id"] / "result.json", {
                **complete, "status": "complete", "algorithm": "ppo", "total_seconds": 1,
                "episode_returns": [1] * 20, "wins": 1,
            })
            with mock.patch("Automation.processes.load_config", return_value={}), \
                    mock.patch("Sweeps.runner.live_lock", side_effect=contextlib.nullcontext), \
                    mock.patch("Sweeps.runner.run_trial") as trial, \
                    mock.patch("importlib.import_module", return_value=SimpleNamespace(Args=FakeArgs)):
                run_plan(plan, output)
                self.assertEqual(trial.call_count, 1)
                self.assertEqual(trial.call_args.args[1], plan["trials"][1])
                with self.assertRaisesRegex(ValueError, "different plan"):
                    run_plan(build_plan({"training_seeds": [3]}), output)

    def test_cli_plan_and_partial_report(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            config = output / "config.json"
            write_json(config, {"training_seeds": [1, 2], "eval_episodes": 2})
            with mock.patch("Sweeps.__main__.run_plan") as live:
                self.assertEqual(main(["plan", str(config), "--output", str(output / "plan.json")]), 0)
                live.assert_not_called()
            plan = read_json(output / "plan.json")
            trial = plan["trials"][0]
            write_json(output / trial["id"] / "result.json", {
                **trial, "status": "complete", "episode_returns": [1, 3], "wins": 1,
                "total_seconds": 2, "algorithm": "ppo",
            })
            summary = report(output, resamples=20)
            self.assertEqual(summary["completed_trials"], 1)
            self.assertEqual(summary["configurations"][0]["planned_seeds"], 2)
            self.assertIsNone(summary["configurations"][0]["return_across_runs"]["mean_ci"])

    def test_child_timeout_reaps_process(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(subprocess.TimeoutExpired):
                _run_child([sys.executable, "-c", "import time; time.sleep(10)"],
                           Path(temp) / "log.txt", time.monotonic() + 0.1)

    def test_core_imports_without_optional_tuning_dependencies(self):
        code = """
import sys
class BlockOptional:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pufferlib', 'gpytorch', 'scipy', 'sklearn', 'torch'}:
            raise AssertionError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from Sweeps import latin_hypercube, summarize_runs
from Sweeps.runner import build_plan
assert len(latin_hypercube({'x': [1, 2]}, 2)) == 2
assert len(build_plan({})['trials']) == 3
"""
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    def test_example_configs_have_expected_trial_counts(self):
        root = Path(__file__).resolve().parents[1] / "Sweeps" / "examples"
        for name, expected in (("grid", 81), ("lhs", 36), ("uq", 5)):
            with self.subTest(name=name):
                self.assertEqual(len(build_plan(read_json(root / f"{name}.json"))["trials"]), expected)


if __name__ == "__main__":
    unittest.main()
