"""Reproducible plans and a single-owner live PPO experiment runner."""

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import time

from Sweeps.search import grid_search, latin_hypercube, monte_carlo, positive_integer
from Sweeps.uq import summarize_runs

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = {"ppo": "PPO_server", "ppo_lstm": "PPO_lstm_server"}
RESERVED = {"seed", "model_path", "exp_name", "save_model", "restore_model_path",
            "batch_size", "minibatch_size", "num_iterations"}


def write_json(path, value):
    """Replace a JSON file atomically, rejecting nonfinite measurements."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_result(path, plan, trial):
    result = read_json(path)
    if (result.get("status") != "complete" or result.get("id") != trial["id"]
            or result.get("parameters") != trial["parameters"]
            or result.get("seed") != trial["seed"]
            or result.get("algorithm") != plan["algorithm"]
            or len(result.get("episode_returns", [])) != plan["eval_episodes"]):
        raise ValueError(f"result does not match planned trial: {path}")
    summarize_runs([result], resamples=2)
    seconds = result.get("total_seconds")
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"invalid elapsed time in {path}")
    return result


def build_plan(config):
    """Resolve a finite design without importing a trainer or connecting to ZMQ."""
    unknown = set(config) - {"algorithm", "method", "parameters", "fixed", "samples",
                             "design_seed", "training_seeds", "eval_episodes",
                             "deterministic", "trial_timeout_seconds", "max_trials"}
    if unknown:
        raise ValueError(f"unknown sweep fields: {sorted(unknown)}")
    algorithm = config.get("algorithm", "ppo")
    if algorithm not in ALGORITHMS:
        raise ValueError("algorithm must be ppo or ppo_lstm")
    method = config.get("method", "grid")
    design_seed = config.get("design_seed", 0)
    if isinstance(design_seed, bool) or not isinstance(design_seed, int) or design_seed < 0:
        raise ValueError("design_seed must be a nonnegative integer")
    seeds = config.get("training_seeds", [1, 2, 3])
    if (not isinstance(seeds, list) or not seeds
            or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("training_seeds must be distinct integers in [0, 2**32)")
    space, fixed = config.get("parameters", {}), config.get("fixed", {})
    if not isinstance(space, dict) or not isinstance(fixed, dict):
        raise ValueError("parameters and fixed must be dictionaries")
    if set(space) & set(fixed):
        raise ValueError("a parameter cannot be both fixed and swept")
    for key in set(space) | set(fixed):
        if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", key) or key in RESERVED:
            raise ValueError(f"unsupported or runner-owned parameter: {key}")
    maximum = positive_integer(config.get("max_trials", 100000), "max_trials")
    if method == "grid":
        design = grid_search(space)
        size = math.prod(len(v["values"] if isinstance(v, dict) else v) for v in space.values())
    elif method in ("lhs", "random", "monte_carlo"):
        size = positive_integer(config.get("samples", 12), "samples")
        if size * len(seeds) > maximum:
            raise ValueError("design exceeds max_trials; narrow it or raise the explicit limit")
        sampler = latin_hypercube if method == "lhs" else monte_carlo
        design = sampler(space, size, design_seed)
    else:
        raise ValueError("method must be grid, lhs, random, or monte_carlo")
    if size * len(seeds) > maximum:
        raise ValueError("design exceeds max_trials; narrow it or raise the explicit limit")
    episodes = positive_integer(config.get("eval_episodes", 20), "eval_episodes")
    deterministic = config.get("deterministic", True)
    if not isinstance(deterministic, bool):
        raise ValueError("deterministic must be a boolean")
    timeout = config.get("trial_timeout_seconds", 21600)
    if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("trial_timeout_seconds must be finite and positive")
    trials, seen = [], set()
    duplicates = 0
    for point in design:
        parameters = {"total_timesteps": 20480, "num_steps": 128,
                      "num_minibatches": 4, **fixed, **point}
        _validate_parameters(parameters)
        signature = json.dumps(parameters, sort_keys=True, allow_nan=False)
        if signature in seen:
            duplicates += 1
            continue
        seen.add(signature)
        config_id = hashlib.sha256(signature.encode()).hexdigest()[:16]
        for seed in seeds:
            trials.append({"id": f"{config_id}-seed{seed}", "config_id": config_id,
                           "seed": seed, "parameters": parameters.copy()})
    random.Random(design_seed).shuffle(trials)
    return {"version": 1, "algorithm": algorithm, "method": method,
            "design_seed": design_seed, "training_seeds": seeds,
            "eval_episodes": episodes, "deterministic": deterministic,
            "trial_timeout_seconds": timeout, "duplicate_configurations_removed": duplicates,
            "trials": trials}


def _validate_parameters(parameters):
    for name, value in parameters.items():
        if value is not None and not isinstance(value, (str, bool, int, float)):
            raise ValueError(f"{name} must be a scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if isinstance(parameters.get("num_envs", 1), bool) or parameters.get("num_envs", 1) != 1:
        raise ValueError("only one external environment is supported")
    for key in ("total_timesteps", "num_steps", "num_minibatches", "update_epochs", "metrics_frequency"):
        if key in parameters:
            positive_integer(parameters[key], key)
    steps, batches = parameters["num_steps"], parameters["num_minibatches"]
    if steps % batches or steps // batches < 2:
        raise ValueError("num_steps must divide evenly into minibatches of at least two steps")
    if parameters["total_timesteps"] % steps:
        raise ValueError("total_timesteps must be a multiple of num_steps for comparable actual budgets")


def training_command(algorithm, trial, checkpoint, experiment):
    """Build an argument list; never execute a shell or load another trial's weights."""
    params = {**trial["parameters"], "seed": trial["seed"], "num_envs": 1,
              "save_model": True, "model_path": str(checkpoint), "exp_name": experiment}
    arguments = [sys.executable, "-m", f"Training.{ALGORITHMS[algorithm]}"]
    for name, value in params.items():
        flag = name.replace("_", "-")
        if isinstance(value, bool):
            arguments.append(f"--{'' if value else 'no-'}{flag}")
        elif value is not None:
            arguments.extend([f"--{flag}", str(value)])
    return arguments


@contextmanager
def live_lock(path=None):
    """Exclude all other sweep runners sharing this repo, even with different outputs."""
    import os

    path = Path(path) if path is not None else ROOT / "runs" / ".sweep-live.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(f"Live sweep lock exists: {path}. Check its PID; remove only if its runner has stopped.") from error
    try:
        with stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink()


def _run_child(command, logfile, deadline):
    from Automation.processes import terminate_process

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("trial time budget exhausted")
    with Path(logfile).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, shell=False)
        try:
            code = process.wait(timeout=remaining)
            if code:
                raise RuntimeError(f"Child exited with {code}; see {logfile}")
        finally:
            terminate_process(process, "sweep child")


def run_trial(plan, trial, directory, bridge_config):
    """Keep one bridge alive while trainer and evaluator take turns owning ZMQ."""
    from Automation.processes import start_process, terminate_process, wait_until_ready

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "model.pt"
    started = time.monotonic()
    deadline = started + plan["trial_timeout_seconds"]
    bridge = None
    write_json(directory / "status.json", {"status": "running", "trial": trial})
    try:
        bridge = start_process(bridge_config["bridge_command"], capture_output=True)
        wait_until_ready(bridge, str(bridge_config.get("bridge_ready_signal", "WebSocket bridge listening")),
                         min(float(bridge_config.get("ready_timeout_seconds", 30)), plan["trial_timeout_seconds"]))
        training_started = time.monotonic()
        command = training_command(plan["algorithm"], trial, checkpoint, f"sweep_{trial['id']}")
        _run_child(command, directory / "training.log", deadline)
        training_seconds = time.monotonic() - training_started
        evaluation_path = directory / "evaluation.json"
        command = [sys.executable, "-m", "Sweeps.evaluate", "--algorithm", plan["algorithm"],
                   "--checkpoint", str(checkpoint), "--output", str(evaluation_path),
                   "--episodes", str(plan["eval_episodes"]), "--seed", str(trial["seed"])]
        if not plan["deterministic"]:
            command.append("--stochastic")
        if trial["parameters"].get("cuda") is False:
            command.append("--cpu")
        evaluation_started = time.monotonic()
        _run_child(command, directory / "evaluation.log", deadline)
        evaluation = read_json(evaluation_path)
        if len(evaluation["episode_returns"]) != plan["eval_episodes"]:
            raise ValueError("evaluation did not return the planned number of episodes")
        summarize_runs([{**evaluation, "seed": trial["seed"]}], resamples=2)
        result = {**trial, **evaluation, "status": "complete", "algorithm": plan["algorithm"],
                  "checkpoint": str(checkpoint), "training_seconds": training_seconds,
                  "evaluation_seconds": time.monotonic() - evaluation_started,
                  "total_seconds": time.monotonic() - started,
                  "training_steps": trial["parameters"]["total_timesteps"]}
        write_json(directory / "result.json", result)
        write_json(directory / "status.json", {"status": "complete"})
        return result
    except BaseException as error:
        write_json(directory / "status.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        terminate_process(bridge, "sweep bridge")


def run_plan(plan, output, *, automation_config=None):
    """Resume complete trials only. Failed/interrupted trials restart from scratch."""
    from dataclasses import fields
    from importlib import import_module
    from Automation.processes import DEFAULT_CONFIG, load_config

    trainer_args = import_module(f"Training.{ALGORITHMS[plan['algorithm']]}").Args
    defaults = {field.name: field.default for field in fields(trainer_args)}
    allowed = set(defaults)
    for trial in plan["trials"]:
        if set(trial["parameters"]) - allowed:
            raise ValueError(f"unknown trainer parameters: {set(trial['parameters']) - allowed}")
        for name, value in trial["parameters"].items():
            default = defaults[name]
            if isinstance(default, bool) and not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
            if type(default) is int and (type(value) is not int):
                raise ValueError(f"{name} must be an integer")
            if type(default) is float and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError(f"{name} must be numeric")
    bridge_config = load_config(automation_config or DEFAULT_CONFIG)
    output = Path(output).resolve()
    with live_lock():
        output.mkdir(parents=True, exist_ok=True)
        plan_path = output / "plan.json"
        if plan_path.exists() and read_json(plan_path) != plan:
            raise ValueError("output contains a different plan; select a new output directory")
        write_json(plan_path, plan)
        for i, trial in enumerate(plan["trials"], 1):
            directory = output / trial["id"]
            result_path = directory / "result.json"
            if result_path.exists():
                _read_result(result_path, plan, trial)
                print(f"[{i}/{len(plan['trials'])}] Already complete: {trial['id']}", flush=True)
                continue
            print(f"[{i}/{len(plan['trials'])}] Starting: {trial['id']} (logs: {directory})", flush=True)
            run_trial(plan, trial, directory, bridge_config)


def report(output, *, confidence=0.95, resamples=2000, seed=0):
    """Group completed trials by configuration; never silently pool configurations."""
    output = Path(output)
    plan = read_json(output / "plan.json")
    groups = {}
    completed = 0
    for trial in plan["trials"]:
        path = output / trial["id"] / "result.json"
        if not path.exists():
            continue
        result = _read_result(path, plan, trial)
        groups.setdefault(trial["config_id"], []).append(result)
        completed += 1
    summaries = []
    for config_id, runs in groups.items():
        summaries.append({
            "config_id": config_id, "parameters": runs[0]["parameters"],
            "complete_seeds": len(runs), "planned_seeds": len(plan["training_seeds"]),
            "total_seconds": sum(run["total_seconds"] for run in runs),
            **summarize_runs(runs, confidence=confidence, resamples=resamples, seed=seed),
        })
    return {"planned_trials": len(plan["trials"]), "completed_trials": completed,
            "confidence": confidence, "configurations": summaries}
