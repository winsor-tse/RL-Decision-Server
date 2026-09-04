"""Shared process supervision for training, inference, and recording."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("automation_config.yaml")
PYTHON_COMMANDS = {"python", "python.exe", "python3", "py"}
DEFAULT_INTERRUPT_GRACE_SECONDS = 30.0


def load_config(config_path: str | Path) -> dict:
    """Load a launcher configuration from YAML."""
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError("Automation config must contain a YAML mapping.")
    return config


def normalize_command(command: str | Sequence[object]) -> list[str]:
    """Return a subprocess argument list using the active Python interpreter."""
    if isinstance(command, str):
        arguments = shlex.split(command, posix=os.name != "nt")
        if os.name == "nt":
            arguments = [
                argument[1:-1]
                if len(argument) >= 2
                and argument[0] == argument[-1]
                and argument[0] in {"'", '"'}
                else argument
                for argument in arguments
            ]
    elif isinstance(command, Sequence):
        arguments = [str(argument) for argument in command]
    else:
        raise TypeError("Process commands must be strings or YAML lists.")

    if not arguments:
        raise ValueError("Process command cannot be empty.")
    if arguments[0].lower() in PYTHON_COMMANDS:
        arguments[0] = sys.executable
    return arguments


def start_process(
    command: str | Sequence[object],
    *,
    capture_output: bool = False,
) -> subprocess.Popen:
    """Start a configured command from the repository root."""
    arguments = normalize_command(command)
    print(f"Starting: {subprocess.list2cmdline(arguments)}", flush=True)
    options = {
        "args": arguments,
        "cwd": ROOT_DIR,
        "shell": False,
    }
    if capture_output:
        options.update(
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    return subprocess.Popen(**options)


def terminate_process(process: subprocess.Popen | None, name: str) -> None:
    """Stop a child process, escalating to kill after five seconds."""
    if process is None or process.poll() is not None:
        return
    print(f"Stopping {name}...", flush=True)
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def wait_for_interrupted_child(
    process: subprocess.Popen | None,
    name: str,
    timeout_seconds: float = DEFAULT_INTERRUPT_GRACE_SECONDS,
) -> int | None:
    """Give a Ctrl+C-aware child time to flush data before forced cleanup."""

    if process is None:
        return None
    return_code = process.poll()
    if return_code is not None:
        return int(return_code)

    try:
        return int(process.wait(timeout=timeout_seconds))
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        return None


def wait_until_ready(
    process: subprocess.Popen,
    ready_signal: str,
    timeout_seconds: float,
) -> None:
    """Forward process output and wait for its configured readiness line."""
    ready_event = threading.Event()

    def stream_output() -> None:
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ""):
            print(line, end="")
            if ready_signal in line:
                ready_event.set()

    output_thread = threading.Thread(target=stream_output, daemon=True)
    output_thread.start()

    deadline = time.monotonic() + timeout_seconds
    while not ready_event.is_set():
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Bridge exited with code {return_code} before becoming ready."
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Bridge did not emit {ready_signal!r} within "
                f"{timeout_seconds:g} seconds."
            )
        ready_event.wait(timeout=min(0.1, max(0, deadline - time.monotonic())))


def run_process(
    command: str | Sequence[object],
    name: str,
) -> int:
    """Run one supervised process without starting the live-game bridge."""

    process = None
    try:
        process = start_process(command)
        return process.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        terminate_process(process, name)


def run_stack(
    config: dict,
    child_command: str | Sequence[object],
    child_name: str,
    *,
    start_tensorboard: bool,
) -> int:
    """Run bridge, optional TensorBoard, and one supervised child process."""
    bridge_command = config.get("bridge_command")
    if not bridge_command:
        raise ValueError("bridge_command is required.")

    bridge_process = None
    tensorboard_process = None
    child_process = None
    try:
        bridge_process = start_process(bridge_command, capture_output=True)
        wait_until_ready(
            bridge_process,
            str(config.get("bridge_ready_signal", "WebSocket bridge listening")),
            float(config.get("ready_timeout_seconds", 30)),
        )

        if start_tensorboard and config.get("start_tensorboard", False):
            tensorboard_command = config.get("tensorboard_command")
            if not tensorboard_command:
                raise ValueError(
                    "start_tensorboard is enabled but tensorboard_command is missing."
                )
            tensorboard_process = start_process(tensorboard_command)

        print(f"Bridge ready. Starting {child_name}.", flush=True)
        child_process = start_process(child_command)
        return child_process.wait()
    except KeyboardInterrupt:
        child_return_code = wait_for_interrupted_child(
            child_process,
            child_name,
            float(
                config.get(
                    "interrupt_grace_seconds",
                    DEFAULT_INTERRUPT_GRACE_SECONDS,
                )
            ),
        )
        return child_return_code if child_return_code is not None else 130
    finally:
        terminate_process(child_process, child_name)
        terminate_process(tensorboard_process, "TensorBoard")
        terminate_process(bridge_process, "bridge")
