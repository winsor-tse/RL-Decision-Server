import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("automation_config.yaml")


def load_simple_yaml(config_path):
    config = {}
    with open(config_path, "r", encoding="utf-8") as config_file:
        for raw_line in config_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            config[key.strip()] = parse_value(value.strip())
    return config


def parse_value(value):
    value = value.split("#", 1)[0].strip()
    value = value.strip('"').strip("'")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        return value


def stream_bridge_output(process, ready_signal, ready_event):
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        if ready_signal in line:
            ready_event.set()


def terminate_process(process, name):
    if process.poll() is not None:
        return
    print(f"Stopping {name}...")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def start_optional_process(config, enabled_key, command_key, name):
    if not config.get(enabled_key, False):
        return None

    command = config.get(command_key)
    if not command:
        raise ValueError(f"{enabled_key} is enabled but {command_key} is not configured.")

    print(f"Starting {name}: {command}")
    return subprocess.Popen(
        command,
        cwd=ROOT_DIR,
        shell=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_simple_yaml(config_path)

    algorithm = str(config.get("rl_algorithm", "dqn")).lower()
    bridge_command = config["bridge_command"]
    ready_signal = config.get("bridge_ready_signal", "WebSocket bridge listening")
    ready_timeout = int(config.get("ready_timeout_seconds", 30))
    algorithm_command = config.get(f"{algorithm}_command")

    if not algorithm_command:
        raise ValueError(f"No command configured for rl_algorithm={algorithm!r}.")

    print(f"Starting bridge: {bridge_command}")
    bridge_process = subprocess.Popen(
        bridge_command,
        cwd=ROOT_DIR,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    ready_event = threading.Event()
    bridge_thread = threading.Thread(
        target=stream_bridge_output,
        args=(bridge_process, ready_signal, ready_event),
        daemon=True,
    )
    bridge_thread.start()

    if not ready_event.wait(timeout=ready_timeout):
        terminate_process(bridge_process, "bridge")
        raise TimeoutError(f"Bridge did not emit ready signal within {ready_timeout} seconds.")

    tensorboard_process = start_optional_process(
        config,
        "start_tensorboard",
        "tensorboard_command",
        "TensorBoard",
    )

    print(f"Bridge ready. Starting {algorithm}: {algorithm_command}")
    algorithm_process = subprocess.Popen(
        algorithm_command,
        cwd=ROOT_DIR,
        shell=True,
    )

    try:
        return algorithm_process.wait()
    except KeyboardInterrupt:
        terminate_process(algorithm_process, algorithm)
        return 130
    finally:
        if tensorboard_process is not None:
            terminate_process(tensorboard_process, "TensorBoard")
        terminate_process(bridge_process, "bridge")


if __name__ == "__main__":
    sys.exit(main())
