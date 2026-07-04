import argparse
import subprocess
import sys
import threading
from pathlib import Path

try:
    from Automation.Training_stack import (
        DEFAULT_CONFIG,
        ROOT_DIR,
        load_simple_yaml,
        stream_bridge_output,
        terminate_process,
    )
except ModuleNotFoundError:
    automation_dir = Path(__file__).resolve().parent
    if str(automation_dir) not in sys.path:
        sys.path.append(str(automation_dir))
    from Training_stack import (
        DEFAULT_CONFIG,
        ROOT_DIR,
        load_simple_yaml,
        stream_bridge_output,
        terminate_process,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--command", default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_simple_yaml(config_path)

    bridge_command = config["bridge_command"]
    ready_signal = config.get("bridge_ready_signal", "WebSocket bridge listening")
    ready_timeout = int(config.get("ready_timeout_seconds", 30))
    inference_command = args.command or config.get("inference_command")

    if not inference_command:
        raise ValueError("No inference command configured. Set inference_command or pass --command.")

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

    print(f"Bridge ready. Starting inference: {inference_command}")
    inference_process = subprocess.Popen(
        inference_command,
        cwd=ROOT_DIR,
        shell=True,
    )

    try:
        return inference_process.wait()
    except KeyboardInterrupt:
        terminate_process(inference_process, "inference")
        return 130
    finally:
        terminate_process(bridge_process, "bridge")


if __name__ == "__main__":
    sys.exit(main())
