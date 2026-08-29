"""Start the WebSocket bridge and the offline player recorder."""

import argparse
from collections.abc import Sequence

from Automation.processes import DEFAULT_CONFIG, load_config, run_stack


def resolve_recorder_command(
    config: dict,
    override: str | Sequence[object] | None = None,
) -> str | Sequence[object]:
    """Resolve an override or the configured offline recorder command."""
    if override:
        return override

    recorder_command = config.get("recorder_command")
    if not recorder_command:
        raise ValueError("recorder_command is required.")
    return recorder_command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Start the bridge, then run the offline player recorder."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--command",
        help="Override recorder_command with a shell-style command string.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    recorder_command = resolve_recorder_command(config, args.command)
    return run_stack(
        config,
        recorder_command,
        "offline recorder",
        start_tensorboard=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
