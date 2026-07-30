"""Start the bridge and a model-inference command."""

import argparse
from typing import Sequence

from Automation.processes import DEFAULT_CONFIG, load_config, run_stack


def resolve_inference_command(
    config: dict,
    override: str | Sequence[object] | None = None,
) -> str | Sequence[object]:
    """Resolve an override or an explicit algorithm-specific command."""
    if override:
        return override

    # Keep custom/smoke configurations with one explicit command supported.
    configured_command = config.get("inference_command")
    if configured_command:
        return configured_command

    algorithm = str(config.get("inference_algorithm", "")).lower()
    if not algorithm:
        raise ValueError(
            "inference_algorithm or inference_command must be configured."
        )

    algorithm_command = config.get(f"{algorithm}_inference_command")
    if not algorithm_command:
        raise ValueError(
            "No explicit inference command configured for "
            f"inference_algorithm={algorithm!r}."
        )
    return algorithm_command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--command",
        help="Override inference_command with a shell-style command string.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    inference_command = resolve_inference_command(config, args.command)

    return run_stack(
        config,
        inference_command,
        "inference",
        start_tensorboard=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
