"""Start the bridge and configured training algorithm."""

import argparse
from typing import Sequence

from Automation.processes import (
    DEFAULT_CONFIG,
    load_config,
    normalize_command,
    run_stack,
)


RESTORABLE_ALGORITHMS = {"ppo", "ppo_lstm"}


def resolve_training_command(
    config: dict,
) -> tuple[str, str | Sequence[object]]:
    """Resolve the command for the configured training algorithm."""
    algorithm = str(config.get("rl_algorithm", "ppo_lstm")).lower()
    algorithm_command = config.get(f"{algorithm}_command")
    if not algorithm_command:
        raise ValueError(f"No command configured for rl_algorithm={algorithm!r}.")

    restore_model_path = config.get("restore_model_path")
    if restore_model_path:
        if algorithm not in RESTORABLE_ALGORITHMS:
            raise ValueError(
                "restore_model_path is only supported for ppo and ppo_lstm."
            )
        command_arguments = (
            normalize_command(algorithm_command)
            if isinstance(algorithm_command, str)
            else [str(argument) for argument in algorithm_command]
        )
        algorithm_command = [
            *command_arguments,
            "--restore-model-path",
            str(restore_model_path),
        ]
    return algorithm, algorithm_command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)

    config = load_config(args.config)
    algorithm, algorithm_command = resolve_training_command(config)

    return run_stack(
        config,
        algorithm_command,
        algorithm,
        start_tensorboard=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
