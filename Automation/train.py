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
    *,
    resume_checkpoint_path: str | None = None,
    total_timesteps: int | None = None,
    stop_after_timesteps: int | None = None,
    checkpoint_interval: int | None = None,
) -> tuple[str, str | Sequence[object]]:
    """Resolve the command for the configured training algorithm."""
    algorithm = str(config.get("rl_algorithm", "ppo_lstm")).lower()
    algorithm_command = config.get(f"{algorithm}_command")
    if not algorithm_command:
        raise ValueError(f"No command configured for rl_algorithm={algorithm!r}.")

    restore_model_path = config.get("restore_model_path")
    resume_checkpoint_path = (
        resume_checkpoint_path or config.get("resume_checkpoint_path")
    )
    if stop_after_timesteps is None:
        stop_after_timesteps = config.get("stop_after_timesteps")
    if total_timesteps is None:
        total_timesteps = config.get("total_timesteps")
    if checkpoint_interval is None:
        checkpoint_interval = config.get("checkpoint_interval")
    if stop_after_timesteps is not None and int(stop_after_timesteps) < 0:
        raise ValueError("stop_after_timesteps must be nonnegative.")
    if checkpoint_interval is not None and int(checkpoint_interval) < 0:
        raise ValueError("checkpoint_interval must be nonnegative.")
    if total_timesteps is not None and int(total_timesteps) <= 0:
        raise ValueError("total_timesteps must be greater than zero.")
    if restore_model_path and resume_checkpoint_path:
        raise ValueError(
            "restore_model_path and resume_checkpoint_path cannot be used together."
        )
    if resume_checkpoint_path and algorithm not in RESTORABLE_ALGORITHMS:
        raise ValueError(
            "resume_checkpoint_path is only supported for ppo and ppo_lstm."
        )
    if (
        total_timesteps is not None
        or stop_after_timesteps is not None
        or checkpoint_interval is not None
    ) and algorithm not in RESTORABLE_ALGORITHMS:
        raise ValueError(
            "checkpoint controls are only supported for ppo and ppo_lstm."
        )
    extra_arguments: list[str] = []
    if resume_checkpoint_path:
        extra_arguments.extend(
            ["--resume-checkpoint-path", str(resume_checkpoint_path)]
        )
    if total_timesteps is not None:
        extra_arguments.extend(["--total-timesteps", str(total_timesteps)])
    if stop_after_timesteps is not None:
        extra_arguments.extend(
            ["--stop-after-timesteps", str(stop_after_timesteps)]
        )
    if checkpoint_interval is not None:
        extra_arguments.extend(
            ["--checkpoint-interval", str(checkpoint_interval)]
        )
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
    if extra_arguments:
        command_arguments = (
            normalize_command(algorithm_command)
            if isinstance(algorithm_command, str)
            else [str(argument) for argument in algorithm_command]
        )
        algorithm_command = [*command_arguments, *extra_arguments]
    return algorithm, algorithm_command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume-checkpoint-path")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--stop-after-timesteps", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    algorithm, algorithm_command = resolve_training_command(
        config,
        resume_checkpoint_path=args.resume_checkpoint_path,
        total_timesteps=args.total_timesteps,
        stop_after_timesteps=args.stop_after_timesteps,
        checkpoint_interval=args.checkpoint_interval,
    )

    return run_stack(
        config,
        algorithm_command,
        algorithm,
        start_tensorboard=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
