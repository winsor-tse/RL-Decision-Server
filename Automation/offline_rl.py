"""Train BC or AWAC on Minari data, or evaluate a BC model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from Automation.processes import (
    DEFAULT_CONFIG,
    load_config,
    run_process,
    run_stack,
)


def build_training_command(
    *,
    dataset_id: str,
    update_steps: int,
    buffer_size: int,
    batch_size: int,
    top_fraction: float,
    gamma: float,
    eval_every: int,
    normalize_state: bool,
    checkpoints_path: str,
) -> list[str]:
    """Build the offline Minari behavior-cloning training command."""

    return [
        "python",
        "-m",
        "Offline.any_percent_bc",
        "--dataset-id",
        dataset_id,
        "--update-steps",
        str(update_steps),
        "--buffer-size",
        str(buffer_size),
        "--batch-size",
        str(batch_size),
        "--top-fraction",
        str(top_fraction),
        "--gamma",
        str(gamma),
        "--eval-every",
        str(eval_every),
        "--checkpoints-path",
        checkpoints_path,
        "--normalize-state" if normalize_state else "--no-normalize-state",
    ]


def build_awac_training_command(
    *,
    dataset_id: str,
    update_steps: int,
    online_iterations: int,
    buffer_size: int,
    batch_size: int,
    gamma: float,
    normalize_state: bool,
    checkpoints_path: str,
    device: str,
    hidden_dim: int,
    learning_rate: float,
    tau: float,
    awac_lambda: float,
) -> list[str]:
    """Map shared launcher options to the discrete AWAC trainer."""

    command = [
        "python", "-m", "Offline.awac",
        "--dataset-id", dataset_id,
        "--offline-iterations", str(update_steps),
        "--online-iterations", str(online_iterations),
        "--buffer-size", str(buffer_size),
        "--batch-size", str(batch_size),
        "--gamma", str(gamma),
        "--checkpoints-path", checkpoints_path,
        "--hidden-dim", str(hidden_dim),
        "--learning-rate", str(learning_rate),
        "--tau", str(tau),
        "--awac-lambda", str(awac_lambda),
        "--normalize-state" if normalize_state else "--no-normalize-state",
    ]
    # AWAC selects CUDA/CPU itself when no device override is supplied.
    if device != "auto":
        command.extend(["--device", device])
    return command


def build_evaluation_command(
    *,
    mode: str,
    checkpoint_path: str,
    dataset_id: str,
    eval_episodes: int,
    top_fraction: float,
    gamma: float,
    device: str,
    normalize_state: bool,
    output_csv: str | None,
) -> list[str]:
    """Build the shared any-percent BC evaluation command."""

    command = [
        "python",
        "-m",
        "Inference.any_percent_bc_eval",
        "--mode",
        mode,
        "--checkpoint-path",
        checkpoint_path,
        "--dataset-id",
        dataset_id,
        "--eval-episodes",
        str(eval_episodes),
        "--top-fraction",
        str(top_fraction),
        "--gamma",
        str(gamma),
        "--device",
        device,
        "--normalize-state" if normalize_state else "--no-normalize-state",
    ]
    if output_csv:
        command.extend(["--output-csv", output_csv])
    return command


def build_awac_evaluation_command(
    *, checkpoint_path: str, eval_episodes: int, device: str,
) -> list[str]:
    """Build the live Env16 AWAC evaluation command."""

    return [
        "python", "-m", "Inference.awac_eval",
        "--checkpoint-path", checkpoint_path,
        "--eval-episodes", str(eval_episodes),
        "--device", device,
    ]


def run_offline_rl(
    config: dict,
    command: str | Sequence[object],
    *,
    mode: str,
    algorithm: str = "bc",
    online_iterations: int = 0,
) -> int:
    """Start the bridge only for live evaluation or AWAC fine-tuning."""

    if algorithm == "awac" and mode == "train" and online_iterations > 0:
        return run_stack(
            config, command, "AWAC offline training and live fine-tuning",
            start_tensorboard=True,
        )
    if mode == "live":
        return run_stack(
            config,
            command,
            f"{'AWAC' if algorithm == 'awac' else 'any-percent BC'} live evaluation",
            start_tensorboard=False,
        )
    process_name = (
        f"{'AWAC' if algorithm == 'awac' else 'any-percent BC'} training"
        if mode == "train"
        else "any-percent BC dataset evaluation"
    )
    return run_process(command, process_name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--algorithm", choices=("bc", "awac"), default="bc")
    parser.add_argument(
        "--mode",
        choices=("train", "dataset", "live"),
        default="train",
    )
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--dataset-id", default="env16/BC-v0")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--top-fraction", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--update-steps", type=int, default=1_000_000)
    parser.add_argument("--online-iterations", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=5e-3)
    parser.add_argument("--awac-lambda", type=float, default=1.0)
    parser.add_argument("--buffer-size", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--checkpoints-path", default="runs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--normalize-state",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-csv")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.algorithm == "awac" and args.mode == "dataset":
        raise ValueError("AWAC supports train and live modes; dataset evaluation requires BC.")
    if args.online_iterations < 0:
        raise ValueError("--online-iterations must be nonnegative.")
    if args.online_iterations and (args.algorithm != "awac" or args.mode != "train"):
        raise ValueError("--online-iterations requires --algorithm awac --mode train.")
    if args.algorithm == "awac" and args.mode == "train" and args.top_fraction != 1.0:
        raise ValueError("AWAC uses all demonstrations; --top-fraction must be 1.0.")
    if args.mode in {"dataset", "live"} and not args.checkpoint_path:
        raise ValueError(
            "--checkpoint-path is required for dataset and live evaluation."
        )

    needs_bridge = args.mode == "live" or args.online_iterations > 0
    config = load_config(args.config) if needs_bridge else {}
    if args.mode == "live" and args.algorithm == "awac":
        command = build_awac_evaluation_command(
            checkpoint_path=args.checkpoint_path,
            eval_episodes=args.eval_episodes,
            device=args.device,
        )
    elif args.mode == "train" and args.algorithm == "awac":
        command = build_awac_training_command(
            dataset_id=args.dataset_id,
            update_steps=args.update_steps,
            online_iterations=args.online_iterations,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            normalize_state=args.normalize_state,
            checkpoints_path=args.checkpoints_path,
            device=args.device,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            tau=args.tau,
            awac_lambda=args.awac_lambda,
        )
    elif args.mode == "train":
        command = build_training_command(
            dataset_id=args.dataset_id,
            update_steps=args.update_steps,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            top_fraction=args.top_fraction,
            gamma=args.gamma,
            eval_every=args.eval_every,
            normalize_state=args.normalize_state,
            checkpoints_path=args.checkpoints_path,
        )
    else:
        command = build_evaluation_command(
            mode=args.mode,
            checkpoint_path=args.checkpoint_path,
            dataset_id=args.dataset_id,
            eval_episodes=args.eval_episodes,
            top_fraction=args.top_fraction,
            gamma=args.gamma,
            device=args.device,
            normalize_state=args.normalize_state,
            output_csv=args.output_csv,
        )
    return run_offline_rl(
        config, command, mode=args.mode, algorithm=args.algorithm,
        online_iterations=args.online_iterations,
    )


if __name__ == "__main__":
    raise SystemExit(main())
