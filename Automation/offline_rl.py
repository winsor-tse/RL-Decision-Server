"""Train or evaluate an any-percent behavior-cloning model."""

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


def run_offline_rl(
    config: dict,
    command: str | Sequence[object],
    *,
    mode: str,
) -> int:
    """Run dataset analysis directly or live evaluation with the bridge."""

    if mode == "live":
        return run_stack(
            config,
            command,
            "any-percent BC live evaluation",
            start_tensorboard=False,
        )
    process_name = (
        "any-percent BC training"
        if mode == "train"
        else "any-percent BC dataset evaluation"
    )
    return run_process(command, process_name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
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
    if args.mode in {"dataset", "live"} and not args.checkpoint_path:
        raise ValueError(
            "--checkpoint-path is required for dataset and live evaluation."
        )

    config = load_config(args.config) if args.mode == "live" else {}
    if args.mode == "train":
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
    return run_offline_rl(config, command, mode=args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
