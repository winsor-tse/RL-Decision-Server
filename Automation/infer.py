"""Start the bridge and a model-inference command."""

import argparse
from pathlib import Path
from typing import Sequence

from Automation.processes import DEFAULT_CONFIG, ROOT_DIR, load_config, run_stack


def find_latest_checkpoint(runs_directory: Path = ROOT_DIR / "runs") -> Path:
    """Return the most recently modified PyTorch checkpoint."""
    checkpoints = list(runs_directory.rglob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No .pt checkpoints found under {runs_directory}. "
            "Train a model or configure inference_model_path."
        )
    return max(checkpoints, key=lambda checkpoint: checkpoint.stat().st_mtime)


def resolve_inference_command(
    config: dict,
    override: str | Sequence[object] | None = None,
    *,
    runs_directory: Path = ROOT_DIR / "runs",
) -> str | Sequence[object]:
    """Resolve an override, configured command, or DQN checkpoint command."""
    if override:
        return override

    configured_command = config.get("inference_command")
    if configured_command:
        return configured_command

    configured_model_path = config.get("inference_model_path", "latest")
    if str(configured_model_path).lower() == "latest":
        model_path = find_latest_checkpoint(runs_directory)
    else:
        model_path = Path(str(configured_model_path))
        if not model_path.is_absolute():
            model_path = ROOT_DIR / model_path
        if not model_path.is_file():
            raise FileNotFoundError(f"Inference checkpoint does not exist: {model_path}")

    return [
        "python",
        "-m",
        "Inference.dqn_eval",
        "--model-path",
        str(model_path),
    ]


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
