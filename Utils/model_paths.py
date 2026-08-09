from pathlib import Path


def training_checkpoint_path(
    run_directory: str | Path,
    configured_path: str | None,
    checkpoint_name: str,
) -> Path:
    """Use an explicit path or place the checkpoint in its run directory."""
    if configured_path:
        return Path(configured_path)
    return Path(run_directory) / checkpoint_name


def inference_checkpoint_path(
    configured_path: str | None,
    checkpoint_name: str,
    runs_directory: str | Path = "runs",
) -> Path:
    """Use an explicit checkpoint or find the newest matching run checkpoint."""
    if configured_path:
        checkpoint_path = Path(configured_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {checkpoint_path}"
            )
        return checkpoint_path

    runs_path = Path(runs_directory)
    candidates = [
        path
        for path in runs_path.glob(f"*/{checkpoint_name}")
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No {checkpoint_name} checkpoint found under "
            f"{runs_path / '<run_name>'}. Train the matching agent first or "
            "pass --model-path explicitly."
        )
    return max(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
